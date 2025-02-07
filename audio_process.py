import os
import re
import queue
import shutil

import ffmpeg
import logging
import pymysql
import threading
from funasr import AutoModel
from modelscope.pipelines import pipeline
from datetime import timedelta, datetime
from pydub import AudioSegment

import config

# 创建队列，用于线程间通信
spk_txt_queue = queue.Queue()
# 音频合并队列
audio_concat_queue = queue.Queue()
# 身份识别队列
identity_queue = queue.Queue()
# 情感识别队列
emotion_recognize_queue = queue.Queue()

# 存储输入音频路径
audio = ''
# 指定临时文件保存路径
save_path = os.path.join(config.tmp_file_path,'audio_process_dir')

# 指定声纹库路径
voiceprint_library_path = config.voiceprint_library_path
# 指定模型文件存储路径
model_path = config.model_path

# 离线版本
asr_model_path = os.path.join(model_path, 'speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
vad_model_path = os.path.join(model_path, 'speech_fsmn_vad_zh-cn-16k-common-pytorch')
punc_model_path = os.path.join(model_path, 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
spk_model_path = os.path.join(model_path, 'speech_campplus_sv_zh-cn_16k-common')
ser_model_path = os.path.join(model_path, 'SenseVoiceSmall')

# 在线版本
if config.enable_online_model:
    asr_model_path = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model_path = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model_path = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    spk_model_path = "cam++"
    ser_model_path = "iic/SenseVoiceSmall"

ngpu = 1
device = "cuda:0"
ncpu = 4

# ASR 模型
model = AutoModel(
    model=asr_model_path,
    vad_model=vad_model_path,
    punc_model=punc_model_path,
    spk_model=spk_model_path,
    ngpu=ngpu,
    ncpu=ncpu,
    device=device,
    disable_pbar=True,
    disable_log=True,
    disable_update=True
)

speaker_recognizer = pipeline(
    task='speaker-verification',
    model=spk_model_path,
    # model_revision='v1.0.0'
)

emotion_recognizer = AutoModel(
    model=ser_model_path,
    vad_model=vad_model_path,
    vad_kwargs={"max_single_segment_time": 30000},
    ngpu=ngpu,
    ncpu=ncpu,
    device=device,
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)


def to_date(milliseconds):
    # 将时间戳转换为SRT格式的时间
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"


# 转写获取时间戳，根据时间戳进行切分，然后根据 spk id 进行分类
# audio: 音频
# return 切分后按照 spk id 的地址

def trans():
    audio_name = os.path.splitext(os.path.basename(audio))[0]
    _, audio_extension = os.path.splitext(audio)
    logging.info(f'正在分割音频 {audio}')
    speaker_audios = {}  # 每个说话人作为 key，value 为列表，列表中为当前说话人对应的每个音频片段
    # 音频预处理
    try:
        audio_bytes, _ = (
            ffmpeg.input(audio, threads=0, hwaccel='cuda')
            .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        res = model.generate(input=audio_bytes, batch_size_s=300, is_final=True, sentence_timestamp=True)
        rec_result = res[0]
        asr_result_text = rec_result['text']
        if asr_result_text != '':
            sentences = []
            for sentence in rec_result["sentence_info"]:
                start = to_date(sentence["start"])
                end = to_date(sentence["end"])
                if sentences and sentence["spk"] == sentences[-1]["spk"]:
                    sentences[-1]["text"] += "" + sentence["text"]
                    sentences[-1]["end"] = end
                else:
                    sentences.append(
                        {"text": sentence["text"], "start": start, "end": end, "spk": sentence["spk"]}
                    )

            # 剪切音频或视频片段
            i = 0
            for stn in sentences:
                stn_txt = stn['text']
                start = stn['start']
                end = stn['end']
                spk = stn['spk']

                # 根据文件名和 spk 创建目录
                date = datetime.now().strftime("%Y-%m-%d")
                final_save_path = os.path.join(save_path, date, audio_name, str(spk))
                os.makedirs(final_save_path, exist_ok=True)
                final_save_file = os.path.join(final_save_path, str(i) + '.mp3')
                spk_txt_path = os.path.join(save_path, date, audio_name)
                spk_txt_file = os.path.join(spk_txt_path, f'spk{spk}.txt')
                spk_txt_queue.put({'spk_txt_file': spk_txt_file, 'spk_txt': stn_txt, 'start': start, 'end': end})
                i += 1
                try:
                    (
                        ffmpeg.input(audio, threads=0, ss=start, to=end, hwaccel='cuda')
                        .output(final_save_file, codec='libmp3lame', preset='medium', ar=16000, ac=1)
                        .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                             capture_stderr=True)
                    )
                except ffmpeg.Error as e:
                    logging.error(f"剪切音频发生错误，错误信息：{e}")
                # 记录说话人和对应的音频片段，用于合并音频片段
                if spk not in speaker_audios:
                    speaker_audios[spk] = []  # 列表中存储音频片段
                speaker_audios[spk].append({'file': final_save_file, 'audio_name': audio_name})

            logging.info(f'{audio} 切分完成')

            # 存入合并队列

            audio_concat_queue.put(speaker_audios)

        else:
            logging.error("没有转写结果")
    except Exception as e:
        logging.error(f"转写异常：{e}")
    audio_concat_queue.put(None)  # 结束信号
    spk_txt_queue.put(None)  # 结束信号

def write_txt():
    while True:
        item = spk_txt_queue.get()
        if item is None:  # 检查结束信号
            break
        spk_txt_file = item['spk_txt_file']
        spk_txt = item['spk_txt']
        spk_start = item['start']
        spk_end = item['end']
        dir_path = os.path.dirname(spk_txt_file)
        os.makedirs(dir_path, exist_ok=True)
        with open(spk_txt_file, 'a', encoding='utf-8') as f:
            f.write(f"{spk_start} --> {spk_end}\n{spk_txt}\n\n")
        spk_txt_queue.task_done()


def audio_concat_worker():
    while True:
        speaker_audios_tmp = audio_concat_queue.get()
        if speaker_audios_tmp is None: break
        for spk, audio_segments in speaker_audios_tmp.items():
            # 合并每个说话人的音频片段
            audio_name = audio_segments[0]['audio_name']
            output_file = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d"), audio_name, f"{spk}.mp3")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            inputs = [seg['file'] for seg in audio_segments]
            concat_audio = AudioSegment.from_file(inputs[0])
            for i in range(1, len(inputs)):
                concat_audio = concat_audio + AudioSegment.from_file(inputs[i])
            concat_audio.export(output_file, format="wav")
            logging.info(f"已将 {spk} 的音频合并到 {output_file}")
            identity_queue.put(output_file)
    identity_queue.put(None)


def identity_worker():
    while True:
        audio_path = identity_queue.get()
        if audio_path is None:
            break
        # 识别身份
        speaker_id = ''
        maxn = config.audio_thr  # 设置最低相似度
        for filename in os.listdir(voiceprint_library_path):  # 读取已知声纹库
            filepath = os.path.join(voiceprint_library_path, filename)
            result = speaker_recognizer([filepath, audio_path])['score']

            if result > maxn:
                maxn = result
                speaker_id = os.path.splitext(filename)[0]
        if speaker_id == '':
            logging.error(f"音频 {audio_path} 声纹识别失败\n")
            continue
        else:
            logging.info(f"音频 {audio_path} 声纹识别成功。speaker_id = {speaker_id}\n")
        emotion_recognize_queue.put([speaker_id, audio_path])
    emotion_recognize_queue.put(None)


# 情感识别
def emotion_recognize_worker():
    while True:
        item = emotion_recognize_queue.get()
        if item is None: break
        speaker_id, audio_path = item
        res = emotion_recognizer.generate(
            input=audio_path,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
        )
        text = res[0]['text']
        parts = re.split(r'(<\|.*?\|>)', text)
        emotion_recognize_result = re.sub(r'<\|(.*?)\|>', r'\1', parts[3])

        # 计算情感得分
        emotion_score = 0
        if emotion_recognize_result == 'HAPPY' or emotion_recognize_result == 'SURPRISED':
            emotion_score = 1
        elif emotion_recognize_result == 'ANGRY' or emotion_recognize_result == 'SAD' or emotion_recognize_result == 'FEARFUL' or emotion_recognize_result == 'DISGUSTED':
            emotion_score = -1

        # 在数据库中找到speaker_id, 并在对应的值上加上emotion_score
        db = config.db
        try:
            with db.cursor() as cursor:

                query = 'SELECT score FROM scores WHERE id = %s'
                cursor.execute(query, (speaker_id,))
                query_result = cursor.fetchone()
                if query_result is not None:
                    current_score = query_result['score']  # 获取当前的 score
                    new_score = current_score + emotion_score  # 计算新的 score
                    # 更新 score
                    update_query = "UPDATE scores SET score = %s WHERE id = %s"
                    cursor.execute(update_query, (new_score, speaker_id))
                    logging.info(f"id {speaker_id} 的得分已从 {current_score} 更新为 {new_score}")
                else:
                    logging.error(f"数据库中没有录入 {speaker_id} 的信息")
            db.commit()

        except pymysql.Error as err:
            logging.error(f"数据库错误: {err}")
            config.disconnect_database()
            config.connect_database()


def audio_worker(input_audio_path):
    global audio
    audio = input_audio_path

    # 启动转写进程
    trans_thread = threading.Thread(target=trans)
    trans_thread.daemon = True
    trans_thread.start()
    logging.info("转写进程启动")

    # 创建音频合并进程
    audio_concat_thread = threading.Thread(target=audio_concat_worker)
    audio_concat_thread.daemon = True
    audio_concat_thread.start()
    logging.info("音频合并进程启动")

    # 创建身份识别进程
    identity_thread = threading.Thread(target=identity_worker)
    identity_thread.daemon = True
    identity_thread.start()
    logging.info("身份识别进程启动")

    # 创建情感识别进程
    emotion_recognition_thread = threading.Thread(target=emotion_recognize_worker)
    emotion_recognition_thread.daemon = True
    emotion_recognition_thread.start()
    logging.info("情感识别进程启动")

    # 创建保存文本文件的进程（可选）
    if config.enable_write_txt:
        write_txt_thread = threading.Thread(target=write_txt)
        write_txt_thread.daemon = True
        write_txt_thread.start()

    trans_thread.join()
    logging.info("转写进程结束")
    audio_concat_thread.join()
    logging.info("音频合并进程结束")
    identity_thread.join()
    logging.info("身份识别进程结束")
    emotion_recognition_thread.join()
    logging.info("情感识别进程结束")
    if config.enable_write_txt:
        write_txt_thread.join()
    if config.enable_delete_tmp_file:
        try:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        except Exception as e:
            logging.info(f"删除临时文件失败: {e}")

# 以下代码仅在测试单个文件时生效

if __name__ == '__main__':
    audio_worker("tmp/videos/HomeWithKids1.mp4/HomeWithKids1_part01.mp4")
