from audio_process import audio_worker
from video_process import video_worker
from realtime import realtime_worker
import shutil
import pymysql
import logging
import os
import ffmpeg

import config

# 设置日志信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)  # 使用 ffprobe 解析视频元数据
        duration_str = probe['format']['duration']  # 从元数据的 'format' 中提取时长（字符串类型）
        duration = float(duration_str)  # 转换为浮点数（秒）
        return duration
    except ffmpeg.Error as e:
        print(f"FFmpeg错误: {e.stderr.decode()}")
        return None
    except KeyError:
        print("未找到视频时长信息")
        return None


def split_and_process_video(input_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    segment_duration = config.segment_duration  # 设置分割间隔
    duration = get_video_duration(input_path)
    # 分割并处理每个片段
    for i, start_time in enumerate(range(0, int(duration), segment_duration)):
        end_time = min(start_time + segment_duration, duration)

        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            output_dir,
            f"{base_name}_part{i + 1:02d}.mp4"
        )
        # 分割音频
        (
            ffmpeg
            .input(input_path, ss=start_time, to=end_time)
            .output(output_path, c="copy")
            .run(overwrite_output=True, )
        )

        audio_worker(output_path)
        video_worker(output_path)
    if config.enable_delete_tmp_file:
        try:
            shutil.rmtree(config.tmp_file_path)
            os.makedirs(config.tmp_file_path)
        except Exception as e:
            logging.info(f"删除临时文件失败: {e}")


def batch_process(input_dir, output_base_dir):
    # 批量处理目录下的所有视频文件
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in video_exts):
            input_path = os.path.join(input_dir, filename)
            output_dir = os.path.join(output_base_dir, filename)
            output_dir = os.path.splitext(output_dir)[0]

            split_and_process_video(
                input_path,
                output_dir,
            )



if __name__ == "__main__":
    config.connect_database()
    if config.enable_realtime:
        realtime_worker()
    else:
        batch_process(
            input_dir=config.input_dir,
            output_base_dir=os.path.join(config.tmp_file_path, 'videos')
        )
    config.disconnect_database()
