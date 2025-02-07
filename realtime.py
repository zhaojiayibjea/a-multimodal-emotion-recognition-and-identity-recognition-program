import os

import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from threading import Thread, Lock
from queue import Queue
import config
from audio_process import audio_worker
from video_process import emotion_recognize_worker

# 配置参数
sample_rate = 16000
channels = 1
duration = config.segment_duration  # 保存间隔（秒）
buffer_time = 0.1  # 音频缓冲时间（秒）
camera_id = config.camera_id  # 摄像头设备索引

# 全局变量
audio_buffer = []
current_volume = 0
time_remaining = duration
last_save_time = time.time()
lock = Lock()
save_queue = Queue()
audio_process_queue = Queue()
frame_process_queue = Queue()

# 音频回调函数
def audio_callback(indata, frames, time, status):
    global current_volume, audio_buffer
    volume = np.sqrt(np.mean(indata ** 2))
    current_volume = min(int(volume * 500), 100)  # 调整缩放系数
    audio_buffer.extend(indata.copy())


# 音频保存线程
def audio_saver():
    os.makedirs(os.path.join(config.tmp_file_path,"audio"), exist_ok=True)
    while True:
        data = save_queue.get()
        if data is None: break
        filename = time.strftime("recording_%Y%m%d_%H%M%S.wav")
        filename = os.path.join(config.tmp_file_path, "audio", filename)
        sf.write(filename, data, sample_rate)
        print(f"Saved {filename}")
        audio_process_queue.put(filename)
    audio_process_queue.put(None)

# 音频处理线程
def audio_processing():
    while True:
        item = audio_process_queue.get()
        if item is None: break
        audio_worker(item)
        os.remove(item)

# 帧处理线程
def frame_processing():
    while True:
        item = frame_process_queue.get()
        if item is None: break
        emotion_recognize_worker(item)


# 视频处理函数
def video_processing():
    global time_remaining
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow('Monitor', cv2.WINDOW_NORMAL)

    interval = config.interval  # 截取间隔（秒）
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_interval = int(fps * interval)  # 计算帧间隔
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 每间隔 interval 秒保存一帧
        if count % frame_interval == 0:
            frame_process_queue.put(frame)
        count += 1

        # 获取当前状态
        with lock:
            vol = current_volume
            tr = time_remaining

        # 绘制音量条
        cv2.rectangle(frame, (20, 20), (20 + vol * 2, 40), (0, 255, 0), -1)
        cv2.putText(frame, f'Vol: {vol}%', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 绘制倒计时
        mins, secs = divmod(tr, 60)
        timer_str = f"{mins:02d}:{secs:02d}"
        cv2.putText(frame, timer_str, (frame.shape[1] - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame_process_queue.put(None)
            break

    cap.release()
    cv2.destroyAllWindows()


# 主控制函数
def realtime_worker():
    global audio_buffer, time_remaining, last_save_time

    # 启动音频保存线程
    saver_thread = Thread(target=audio_saver)
    saver_thread.start()

    # 启动音频流
    stream = sd.InputStream(callback=audio_callback,
                            samplerate=sample_rate,
                            channels=channels,
                            blocksize=int(sample_rate * buffer_time))
    stream.start()

    # 启动视频线程
    video_thread = Thread(target=video_processing)
    video_thread.start()

    # 启动音频处理线程
    audio_processing_thread = Thread(target=audio_processing)
    audio_processing_thread.start()

    # 启动帧处理线程
    frame_processing_thread = Thread(target=frame_processing)
    frame_processing_thread.start()

    try:
        while True:
            current_time = time.time()
            elapsed = current_time - last_save_time
            time_remaining = max(duration - int(elapsed), 0)

            if elapsed >= duration:
                # 保存音频
                with lock:
                    if len(audio_buffer) > 0:
                        save_data = np.array(audio_buffer)
                        save_queue.put(save_data)
                        audio_buffer = []

                    last_save_time = current_time
                    time_remaining = duration

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        save_queue.put(None)
        frame_process_queue.put(None)
        audio_process_queue.put(None)
        saver_thread.join()
        video_thread.join()
        audio_processing_thread.join()
        frame_processing_thread.join()



if __name__ == "__main__":
    realtime_worker()