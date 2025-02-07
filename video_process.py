import cv2
import os
import logging
import pymysql
from deepface import DeepFace

import config


def identity_worker(img):
    dfs = DeepFace.find(
        img_path=img,
        db_path=config.facelib_path,
        silent=True,
        enforce_detection=False,
        threshold=config.video_thr
    )

    return dfs[0]['identity']


def emotion_recognize_worker(img):
    # 面部情绪识别
    results = DeepFace.analyze(
        img_path=img,
        actions=['emotion'],
        enforce_detection=False,
    )

    # 遍历每一张检测到的面孔
    for idx in enumerate(results):
        emotion = results[idx[0]]['emotion']
        region = results[idx[0]]['region']

        # 计算得分
        num = (emotion['happy'] - emotion['sad'] - emotion['fear'] - emotion['disgust'] - emotion['angry']) / 100

        # 将脸部裁剪出来，并进行人脸识别
        x, y, w, h, l, r = region.values()
        cropped_img = img[y:y + h, x:x + w]
        id = identity_worker(cropped_img)

        # id 是 pandas dataframe
        if id.isna().all():
            logging.error("人脸识别失败")
            continue
        id = os.path.splitext(os.path.split(id[0])[1])[0]
        logging.info(f"识别到人脸。id = {id}")

        # 在数据库中给识别到的人更新分数
        db = config.db
        try:
            with db.cursor() as cursor:
                query = 'SELECT score FROM scores WHERE id = %s'
                cursor.execute(query, (id,))
                query_result = cursor.fetchone()
                if query_result is not None:
                    current_score = query_result['score']  # 获取当前的 score
                    new_score = current_score + num  # 计算新的 score

                    # 更新 score
                    update_query = "UPDATE scores SET score = %s WHERE id = %s"
                    cursor.execute(update_query, (new_score, id))
                    logging.info(f"id {id} 的得分已从 {current_score} 更新为 {new_score}")
                else:
                    logging.error(f"数据库中没有录入 {id} 的信息")
            db.commit()

        except pymysql.Error as err:
            logging.error(f"数据库错误: {err}")
            config.disconnect_database()
            config.connect_database()


def video_worker(input_video_path):
    interval = config.interval  # 截取间隔（秒）
    cap = cv2.VideoCapture(input_video_path)  # 打开视频文件
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    frame_interval = int(fps * interval)  # 计算帧间隔
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 每间隔 interval 秒保存一帧
        if count % frame_interval == 0:
            emotion_recognize_worker(frame)
        count += 1

    cap.release()


# 以下代码仅在测试单个文件时生效

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    config.connect_database()
    img = cv2.imread('test/img_1.png')
    emotion_recognize_worker(img)
    config.disconnect_database()
