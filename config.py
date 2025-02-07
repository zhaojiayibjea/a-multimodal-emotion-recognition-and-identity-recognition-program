import os
import logging
import pymysql

"""全局通用配置"""

# 是否调用摄像头和麦克风处理实时画面
enable_realtime = False
# 摄像头设备索引（enable_realtime = False时无效）
camera_id = 0


# 存放输入视频文件的路径（enable_realtime = True时无效）
input_dir = os.path.join('test','videos')
# 被分割后的视频的时长（秒）
segment_duration = 6
# 临时文件保存位置
tmp_file_path = 'tmp'
# 测试文件地址
test_file_path = 'test'
# 是否删除临时文件
enable_delete_tmp_file = True
# 数据库配置
database_config = {
    'user': '',  # MySQL 用户名
    'password': '',  # MySQL 密码
    'host': '',  # 数据库服务器地址
    'database': '',  # 要连接的数据库名称
    'charset': '', # 字符编码
    'cursorclass': pymysql.cursors.DictCursor, # 游标
    'port': 3306 # Mysql 服务端口
}
db = None # 数据库连接器，不用修改


"""音频处理配置"""

# 模型文件存储地址
model_path = 'models'
# 是否使用在线的模型文件
enable_online_model = True
# 用于声纹识别的声纹库储存地址
voiceprint_library_path = 'speakers'
# 声纹识别的最低相似度。0 ~ 1 之间的浮点数，数值越大要求越严格
audio_thr = 0.3
# 是否将语音转换成文字并保存成 txt 文件
enable_write_txt = False

"""视频处理配置"""

# 人脸识别的最远距离（不相似度）。0 ~ 1 之间的浮点数，数值越大要求越松
video_thr = 0.6
# 用于人脸识别的人脸库地址
facelib_path = 'facelib'
# 帧的截取间隔（秒）
interval = 5

def connect_database():
    try:
        global db
        db = pymysql.connect(**database_config)
        logging.info("数据库连接成功")
    except pymysql.Error as err:
        logging.error(f"数据库连接失败: {err}")


def disconnect_database():
    try:
        db.close()
        logging.info("数据库已经断开连接")
    except pymysql.Error as err:
        logging.error(f"数据库断开连接失败: {err}")



