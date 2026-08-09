
   # Usage Guide

2025/2/7 By ZJY

## Description

A multimodal emotion and identity recognition program. This project can capture video from a camera + microphone or video files, split the video into segments of 10 minutes (customizable), and process them sequentially.

Specifically, each segment first undergoes speaker (voiceprint) recognition and voice emotion recognition, producing quantified scores that are updated in the database. Subsequently, each video segment undergoes face recognition and facial emotion recognition at a frequency of one frame every five seconds (customizable). The recognition results are also synchronized to the database. Audio processing is based on [FunASR](https://github.com/modelscope/FunASR), and video processing is based on [Deepface](https://github.com/serengil/deepface).

## Quick Start

1. It is recommended to use Python **3.10** before running this project. The author has only tested the program on Python 3.10; other versions may cause runtime issues. It is recommended to use a virtual environment such as venv.

2. Clone the project to your local machine:

    ```bash
    git clone https://github.com/zhaojiayibjea/a-multimodal-emotion-recognition-and-identity-recognition-program.git
    ```

3. Install Python dependencies:

    Open a terminal in the project's root directory and run

    ```bash
    pip install -r requirements.txt
    ```

4. If your computer has an NVIDIA GPU with CUDA support, installing CUDA is highly recommended as it can significantly improve runtime performance.

5. Configure your database, input file directory, speaker library directory, face library directory, and other settings in the `config.py` file.

6. Determine the ID of the detection target and add a row in the database:

```sql
INSERT INTO scores (id, score) VALUES (你的编号, 0)
```

7. If you do not enable the `realtime detection` feature in the configuration file, please add video files to the input directory.

8. Collect voice and face samples of the detection target and add them to the speaker library and face library. Please use the target's ID as the filename.

9. Verify the runtime environment requirements and run the `main.py` file.

## Runtime Results and Debugging

Since the frontend part of this project is not yet complete, you can currently only view the program's output by observing changes in the database values.

You can view the results by running:

```sql
SELECT * FROM scores
```

or other SQL commands.

If you want to optimize the identity recognition results, you can adjust the tolerance threshold in the `config.py` file for debugging.

You can run `audio_process.py` and `video_process.py` individually to save time. See the following code block in both files for debugging methods:

```python
if __name__ == '__main__':
```

below the comment.

## Project File Overview

The core functionalities of this project are implemented through the following files:

`config.py`: Configuration file for this program. You can adjust the settings according to your needs. The author has detailed the function of each item in the comments. All paths mentioned in this document can be modified here.

`main.py`: **The entry point of the program**. Run this file to execute the entire project. It handles video acquisition and segmentation, calls the audio processing function `audio_worker()` and video processing function `video_worker()`, and establishes database connections and log configurations.

`realtime.py`: Activated when the real-time feature is enabled. It handles real-time microphone recording, periodically chunks and saves audio, simultaneously displays the camera feed, and shows the microphone volume and a countdown for file saving via a bar overlay on the camera view.

`audio_process.py`: Implements audio processing functionality. It exposes the `audio_worker()` function as an interface. Due to the high computational load, the author implemented it using multithreading.

`video_process.py`: Implements video processing functionality. It exposes the `video_worker()` function as an interface.

## Acknowledgements

This project utilizes and references several open-source projects. See: [LICENSE](LICENSE.md)
