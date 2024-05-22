import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)

import cv2
import numpy as np
from collections import deque
import time
import traceback


def camera_fn(shared_dict, capture_looptime, camera_rotation_180, capture_arr_ctypes, capture_timestamp, camera_idx, shape_capture):
    try :
        time_buffer = deque(maxlen=20)
        capture_arr = np.frombuffer(capture_arr_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        
        cap = cv2.VideoCapture(camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape_capture[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape_capture[0])
        same_width = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) == shape_capture[1])
        same_height = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == shape_capture[0])
        resizing = False if same_width and same_height else True
        min_fps = 30 # 파일인 경우
        spf = 1 / max(min_fps, cap.get(cv2.CAP_PROP_FPS)) 
        fail_count = 0
        shared_dict['ready_camera'] = True
        while True :
            s_time = time.time()
            ret, frame = cap.read()
            if not ret :
                fail_count += 1
                if fail_count > 5:
                    cap.release()
                    cap = cv2.VideoCapture(camera_idx)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape_capture[1])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape_capture[0])
                    same_width = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) == shape_capture[1])
                    same_height = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == shape_capture[0])
                    resizing = False if same_width and same_height else True
                    spf = 1 / max(min_fps, cap.get(cv2.CAP_PROP_FPS)) 
                    fail_count = 0
                time.sleep(0.01)
                continue 
            else :
                fail_count = 0
            
            if camera_rotation_180 :
                frame = frame[::-1, ::-1]
        
            if resizing:
                frame = cv2.resize(frame, shape_capture[:2][::-1])

            # 공유 변수에 저장
            capture_arr[:, :, :] = frame
            capture_timestamp.value = s_time
            capture_looptime.value = sum(time_buffer) / max(len(time_buffer),1)
            
            # 파일이 경우, fps가 너무 높게 나오는 것을 방지
            while True:
                if time.time() - s_time > spf :
                    break

            # 버퍼
            time_buffer.append(time.time() - s_time)
            
    except Exception as e:
        error_message = traceback.format_exc()
        shared_dict["error"] = "camera.py \n" + error_message