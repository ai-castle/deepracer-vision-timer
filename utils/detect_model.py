import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)
import hparams as hp
import config as cf

from util_functions import get_off_fence_tf_arr, get_off_track_tf_arr, get_arr_shape, download_model, modify_array, rectangle_line_intersect, error_detection_handler
import traceback
from ultralytics import YOLO
from collections import deque
import time
import numpy as np
import cv2
import torch

detect_model_path = download_model(cf.model_version, cf.model_type)
shape_capture, shape_detect = get_arr_shape()
start_pt_arr = np.array(cf.starting_line_endpoints).flatten()
out_wheel_num = hp.out_wheel_num
decision_point = hp.decision_point
shape_yolo_output = hp.shape_yolo_output
detect_confidence_min= hp.detect_confidence_min
off_track_tf_arr1, off_track_tf_arr2 = get_off_track_tf_arr()
off_fence_tf_arr1, off_fence_tf_arr2 = get_off_fence_tf_arr()
device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_model_fn(shared_dict, detect_model_looptime, observe_on, start_tf, out_tf, capture_arr_ctypes, capture_arr2_ctypes, detect_best_arr_ctypes, detect_model_timestamp, camera_timestamp):
    try :
        time_buffer = deque(maxlen=20)
        capture_arr = np.frombuffer(capture_arr_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        capture_arr2 = np.frombuffer(capture_arr2_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        detect_best_arr = np.frombuffer(detect_best_arr_ctypes.get_obj(), dtype=np.float32)
        shape_ratio = (np.array(shape_capture) / np.array(shape_detect))[:2]
        model_detect = YOLO(detect_model_path).to(device)
        resized_image_test = cv2.resize(capture_arr, shape_detect[:2][::-1])
        model_detect(resized_image_test, verbose=False)  # cold start
        
        camera_timestamp_value_prev = 0
        shared_dict['ready_detect'] = True
        while True :
            while True :
                s_time = time.time()
                camera_timestamp_value = camera_timestamp.value
                if camera_timestamp_value > camera_timestamp_value_prev :
                    camera_timestamp_value_prev = camera_timestamp_value
                    break
            capture_arr_copy = capture_arr.copy()
            resized_image = cv2.resize(capture_arr_copy, shape_detect[:2][::-1])
            modify_array(resized_image, off_fence_tf_arr2)
            
            # 임시
            predict_detect_best = np.zeros(shape_yolo_output, dtype=np.float32)
            out_tf_value = 0
            start_tf_value = 0
            
            # 검증
            if observe_on.value == 1 :
                results = model_detect(resized_image, verbose=False)[0]
                predict_boxes = results.boxes.data
                predict_keypoints = results.keypoints.data
                
                if predict_boxes.shape[0] > 0 :
                    predict_boxes_best = predict_boxes[0].cpu().numpy()
                    predict_best_confidence = predict_boxes_best[4]
                    predict_best_class = predict_boxes_best[5]  # 0: deepracer
                    if predict_best_confidence > detect_confidence_min :
                        predict_boxes_best[0] = min(shape_capture[1]-1, predict_boxes_best[0]*shape_ratio[1])
                        predict_boxes_best[1] = min(shape_capture[0]-1, predict_boxes_best[1]*shape_ratio[0])
                        predict_boxes_best[2] = min(shape_capture[1]-1, predict_boxes_best[2]*shape_ratio[1])
                        predict_boxes_best[3] = min(shape_capture[0]-1, predict_boxes_best[3]*shape_ratio[0])
                        box_error_validation = error_detection_handler(predict_boxes_best[:4], camera_timestamp_value)
                        if box_error_validation :
                            predict_keypoints_flatten_best = predict_keypoints[0].cpu().numpy().flatten()
                            predict_keypoints_flatten_best[0] = min(shape_capture[1]-1, predict_keypoints_flatten_best[0]*shape_ratio[1])
                            predict_keypoints_flatten_best[1] = min(shape_capture[0]-1, predict_keypoints_flatten_best[1]*shape_ratio[0])
                            predict_keypoints_flatten_best[2] = min(shape_capture[1]-1, predict_keypoints_flatten_best[2]*shape_ratio[1])
                            predict_keypoints_flatten_best[3] = min(shape_capture[0]-1, predict_keypoints_flatten_best[3]*shape_ratio[0])
                            predict_keypoints_flatten_best[4] = min(shape_capture[1]-1, predict_keypoints_flatten_best[4]*shape_ratio[1])
                            predict_keypoints_flatten_best[5] = min(shape_capture[0]-1, predict_keypoints_flatten_best[5]*shape_ratio[0])
                            predict_keypoints_flatten_best[6] = min(shape_capture[1]-1, predict_keypoints_flatten_best[6]*shape_ratio[1])
                            predict_keypoints_flatten_best[7] = min(shape_capture[0]-1, predict_keypoints_flatten_best[7]*shape_ratio[0])
                            predict_detect_best = np.concatenate([predict_boxes_best, predict_keypoints_flatten_best])
                            if decision_point == "wheel":
                                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, predict_detect_best[6:])
                            else : # decision_point == "bounding_box"
                                x1, y1, x2, y2 = map(int, predict_detect_best[:4])
                                x3, x4, y3, y4 = x1, x2, y2, y1
                            out_sum = off_track_tf_arr1[[y1,y2,y3,y4],[x1,x2,x3,x4]].sum()
                            out_tf_value = 1 if out_sum >= out_wheel_num else 0
                            start_tf_value = rectangle_line_intersect(predict_detect_best[:4], start_pt_arr)
                    
            # 공유 변수에 저장
            capture_arr2[:, :, :] = capture_arr_copy
            detect_best_arr[:] = predict_detect_best
            out_tf.value = out_tf_value
            start_tf.value = start_tf_value
            detect_model_timestamp.value = camera_timestamp_value
            detect_model_looptime.value = sum(time_buffer) / max(len(time_buffer),1)
                    
            # 시간 계산
            time_buffer.append(time.time()  - s_time)
            
            
    except Exception as e:
        error_message = traceback.format_exc()
        shared_dict["error"] = "detect_model.py \n" + error_message