import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)
from util_functions import time_to_str
import cv2
import numpy as np
import time
import traceback
from collections import deque
import logging
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def display_fn(
    shared_dict, display_looptime, boost_mode, display_wheel_point, display_time, detect_model_timestamp, status2str_dict, last_out_arr_ctypes, status, detect_best_arr_ctypes, capture_arr2_ctypes, capture_arr3_ctypes, out_tf, start_tf, shape_capture
    ):
    try :
        time_buffer = deque(maxlen=20)
        last_out_arr = np.frombuffer(last_out_arr_ctypes.get_obj(), dtype=np.float32)
        detect_best_arr = np.frombuffer(detect_best_arr_ctypes.get_obj(), dtype=np.float32)
        capture_arr2 = np.frombuffer(capture_arr2_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        capture_arr3 = np.frombuffer(capture_arr3_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        
        detect_rectangle_thickness = 2
        text_thickness = 2
        wheel_circle_radius = 4
        
        out_text_y_margin =  round(shape_capture[1] * (10 / 1280))
        out_text_fontsize = round(shape_capture[1] * (1 / 1280), 1)

        start_text_y_margin =  round(shape_capture[1] * (10 / 1280))
        start_text_fontsize = round(shape_capture[1] * (1 / 1280), 1)

        inside_text_y_margin =  round(shape_capture[1] * (10 / 1280))
        inside_text_fontsize = round(shape_capture[1] * (0.9 / 1280), 1)

        last_out_circle_radius = round(shape_capture[1] * (20 / 1280))
        last_out_text_x_margin =  round(shape_capture[1] * (50 / 1280))
        last_out_text_y_margin =  round(shape_capture[1] * (30 / 1280))
        last_out_text_fontsize = round(shape_capture[1] * (0.8 / 1280), 1)

        time_rectangle_x = round(shape_capture[1] * (340 / 1280))
        time_rectangle_y = round(shape_capture[1] * (30 / 1280))
        time_text_x_margin = round(shape_capture[1] * (10 / 1280))
        time_text_y_margin = round(shape_capture[1] * (22 / 1280))
        time_text_fontsize = round(shape_capture[1] * (0.8 / 1280), 1)
        
        detect_model_timestamp_value_prev = 0
        shared_dict['ready_display'] = True
        
        while True:
            if boost_mode :
                s_time = time.time()
                detect_model_timestamp_value = detect_model_timestamp.value
            else :
                while True :
                    s_time = time.time()
                    detect_model_timestamp_value = detect_model_timestamp.value
                    if detect_model_timestamp_value > detect_model_timestamp_value_prev:
                        detect_model_timestamp_value_prev = detect_model_timestamp_value
                        break
            last_out_arr_copy = last_out_arr.copy()
            detect_best_arr_copy = detect_best_arr.copy()
            capture_arr2_copy = capture_arr2.copy()
            if detect_best_arr_copy.sum() > 0:
                box_x1, box_y1, box_x2, box_y2 = map(int, detect_best_arr_copy[:4])
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, detect_best_arr_copy[6:])
                if out_tf.value :
                    cv2.rectangle(capture_arr2_copy, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), detect_rectangle_thickness)
                    cv2.putText(capture_arr2_copy, "OUT!!", (box_x1, box_y1 - out_text_y_margin), cv2.FONT_HERSHEY_SIMPLEX,  out_text_fontsize, (0, 0, 255), text_thickness)
                    if display_wheel_point :
                        cv2.circle(capture_arr2_copy, (x1, y1), wheel_circle_radius,  (0, 0, 255), -1)
                        cv2.circle(capture_arr2_copy, (x2, y2), wheel_circle_radius,  (0, 0, 255), -1)
                        cv2.circle(capture_arr2_copy, (x3, y3), wheel_circle_radius,  (0, 0, 255), -1)
                        cv2.circle(capture_arr2_copy, (x4, y4), wheel_circle_radius,  (0, 0, 255), -1)
                elif start_tf.value : 
                    cv2.rectangle(capture_arr2_copy, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), detect_rectangle_thickness)
                    cv2.putText(capture_arr2_copy, "Start!!", (box_x1, box_y1 - start_text_y_margin), cv2.FONT_HERSHEY_SIMPLEX,  start_text_fontsize, (0, 255, 0), text_thickness)
                    if display_wheel_point :
                        cv2.circle(capture_arr2_copy, (x1, y1), wheel_circle_radius,  (0, 255, 0), -1)
                        cv2.circle(capture_arr2_copy, (x2, y2), wheel_circle_radius,  (0, 255, 0), -1)
                        cv2.circle(capture_arr2_copy, (x3, y3), wheel_circle_radius,  (0, 255, 0), -1)
                        cv2.circle(capture_arr2_copy, (x4, y4), wheel_circle_radius,  (0, 255, 0), -1)
                else : 
                    cv2.rectangle(capture_arr2_copy, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), detect_rectangle_thickness)
                    cv2.putText(capture_arr2_copy, "Inside", (box_x1, box_y1 - inside_text_y_margin), cv2.FONT_HERSHEY_SIMPLEX,  inside_text_fontsize, (255, 0, 0), text_thickness)
                    if display_wheel_point :
                        cv2.circle(capture_arr2_copy, (x1, y1), wheel_circle_radius,  (255, 0, 0), -1)
                        cv2.circle(capture_arr2_copy, (x2, y2), wheel_circle_radius,  (255, 0, 0), -1)
                        cv2.circle(capture_arr2_copy, (x3, y3), wheel_circle_radius,  (255, 0, 0), -1)
                        cv2.circle(capture_arr2_copy, (x4, y4), wheel_circle_radius,  (255, 0, 0), -1)
            if status2str_dict[status.value] == 'paused' :
                x_out, y_out = map(int, [last_out_arr_copy[[0,2]].mean(), last_out_arr_copy[[1,3]].mean()])
                cv2.circle(capture_arr2_copy, (x_out, y_out), last_out_circle_radius, (0, 0, 255), -1)
                cv2.putText(capture_arr2_copy, "Last Out", (x_out -last_out_text_x_margin, y_out - last_out_text_y_margin), cv2.FONT_HERSHEY_SIMPLEX,  last_out_text_fontsize, (0, 0, 255), text_thickness)
                
            # 시간표시
            if display_time :
                detect_model_timestamp_str = time_to_str(detect_model_timestamp_value)
                cv2.rectangle(capture_arr2_copy, (0, 0), (time_rectangle_x, time_rectangle_y), (0, 0, 0), -1)
                cv2.putText(capture_arr2_copy, detect_model_timestamp_str, (time_text_x_margin, time_text_y_margin), cv2.FONT_HERSHEY_SIMPLEX, time_text_fontsize, (255, 255, 255), text_thickness)
            
            # 저장
            capture_arr3[:, :, :] = capture_arr2_copy
            display_looptime.value = sum(time_buffer) / max(len(time_buffer),1)
                    
            # 시간 계산               
            time_buffer.append(time.time() - s_time)
            
    except Exception as e:
        error_message = traceback.format_exc()
        shared_dict["error"] = "display.py \n" + error_message

        
