import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)
import hparams as hp
import config as cf

import pytz
from datetime import datetime, timedelta
from numba import jit
import requests
from tqdm import tqdm
import cv2
import socket
import time
from collections import deque



status2str_dict = {
    0 : 'waiting',
    1 : 'ready',
    2 : 'driving',
    3 : 'paused',
    4 : 'finished'
}
str2status_dict = {value: key for key, value in status2str_dict.items()}
status_list = list(status2str_dict.values())

action2str_dict = {
    0 : 'none',
    1 : 'ready',
    2 : 'start',
    3 : 'out',
    4 : 'stop',
    5 : 'complete',
    6 : 'finish',
    7 : 'reset'
}
str2action_dict = {value: key for key, value in action2str_dict.items()}
action_list = list(action2str_dict.values())


# ['waiting', 'ready', 'driving', 'paused', 'finished']
def observe_to_action_str(status_str, out_tf_value, start_tf_value, detect_best_arr, finish_time_value) :
    if finish_time_value < time.time() :
        return 'finish', None
    elif status_str == 'waiting' :
        return 'none', None
    elif status_str == 'ready' :
        if (out_tf_value == 0) and (start_tf_value == 1) :
            return 'start', None
        else :
            return 'none', None
    elif status_str == 'driving' :
        if out_tf_value == 1 :
            return 'out', detect_best_arr.copy()
        elif start_tf_value == 1 :
            return 'complete', None
        else :
            return 'none', None
    elif status_str == 'paused' :
        return 'none', None
    elif status_str == 'finished' :
        return 'none', None



data_queue = deque(maxlen=hp.error_detection_deque_max_length)
def error_detection_handler(boxes, timestamp,
        data_queue = data_queue, 
        max_age = hp.error_detection_deque_max_age,
        threshold = hp.error_detection_threshold
    ):
    
	x1, y1, x2, y2 = boxes
	x = (x1 + x2)/2
	y = (y1 + y2)/2
	box_diagonal_length = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

	# 오래된 데이터 제거
	while data_queue:
		x, y, timestamp_old, _ = data_queue[0]  
		if timestamp - timestamp_old > max_age:
			data_queue.popleft() 
		else:
			break
	
	data_len = len(data_queue)
	if data_len == 0 :
		data_queue.append((x, y, timestamp, 0))
		return True
	else :
		mean_velocity = sum(target_velocity for *_, target_velocity in data_queue) / data_len  
		x_prev, y_prev, timestamp_prev, velocity_prev = data_queue[-1]
		time_delta =  max(0.01, timestamp - timestamp_prev)
		moving_delta = ((x - x_prev)**2 + (y - y_prev)**2)**0.5
		if moving_delta < mean_velocity * time_delta + box_diagonal_length * threshold :
			velocity = moving_delta / time_delta
			data_queue.append((x, y, timestamp, velocity))
			return True
		else :
			return False


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        IP = s.getsockname()[0]
    finally:
        s.close()
    return IP


def get_off_track_tf_arr():
    rgb_sum_threshold = hp.rgb_sum_threshold
    resolution = cf.resolution
    off_track_img_path = hp.off_track_img_path
    if not os.path.isfile(off_track_img_path):
        raise Exception(f"The image file could not be found : {off_track_img_path}")
    
    camera_capture_width, camera_capture_height, detect_input_width, detect_input_height = get_resolution_config(resolution)
    off_track_img = cv2.imread(off_track_img_path)
    off_track_img1 = cv2.resize(off_track_img, (camera_capture_width, camera_capture_height))
    off_track_img2 = cv2.resize(off_track_img, (detect_input_width, detect_input_height))
    off_track_tf_arr1 = (off_track_img1.sum(axis=2) < rgb_sum_threshold)
    off_track_tf_arr2 = (off_track_img2.sum(axis=2) < rgb_sum_threshold)
    return off_track_tf_arr1, off_track_tf_arr2

def get_off_fence_tf_arr():
    rgb_sum_threshold = hp.rgb_sum_threshold
    resolution = cf.resolution
    off_fence_img_path = hp.off_fence_img_path
    if not os.path.isfile(off_fence_img_path):
        raise Exception(f"The image file could not be found : {off_fence_img_path}")
    
    camera_capture_width, camera_capture_height, detect_input_width, detect_input_height = get_resolution_config(resolution)
    off_fence_img = cv2.imread(off_fence_img_path)
    off_fence_img1 = cv2.resize(off_fence_img, (camera_capture_width, camera_capture_height))
    off_fence_img2 = cv2.resize(off_fence_img, (detect_input_width, detect_input_height))
    off_fence_tf_arr1 = (off_fence_img1.sum(axis=2) < rgb_sum_threshold)
    off_fence_tf_arr2 = (off_fence_img2.sum(axis=2) < rgb_sum_threshold)
    return off_fence_tf_arr1, off_fence_tf_arr2



# @jit 데코레이터를 사용하여 함수를 JIT 컴파일합니다.
@jit(nopython=True)
def modify_array(origin, condition):
    for i in range(origin.shape[0]):
        for j in range(origin.shape[1]):
            if condition[i, j]:
                origin[i, j] = 0

def on_segment(p, q, r):
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    
    return False

def rectangle_line_intersect(rect, line):
    x1, y1, x2, y2 = rect  # 사각형의 좌하단과 우상단 좌표
    x3, y3, x4, y4 = line  # 선분의 끝점 좌표
    
    if (x3 == x4) and (y3 == y4): return 0  # 선분 양끝점이 같은 경우
    
    # 사각형의 네 변을 선분으로 보고 각각 검사
    if do_intersect((x1, y1), (x1, y2), (x3, y3), (x4, y4)): return 1  # 왼쪽 변
    if do_intersect((x1, y2), (x2, y2), (x3, y3), (x4, y4)): return 1  # 상단 변
    if do_intersect((x2, y2), (x2, y1), (x3, y3), (x4, y4)): return 1  # 오른쪽 변
    if do_intersect((x2, y1), (x1, y1), (x3, y3), (x4, y4)): return 1  # 하단 변
    
    return 0

def check_model(model_type = 'small'):
    model_type = model_type.strip().lower()
    latest_version = 0
    for model_version in range(1, 100):
        model_name = f"v{model_version}_{model_type}"
        download_url = f"https://pub-a06c1fb0ad3e476f910f5ce72aff9f9b.r2.dev/shared_resources/models/{model_name}.pt"
        response = requests.head(download_url)
        if response.status_code == 200:
            latest_version = model_version
        else :
            return latest_version
    return latest_version

def download_model(model_version, model_type):
    model_type = model_type.strip().lower()
    model_name = f"v{model_version}_{model_type}"
    download_url = f"https://pub-a06c1fb0ad3e476f910f5ce72aff9f9b.r2.dev/shared_resources/models/{model_name}.pt"
    download_path = f"./data/saved_model/{model_name}.pt"
    if not os.path.isfile(download_path):
        os.makedirs("./data/saved_model/", exist_ok=True)
        try :
            print(f"trying {model_name} download...")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))

                with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True) as progress_bar:
                    with open(download_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            progress_bar.update(len(chunk))
                            f.write(chunk)
        except :
            raise Exception(f"Unable to download the model : {model_name}")
    
    return download_path
                    
def get_resolution_config(resolution = cf.resolution):
    if resolution == "FHD" :
        camera_capture_width = 1920
        camera_capture_height = 1080
        detect_input_width = 1920
        detect_input_height = 1056
    elif resolution == "HD" :
        camera_capture_width = 1280
        camera_capture_height = 720
        detect_input_width = 1280
        detect_input_height = 704
    elif resolution == "qHD" :
        camera_capture_width = 960
        camera_capture_height = 540
        detect_input_width = 960
        detect_input_height = 528
    elif resolution.upper() == "custom" :
        camera_capture_width = 1920
        camera_capture_height = 1080
        detect_input_width = 1024
        detect_input_height = 576
    else :
        raise Exception("The resolution is not allowed. Resolutions can only be FHD, HD, or QHD.")

    return camera_capture_width, camera_capture_height, detect_input_width, detect_input_height


def get_arr_shape(resolution = cf.resolution):
    camera_capture_width, camera_capture_height, detect_input_width, detect_input_height = get_resolution_config(resolution)
    shape_capture = (camera_capture_height, camera_capture_width, 3)
    shape_detect = (detect_input_height, detect_input_width, 3)
    return shape_capture, shape_detect

time_zone = cf.time_zone
target_timezone = pytz.timezone(time_zone)

def time_to_str(unix_timestamp, target_timezone = target_timezone):
    utc_time = datetime.utcfromtimestamp(unix_timestamp)
    
    utc_time = pytz.utc.localize(utc_time)
    kst_time = utc_time.astimezone(target_timezone)
    formatted_time = kst_time.strftime('%y-%m-%d %H:%M:%S.%f')[:-3]
    return formatted_time

def time_to_str2(unix_timestamp, target_timezone = target_timezone):
    utc_time = datetime.utcfromtimestamp(unix_timestamp)
    
    utc_time = pytz.utc.localize(utc_time)
    kst_time = utc_time.astimezone(target_timezone)
    formatted_time = kst_time.strftime('%y%m%d-%H%M%S')
    return formatted_time


def str_to_time(time_str, target_timezone = target_timezone):
    if '.' not in time_str:
        time_str += '.000'  # 밀리초 부분 추가
    dt = datetime.strptime(time_str, '%y-%m-%d %H:%M:%S.%f')
    dt = target_timezone.localize(dt)
    utc_dt = dt.astimezone(pytz.utc)
    unix_timestamp = utc_dt.timestamp()
    return unix_timestamp

def seconds_to_str(seconds):
    delta = timedelta(seconds=seconds)
    minutes, seconds = divmod(delta.seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{delta.microseconds//1000:03d}"

def str_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds
