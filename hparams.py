detect_confidence_min = 0.5
shape_yolo_box_output = 6
shape_yolo_keypoints_flatten_output = 8
shape_yolo_output = shape_yolo_box_output + shape_yolo_keypoints_flatten_output
error_detection_deque_max_length = 30
error_detection_deque_max_age = 0.5 # sec
error_detection_threshold = 1  # 0 ~ 
out_wheel_num = 4
decision_point = "wheel" # wheel or bounding_box

off_track_img_path = "source/off_track.jpg"
off_fence_img_path = "source/off_fence.jpg"
rgb_sum_threshold = 30

boost_mode = False

port_webserver = 5000
display_wheel_point = False
display_time = True

log_saving_interval_seconds = 60

complete_interval_min = 3  # seconds
default_finish_time_str = '30-12-31 23:59:59.000'

df_logs_path = "./data/saved_logs/df_logs.pickle"
excel_logs_path = "./data/saved_logs/df_logs.xlsx"
logs_pickle_path = "./data/saved_logs/pickle_history/"
logs_excel_path = "./data/saved_logs/excel_history/"
