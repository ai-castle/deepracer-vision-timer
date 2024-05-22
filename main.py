if __name__ == '__main__':
    print("Please wait a moment ...")
    print("The process is being prepared ...")
    print("This task may take a few minutes ...")

    import multiprocessing
    try :
        multiprocessing.set_start_method('spawn')
    except :
        pass

    from multiprocessing import Manager

    import time
    import hparams as hp
    import config as cf
    import pandas as pd
    import os
    import ctypes
    import numpy as np
    import time
    import socket
    import torch

    from utils.util_functions import get_off_track_tf_arr, get_off_fence_tf_arr, download_model, get_resolution_config, time_to_str, str_to_time, seconds_to_str, str_to_seconds
    from utils.camera import camera_fn
    from utils.detect_model import detect_model_fn
    from utils.display import display_fn
    from utils.webserver import webserver_fn
    from utils.save import save_fn



    ### config
    model_version = cf.model_version
    model_type = cf.model_type
    detect_model_path = download_model(model_version, model_type)
    camera_idx = cf.camera_idx
    resolution = cf.resolution
    camera_rotation_180 = cf.camera_rotation_180
    camera_capture_width, camera_capture_height, detect_input_width, detect_input_height = get_resolution_config(resolution)
    shape_capture = (camera_capture_height, camera_capture_width, 3)
    shape_detect = (detect_input_height, detect_input_width, 3)
    total_laps = cf.total_laps
    offtrack_penalty = cf.offtrack_penalty
    team_name_list = cf.team_name_list

    starting_line_endpoints = cf.starting_line_endpoints
    start_pt_arr = np.array(starting_line_endpoints).flatten()
    time_zone = cf.time_zone

    ### hparams
    df_logs_path = hp.df_logs_path
    excel_logs_path = hp.excel_logs_path
    logs_pickle_path = hp.logs_pickle_path
    logs_excel_path = hp.logs_excel_path
    shape_yolo_output = hp.shape_yolo_output
    log_saving_interval_seconds = hp.log_saving_interval_seconds
    complete_interval_min = hp.complete_interval_min
    default_finish_time_str = hp.default_finish_time_str
    port_webserver = hp.port_webserver
    out_wheel_num = hp.out_wheel_num
    boost_mode = hp.boost_mode
    display_wheel_point = hp.display_wheel_point
    display_time = hp.display_time
    decision_point = hp.decision_point
    
    # 기타 정보
    ip = socket.gethostbyname(socket.gethostname())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print()
    print("-------- infomation --------")
    print(f"- model_type : {model_type}")
    print(f"- model_version : {model_version}")
    print(f"  ({detect_model_path})")
    print(f"- model device : {device}")
    print(f"- camera_idx : {camera_idx}")
    print(f"- resolution : {resolution}")
    print(f"- camera_rotation_180 : {camera_rotation_180}")
    print(f"- starting_line_endpoints : {starting_line_endpoints}")
    print(f"- total_laps : {total_laps}")
    print(f"- offtrack_penalty : {offtrack_penalty} sec")
    print(f"- team_name_list : {team_name_list}")
    print(f"- time_zone : {time_zone}")
    print(f"- ip : {ip}")
    print(f"- port_webserver : {port_webserver}")

    off_track_tf_arr1, off_track_tf_arr2 = get_off_track_tf_arr()
    off_fence_tf_arr1, off_fence_tf_arr2 = get_off_fence_tf_arr()

    observe_on = multiprocessing.Value('i',1)
    auto_start_on = multiprocessing.Value('i',1)
    team_penalty = multiprocessing.Value('f',0.0)
    finish_time = multiprocessing.Value('d', str_to_time(default_finish_time_str))
    team_name = multiprocessing.Array('c', 100)
    action = multiprocessing.Value('i',0)
    status = multiprocessing.Value('i',0)
    start_time = multiprocessing.Value('d',0.0)
    start_time_last = multiprocessing.Value('d',0.0)
    complete_time_last = multiprocessing.Value('d',0.0)
    trial_count = multiprocessing.Value('i',0)
    lap_count = multiprocessing.Value('i',0)
    last_out_arr_ctypes = multiprocessing.Array(ctypes.c_float, shape_yolo_output)
    last_out_arr = np.frombuffer(last_out_arr_ctypes.get_obj(), dtype=np.float32)
    record_origin = multiprocessing.Value('f',0.0)
    record_last = multiprocessing.Value('f',0.0)
    offtrack_count = multiprocessing.Value('i',0)
    record_final = multiprocessing.Value('f',0.0)

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
    
    
    #################################################################
    print()
    print(" ready... ")
    if os.path.isfile(df_logs_path) :
        df_logs = pd.read_pickle(df_logs_path)
    else :
        df_logs = pd.DataFrame(columns = ['ID', 'valid', 'team_name', 'record_final', 'record_origin', 'offtrack_count', 'team_penalty', 'start_time', 'memo'])
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['df_logs'] = df_logs
    shared_dict['error'] = ""
    shared_dict['ready_save'], ready_save = False, False
    shared_dict['ready_detect'], ready_detect = False, False
    shared_dict['ready_camera'], ready_camera = False, False
    shared_dict['ready_display'], ready_display = False, False
    shared_dict['ready_webserver'], ready_webserver = False, False
    ready_all_done = False

    ##########
    p_save = multiprocessing.Process(
        target=save_fn, 
        args=(shared_dict, team_name_list, logs_excel_path, logs_pickle_path, df_logs_path, excel_logs_path, log_saving_interval_seconds), 
        daemon=True
    )
    p_save.start()

    ##########
    capture_arr_ctypes = multiprocessing.Array(ctypes.c_uint8, int(np.prod(shape_capture)))
    capture_arr = np.frombuffer(capture_arr_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
    capture_looptime = multiprocessing.Value('f',0.1)
    capture_timestamp = multiprocessing.Value('d',0.0)
    p_camera = multiprocessing.Process(target=camera_fn, args=(shared_dict, capture_looptime, camera_rotation_180, capture_arr_ctypes, capture_timestamp, camera_idx, shape_capture), daemon=True)
    p_camera.start()

    ##########
    detect_best_arr_ctypes = multiprocessing.Array(ctypes.c_float, shape_yolo_output)
    detect_best_arr = np.frombuffer(detect_best_arr_ctypes.get_obj(), dtype=np.float32)
    capture_arr2_ctypes = multiprocessing.Array(ctypes.c_uint8, int(np.prod(shape_capture)))
    capture_arr2 = np.frombuffer(capture_arr2_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
    detect_model_looptime = multiprocessing.Value('f',0.1)
    detect_model_timestamp = multiprocessing.Value('d',0.0)
    # squared_diff_mean = multiprocessing.Value('d',0.0)
    # debug_copy_time = multiprocessing.Value('f',0.0)
    # debug_step = multiprocessing.Value('i',0)
    out_tf = multiprocessing.Value('i',0)
    start_tf = multiprocessing.Value('i',0)
    p_detect_model = multiprocessing.Process(
        target=detect_model_fn, 
        args=(shared_dict, detect_model_looptime, decision_point, boost_mode, out_wheel_num, device, observe_on, start_tf, out_tf, start_pt_arr, off_track_tf_arr1, off_fence_tf_arr2, shape_detect, shape_yolo_output, hp.detect_confidence_min, capture_arr_ctypes, capture_arr2_ctypes, detect_best_arr_ctypes, detect_model_timestamp, capture_timestamp, detect_model_path, shape_capture), 
        daemon=True
    )
    p_detect_model.start()


    ##########
    # video_looptime = multiprocessing.Value('f',0.1)
    # p_video = multiprocessing.Process(
    #     target=video_fn, 
    #     args=(shared_dict, video_looptime, detect_model_timestamp, status2str_dict, last_out_arr_ctypes, status, hp.port_video, detect_best_arr_ctypes, capture_arr2_ctypes, out_tf, start_tf, shape_capture), 
    #     daemon=True
    # )
    # p_video.start()
    
    display_looptime = multiprocessing.Value('f',0.1)
    capture_arr3_ctypes = multiprocessing.Array(ctypes.c_uint8, int(np.prod(shape_capture)))
    capture_arr3 = np.frombuffer(capture_arr3_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
    p_display = multiprocessing.Process(
        target=display_fn, 
        args=(shared_dict, display_looptime, boost_mode, display_wheel_point, display_time, detect_model_timestamp, status2str_dict, last_out_arr_ctypes, status, detect_best_arr_ctypes, capture_arr2_ctypes, capture_arr3_ctypes, out_tf, start_tf, shape_capture), 
        daemon=True
    )
    p_display.start()


    ##########
    p_webserver = multiprocessing.Process(
        target=webserver_fn, 
        args=(shared_dict,
            status, start_time, trial_count, lap_count, offtrack_count, record_origin, record_final, total_laps,
            status2str_dict, str2action_dict, action, finish_time, auto_start_on, observe_on, team_name, team_name_list, 
            port_webserver, capture_arr3_ctypes, shape_capture), 
        daemon=True
    )
    p_webserver.start()
    
    
    #################################################################
    s_time = time.time()
    while True :
        ################# part1 : human #################
        status_value = status.value
        status_str = status2str_dict[status_value]

        action_value = action.value
        action.value = 0
        action_str = action2str_dict[action_value]

        time_now = time.time()
        if status_str == 'waiting' :
            if action_str == 'ready' :
                status.value = str2status_dict['ready']
                start_time.value = 0.0
                start_time_last.value = 0.0
                complete_time_last.value = 0.0
                # trial_count.value = 0
                lap_count.value = 0
                record_last.value = 0.0
                record_origin.value = 0.0
                offtrack_count.value = 0
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            else :
                pass
            
        elif status_str == 'ready' :
            if action_str == 'start' :
                status.value = str2status_dict['driving']
                start_time.value = time_now
                start_time_last.value = time_now
                complete_time_last.value = time_now
                trial_count.value += 1
                lap_count.value = 1
            elif action_str == 'finish' : 
                status.value = str2status_dict['finished']
                trial_count.value -= 1
                
        elif status_str == 'driving' :
            if action_str == 'stop' :
                status.value = str2status_dict['ready']
                start_time.value = 0.0
                start_time_last.value = 0.0
                complete_time_last.value = 0.0
                trial_count.value -= 1
                lap_count.value = 0
                record_last.value = 0.0
                record_origin.value = 0.0
                offtrack_count.value = 0
                last_out_arr[:] = detect_best_arr.copy()
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            elif action_str == 'out' :
                status.value = str2status_dict['paused']
                record_last.value += (time_now - start_time_last.value)
                record_origin.value = record_last.value
                offtrack_count.value += 1
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            elif action_str == 'complete' :
                complete_interval = time_now - complete_time_last.value
                if complete_interval > complete_interval_min :
                    ### 최종 완주
                    if lap_count.value == total_laps :  
                        # 로그 기록에 넣기
                        record_last.value += (time_now - start_time_last.value)
                        record_origin.value = record_last.value
                        record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                        df_logs = shared_dict['df_logs']
                        if df_logs.shape[0] > 0 :
                            last_id = df_logs.iloc[0]['ID']
                        else :
                            last_id = 0
                        add_log_row = pd.DataFrame([{
                            'ID': last_id+1,
                            'valid': 'True',
                            'team_name': team_name.value.decode('utf-8'),
                            'record_final': seconds_to_str(record_final.value),
                            'record_origin': seconds_to_str(record_origin.value),
                            'offtrack_count': str(offtrack_count.value),
                            'team_penalty': str(team_penalty.value),
                            'start_time': time_to_str(start_time.value),
                            'memo': 'Poor'
                        }])
                        shared_dict['df_logs'] = pd.concat([add_log_row, df_logs], ignore_index=True)
                        
                        # 자동 다음 랩타임
                        if auto_start_on.value == 1 :  
                            status.value = str2status_dict['driving']
                            start_time.value = time_now
                            start_time_last.value = time_now
                            complete_time_last.value = time_now
                            trial_count.value += 1
                            lap_count.value = 1
                            record_last.value = 0.0
                            record_origin.value = 0.0
                            offtrack_count.value = 0
                            record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                            
                        # 자동으로 랩타임 넘어가지 않음
                        else :
                            status.value = str2status_dict['waiting']
                            trial_count.value += 1
                    ### 아직 몇바퀴 더 남음
                    else :
                        status.value = str2status_dict['driving']
                        lap_count.value += 1
                        complete_time_last.value = time_now
                        record_origin.value = record_last.value + (time_now - start_time_last.value)
                        record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                else :
                    record_origin.value = record_last.value + (time_now - start_time_last.value)
                    record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            elif action_str == 'finish' : 
                status.value = str2status_dict['finished']
                trial_count.value -= 1
            else :
                record_origin.value = record_last.value + (time_now - start_time_last.value)
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                
        elif status_str == 'paused' :
            if action_str == 'stop' :
                status.value = str2status_dict['ready']
                start_time.value = 0.0
                start_time_last.value = 0.0
                complete_time_last.value = 0.0
                trial_count.value -= 1
                lap_count.value = 0
                record_last.value = 0.0
                record_origin.value = 0.0
                offtrack_count.value = 0
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            elif action_str == 'start' :
                status.value = str2status_dict['driving']
                start_time_last.value = time_now
            elif action_str == 'finish' : 
                status.value = str2status_dict['finished']
                trial_count.value -= 1
                
        elif status_str == 'finished' :
            if action_str == 'reset' :
                status.value = str2status_dict['waiting']
                start_time.value = 0.0
                start_time_last.value = 0.0
                complete_time_last.value = 0.0
                trial_count.value = 0
                df_logs = shared_dict['df_logs']
                df_logs_team = df_logs[df_logs['team_name'] == team_name.value.decode('utf-8')]
                trial_count.value = df_logs_team.shape[0]
                lap_count.value = 0
                record_last.value = 0.0
                record_origin.value = 0.0
                offtrack_count.value = 0
                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
            else :
                pass        
        
        ################# part2 : machine observe #################
        if observe_on.value == 1 :
            status_value = status.value
            status_str = status2str_dict[status_value]
            action_str, last_detect_arr = observe_to_action_str(status_str, out_tf.value, start_tf.value, detect_best_arr, finish_time.value)

            time_now2 = detect_model_timestamp.value
            if status_str == 'waiting' :
                pass
                
            elif status_str == 'ready' :
                if action_str == 'start' :
                    status.value = str2status_dict['driving']
                    start_time.value = time_now2
                    start_time_last.value = time_now2
                    complete_time_last.value = time_now2
                    trial_count.value += 1
                    lap_count.value = 1
                else : 
                    pass
                    
            elif status_str == 'driving' :
                if action_str == 'stop' :
                    pass
                elif action_str == 'out' :
                    # x1, y1, x2, y2, x3, y3, x4, y4 = map(int, detect_best_arr[6:])
                    # out_sum = off_track_tf_arr1[[y1,y2,y3,y4],[x1,x2,x3,x4]].sum()
                    # if out_sum >= out_wheel_num :
                        status.value = str2status_dict['paused']
                        record_last.value += (time_now2 - start_time_last.value)
                        record_origin.value = record_last.value
                        offtrack_count.value += 1
                        last_out_arr[:] = last_detect_arr
                        record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                    # else :
                    #     record_origin.value = record_last.value + (time_now - start_time_last.value)
                    #     record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty 
                elif action_str == 'complete' :
                    complete_interval = time_now2 - complete_time_last.value
                    if complete_interval > complete_interval_min :
                        ### 최종 완주
                        if lap_count.value == total_laps :  
                            # 로그 기록에 넣기
                            record_last.value += (time_now2 - start_time_last.value)
                            record_origin.value = record_last.value
                            record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                            df_logs = shared_dict['df_logs']
                            if df_logs.shape[0] > 0 :
                                last_id = df_logs.iloc[0]['ID']
                            else :
                                last_id = 0
                            add_log_row = pd.DataFrame([{
                                'ID': last_id+1,
                                'valid': 'True',
                                'team_name': team_name.value.decode('utf-8'),
                                'record_final': seconds_to_str(record_final.value),
                                'record_origin': seconds_to_str(record_origin.value),
                                'offtrack_count': str(offtrack_count.value),
                                'team_penalty': str(team_penalty.value),
                                'start_time': time_to_str(start_time.value),
                                'memo': 'Poor'
                            }])
                            shared_dict['df_logs'] = pd.concat([add_log_row, df_logs], ignore_index=True)
                            
                            # 자동 다음 랩타임
                            if auto_start_on.value == 1 :  
                                status.value = str2status_dict['driving']
                                start_time.value = time_now2
                                start_time_last.value = time_now2
                                complete_time_last.value = time_now2
                                trial_count.value += 1
                                lap_count.value = 1
                                record_last.value = 0.0
                                record_origin.value = 0.0
                                offtrack_count.value = 0
                                record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                                
                            # 자동으로 랩타임 넘어가지 않음
                            else :
                                status.value = str2status_dict['waiting']
                                trial_count.value += 1
                        ### 아직 몇바퀴 더 남음
                        else :
                            status.value = str2status_dict['driving']
                            lap_count.value += 1
                            complete_time_last.value = time_now2
                            record_origin.value = record_last.value + (time_now2 - start_time_last.value)
                            record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                    else :
                        record_origin.value = record_last.value + (time_now2 - start_time_last.value)
                        record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
                elif action_str == 'finish' : 
                    status.value = str2status_dict['finished']
                    trial_count.value -= 1
                else :
                    record_origin.value = record_last.value + (time_now - start_time_last.value)
                    record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty      
            elif status_str == 'paused' :
                if action_str == 'stop' :
                    pass
                elif action_str == 'start' :
                    pass
                elif action_str == 'finish' : 
                    status.value = str2status_dict['finished']
                    trial_count.value -= 1
                    
            elif status_str == 'finished' :
                pass

        record_final.value = record_origin.value + team_penalty.value + offtrack_count.value * offtrack_penalty
        
        # ready
        if not ready_save :
            if shared_dict['ready_save']:
                ready_save = True
                print(f"(ready) save")
        if not ready_detect :
            if shared_dict['ready_detect']:
                ready_detect = True
                print(f"(ready) detect model")
        if not ready_camera :
            if shared_dict['ready_camera']:
                ready_camera = True
                print(f"(ready) camera")
        if not ready_display :
            if shared_dict['ready_display']:
                ready_display = True
                print(f"(ready) display")
        if not ready_webserver :
            if shared_dict['ready_webserver']:
                ready_webserver = True
                print(f"(ready) webserver")
        
        # fps
        if ready_all_done :
            f_time = time.time()
            if f_time - s_time > 10 :
                camera_fps = int(1 / capture_looptime.value)
                detect_fps = int(1 / detect_model_looptime.value)
                display_fps = int(1 / display_looptime.value)
                print(f"[FPS] camera : {camera_fps} < detect model : {detect_fps} < display : {display_fps}")
                s_time = f_time
        else :
            if ready_save and ready_detect and ready_camera and ready_display and ready_webserver :
                ready_all_done = True
                s_time = time.time()
                print("")
                print(" ========== start ========= ")
                print("(start) All ready. Start!!")
                print(f"(Timer View)")
                print(f"- http://localhost:5000/")
                print(f"- http://{ip}:5000/")
                print("(Remote Control)")
                print(f"- http://localhost:5000/remote")
                print(f"- http://{ip}:5000/remote")
                print("(Logs Control)")
                print(f"- http://localhost:5000/logs")
                print(f"- http://{ip}:5000/logs")
                print("")
        # error
        if shared_dict["error"]:
            print("[error] " + shared_dict["error"])
            shared_dict["error"] = ""
            break