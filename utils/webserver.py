import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)
from util_functions import time_to_str, str_to_time, seconds_to_str, str_to_seconds

from flask import Flask, Response, render_template, request, jsonify, stream_with_context
import pandas as pd
import os
import traceback
import logging
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
import numpy as np
import cv2


current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '../data/templates')
static_dir = os.path.join(current_dir, '../data/static')
app = Flask(__name__, template_folder=template_dir, static_folder = static_dir)


# from logging.handlers import RotatingFileHandler
# log_file_path = './flask_app.log'  # 로그 파일 경로 설정
# log_file_handler = RotatingFileHandler(log_file_path, maxBytes=100000, backupCount=10)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log_file_handler.setFormatter(formatter)
# log_file_handler.setLevel(logging.INFO)
# app.logger.addHandler(log_file_handler)
# app.logger.setLevel(logging.INFO)


def webserver_fn(shared_dict,
    status, start_time, trial_count, lap_count, offtrack_count, record_origin, record_final, total_laps,
    status2str_dict, str2action_dict, action, finish_time, auto_start_on, observe_on, team_name, team_name_list, 
    port_webserver, capture_arr3_ctypes, shape_capture, # video_looptime
    ):
    try:
        capture_arr3 = np.frombuffer(capture_arr3_ctypes.get_obj(), dtype=np.uint8).reshape(shape_capture)
        
        def generate_frames():
            nonlocal  capture_arr3
            while True:
                ret, buffer = cv2.imencode('.jpg', capture_arr3)
                if not ret:
                    continue  # 인코딩 실패시 다음 프레임으로 넘어갑니다.
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        @app.route('/')
        def index():
            return render_template('index.html')
        
        @app.route('/video')
        def video_feed():
            # return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
            return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/logs')
        def logs():
            nonlocal shared_dict
            df_logs = shared_dict['df_logs']
            return render_template('logs.html', table=df_logs.to_dict(orient='records'))

        @app.route('/logs/add', methods=['POST'])
        def add():
            nonlocal shared_dict
            df_logs = shared_dict['df_logs']
            content = request.json
            new_row = pd.DataFrame([content])
            df_logs = pd.concat([new_row, df_logs], ignore_index=True)
            df_logs.sort_values(by='ID', ascending=False, inplace=True)
            df_logs.reset_index(drop=True, inplace=True)
            shared_dict['df_logs'] = df_logs
            return jsonify(status='success')

        @app.route('/logs/update', methods=['POST'])
        def update():
            nonlocal shared_dict
            df_logs = shared_dict['df_logs']
            content = request.json
            id_to_update = int(content['ID'])
            field = content['Field']
            new_value = content['Value']

            if field in df_logs.columns:
                df_logs.loc[df_logs['ID'] == id_to_update, field] = new_value
                shared_dict['df_logs'] = df_logs
            else:
                print(f"Error: The field {field} does not exist.")
            return jsonify(status='success')

        @app.route('/logs/delete', methods=['POST'])
        def delete():
            nonlocal shared_dict
            df_logs = shared_dict['df_logs']
            content = request.json
            id_to_delete = int(content['ID'])
            df_logs = df_logs[df_logs['ID'] != id_to_delete]
            shared_dict['df_logs'] = df_logs
            return jsonify(status='success')


        @app.route('/remote')
        def remote_control():
            return render_template('remote.html', team_name_list=team_name_list)

        @app.route('/get_status')
        def get_status():
            nonlocal status
            return jsonify({"status": status2str_dict[status.value]})

        @app.route('/get_view_data')
        def get_view_data():
            view_data_dict = {
                "team_name" : team_name.value.decode('utf-8'),
                "status": status2str_dict[status.value],
                # "start_time" : time_to_str(start_time.value),
                "trial_count" : trial_count.value,
                "lap_count" : lap_count.value,
                "offtrack_count" : offtrack_count.value,
                # "record_origin" : seconds_to_str(record_origin.value),
                "record_final" : seconds_to_str(record_final.value),
                "total_laps" : total_laps
            }
            return jsonify(view_data_dict)

        @app.route('/get_best_record')
        def get_best_record():
            return jsonify(shared_dict['best_record_srr'].to_dict())

        @app.route('/get_initial_settings')
        def get_initial_settings():
            settings = {
                'team_name': team_name.value.decode('utf-8'),
                'observe_on': observe_on.value,
                'auto_start_on': auto_start_on.value,
                'finish_time': time_to_str(finish_time.value).split(".")[0]
            }
            return jsonify(settings)


        @app.route('/update_team', methods=['POST'])
        def update_team():
            nonlocal team_name
            team_name.value = request.form['team_name'].encode('utf-8')
            df_logs = shared_dict['df_logs']
            df_logs_team = df_logs[df_logs['team_name'] == team_name.value.decode('utf-8')]
            trial_count.value = df_logs_team.shape[0]
            return "OK"

        @app.route('/update_observe_on', methods=['POST'])
        def update_observe_on():
            nonlocal observe_on
            observe_on.value = int(request.form['observe_on'])
            return "OK"

        @app.route('/update_auto_start_on', methods=['POST'])
        def update_auto_start_on():
            nonlocal auto_start_on
            auto_start_on.value = int(request.form['auto_start_on'])
            return "OK"

        @app.route('/update_finish_time', methods=['POST'])
        def update_finish_time():
            nonlocal finish_time
            finish_time.value = str_to_time(request.form['finish_time'])
            return "OK"

        @app.route('/update_action', methods=['POST'])
        def update_action():
            nonlocal action
            action.value = str2action_dict[request.form['action']]
            return "OK"

        shared_dict['ready_webserver'] = True
        app.run(
            host='0.0.0.0', port=port_webserver, threaded=True,  
            # debug=True, use_reloader=False
        )
    
    except Exception as e:
        error_message = traceback.format_exc()
        shared_dict["error"] = "webserver.py \n" + error_message

        
