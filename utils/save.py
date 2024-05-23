import os, sys
current_path_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path_dir)
parent_path_dir = os.path.dirname(current_path_dir)
sys.path.append(parent_path_dir)
import hparams as hp
import config as cf

from util_functions import time_to_str2
import time
import traceback

team_name_list = cf.team_name_list
df_logs_path = hp.df_logs_path
excel_logs_path = hp.excel_logs_path
logs_pickle_path = hp.logs_pickle_path
logs_excel_path = hp.logs_excel_path
log_saving_interval_seconds = hp.log_saving_interval_seconds
def save_fn(shared_dict):
    try :
        os.makedirs(logs_pickle_path, exist_ok=True)
        os.makedirs(logs_excel_path, exist_ok=True)
        
        shared_dict['ready_save'] = True
        repeat_count = 0
        while True :
            time.sleep(1)
            repeat_count += 1
            
            df_logs = shared_dict['df_logs']
            best_record_srr = df_logs[df_logs['valid'] == 'True'].groupby('team_name')['record_final'].min()
            for i in team_name_list :
                if not i in best_record_srr.index :
                    best_record_srr[i] = 'none'
            best_record_srr = best_record_srr.sort_values()
            shared_dict['best_record_srr'] = best_record_srr

            if repeat_count % log_saving_interval_seconds == 0 :
                timestamp_formatted = time_to_str2(time.time())
                df_logs.to_pickle(df_logs_path)
                df_logs.to_excel(excel_logs_path)

                sub_pickle_path = os.path.join(logs_pickle_path, f"df_logs_{timestamp_formatted}.pickle")
                sub_excel_path = os.path.join(logs_excel_path, f"df_logs_{timestamp_formatted}.xlsx")
                df_logs.to_pickle(sub_pickle_path)
                df_logs.to_excel(sub_excel_path)

    except Exception as e:
        error_message = traceback.format_exc()
        shared_dict["error"] = "save.py \n" + error_message