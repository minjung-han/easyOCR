import os
import json
import time
import logging
import torch
import requests
from logging.handlers import RotatingFileHandler
from scipy.signal import find_peaks

# ocr 클라이언트 측 코드

def config_reading(json_file_name):
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, json_file_name)
    
    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        logging.error(f"{current_directory} - {json_file_name} 파일을 찾을 수 없습니다.")
        return None

def get_log_level(log_level):
    log_level = log_level.upper()
    if log_level == "DEBUG":
        return logging.DEBUG
    elif log_level == "INFO":
        return logging.INFO
    elif log_level == "WARNING":
        return logging.WARNING
    elif log_level == "ERROR":
        return logging.ERROR
    elif log_level == "CRITICAL":
        return logging.CRITICAL
    else:
        return logging.INFO

def main():
    try:
        json_data = config_reading('config.json')

        if json_data is not None:
            datafilter = json_data['datafilter']
            image_extensions = datafilter['image_extensions']
            max_cpu_use = json_data['cpu_use_persent']
            max_mem_use = json_data['memory_use_persent']
            tika_app = json_data['tika_app']
            tika_server_ip = tika_app['tika_server_ip']
            tika_server_port = tika_app['tika_server_port']
            tika_ocr_server_count = tika_app['tika_ocr_server_count']
            tika_ocr_process_num = tika_app['tika_ocr_process_num']

            # ocr language info, config.json에 추가해야함
            ocr_info = json_data['ocr_info']
            ocr_languages = ocr_info['ocr_languages']

            root_path = json_data['root_path']
            source_path = json_data['datainfopath']['source_path']
            source_path = os.path.join(root_path, source_path)

            el_target_path = json_data['elasticsearch']['normal_el_file_target_path']
            el_file_path = json_data['elasticsearch']['el_file_path']
            result_path = os.path.join(root_path, el_target_path, el_file_path)

            log_level = ocr_info['log_to_level']
            log_file = ocr_info['log_file']
            log_to_console = ocr_info['log_to_console']

            log_file_path = ocr_info['log_file_path']
            ocr_ver = ocr_info["ocr_ver"]

            if not os.path.exists(log_file_path):
                os.mkdir(log_file_path)

            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)

            if not os.path.exists(log_file_path):
                os.mkdir(log_file_path)

            current_time = time.strftime("%Y%m%d%H%M%S")
            log_file_name = log_file + "_" + current_time + ".log"
            log_file_path = os.path.join(log_file_path, log_file_name)

            logger = logging.getLogger('')
            log_level = get_log_level(log_level)
            logger.setLevel(log_level)

            file_handler = RotatingFileHandler(log_file_path, maxBytes=1024*1024*1024, backupCount=7, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            logging.getLogger("PIL").setLevel(logging.ERROR)
        
            if log_to_console:
                console = logging.StreamHandler()
                console.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                console.setFormatter(formatter)
                logging.getLogger('').addHandler(console)

            logging.info(f"ocr_processor ver {ocr_ver}")

    except Exception as e:
        error_log = f"config.json 을 읽는 도중 오류 발생 : {str(e)}"
        logging.error(f"{error_log}")        
        return
    
    info_message = f"ocr_languages: {ocr_languages}"
    logging.info(info_message)

    # 이미지 파일 경로 초기화
    image_files = []
    for root, _, files in os.walk(source_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)

    # FastAPI 서버에 이미지 경로 전달
    server_url = "http://192.168.0.215:5000/perform_ocr/"

    for image_path in image_files:
        response = requests.post(server_url, params={"image_path": image_path})

        if response.status_code == 200:
            ocr_result = response.json().get("result")
            print(f"OCR Result for {image_path}: {ocr_result}")
        else:
            print(f"Error for {image_path}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
