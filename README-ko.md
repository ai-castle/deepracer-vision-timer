# DeepRacer Vision Timer

## [0] 소개

DeepRacer Vision Timer는 [AWS DeepRacer](https://aws.amazon.com/deepracer/)를 오프라인에서 대회를 운영하기 위한 비전 기반의 인공지능 자동 계측 시스템 및 타이머를 제공합니다.

- DeepRacer Vision Timer는 Python 3.9 (또는 그 이상의 버전)에서 실행됩니다. GPU 가속에는 NVIDIA GPU가 필요합니다.

- 웹캠 (또는 IP 카메라 등)만 있으면 인공지능 기반으로 차량의 이탈 및 출발선 감지 등을 수행할 수 있습니다. 인공지능 Pose Detection 기술을 활용하여 딥레이서 차량 및 바퀴의 위치를 추적하여 차량이 트랙 이탈 및 출발선 통과 감지 등을 자동화할 수 있습니다. 아래의 영상을 참고하세요.

- 팀 기록 및 순위를 관리할 수 있는 타이머 뷰어 (웹기반)를 제공합니다. 아래 영상을 참고하세요.

- 반드시 [1] 준비사항, [2] 설치, [3] 설정, [4] 실행을 수행해야 정상적으로 작동되므로 아래의 글을 꼼꼼하게 확인하세요.

### 참고 영상

[![DeepRacer Vision Timer - ChunCheon DeepRacer Championship](http://img.youtube.com/vi/PB3EHm5Zjh4/0.jpg)](http://www.youtube.com/watch?v=PB3EHm5Zjh4)

[![Pose Detection - AWS Sydney Summit 2023](http://img.youtube.com/vi/c8uExPBnlKc/0.jpg)](http://www.youtube.com/watch?v=c8uExPBnlKc)

### Contact

- 개발 회사 : 에이아이캐슬 (<https://aicastle.io>)
- 개발자 : 이동재 (dongorae@ai-castle.com)

## [1] 준비사항

1. 컴퓨터
    - Python 버전: 3.9 또는 그 이상
    - CPU: 8코어 이상
    - RAM: 8GB 이상
    - NVIDIA GPU (선택)

2. 웹캠
    - 해상도 지원: qHD(960x540) 또는 HD(1280x720) 또는 FHD(1920x1080)
    - FPS: 30 이상 (권장)
    - 웹캠 대신 IP 카메라 등으로 대체 가능 (실험적)

3. 웹캠 설치
    - 트랙 전체가 촬영될 수 있도록 높은 위치에 웹캠을 설치
    - 웹캠이 최대한 트랙 중앙 위쪽에 설치될수록 좋습니다. 트랙이 비스듬하게 촬영될수록 성능이 저하될 수 있습니다.


## [2] 설치

### 1. Python 및 Git 설치

- 파이썬 3.9 및 그 이상의 버전을 설치하세요: [파이썬 다운로드](https://www.python.org/downloads/)

- Git 설치: [Git 다운로드](https://git-scm.com/downloads)

### 2. Git Clone

- 터미널을 열어 레포지토리를 복사합니다.
    ```shell
    git clone https://github.com/ai-castle/deepracer-vision-timer.git
    cd deepracer-vision-timer
    ```

### 3. PyTorch 설치

- [PyTorch 설치](https://pytorch.org/get-started/locally/)에서 본인의 컴퓨터 환경에 맞는 Torch 버전을 설치합니다.

- NVIDIA GPU를 사용하는 경우 본인의 CUDA 버전에 맞는 PyTorch로 설치하세요.

- 예시: Windows, CUDA 12.1인 경우 아래와 같이 설치합니다.
    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

### 4. 기타 라이브러리 설치

- 이 외의 라이브러리를 설치합니다.
    ```shell
    pip3 install -r requirements.txt
    ```


## [3] 설정

### 1. CUDA (NVIDIA GPU) 가속 사용 가능 여부 확인

- 아래 명령어를 통해 CUDA (NVIDIA GPU) 사용 여부를 체크합니다.
    ```shell
    python setup.py --check_cuda_available
    ```

- 만약, `Cuda : True, Cuda is available !`로 표기된다면 GPU가 사용 가능 상태입니다.

- GPU 없이 CPU로도 차량 탐지가 가능하지만 속도가 매우 느릴 수 있습니다.

- 탐지 모델의 FPS가 카메라의 FPS보다 높게 나와야 좋습니다. FPS 정보는 서버 실행 후 콘솔 창에서 확인 가능합니다. 일반적으로 카메라의 FPS는 30이므로

#### 퍼포먼스

- CPU 기준: 13th Gen Intel(R) Core(TM) i9-13900HX
- GPU 기준: NVIDIA GeForce RTX 4070 Laptop GPU
- 해상도: qHD(960x540), HD(1280x720), FHD(1920x1080)
- 탐지 모델: nano, small, medium

1. GPU (CUDA)
    | Type   | FHD    | HD     | qHD    |
    |--------|--------|--------|--------|
    | nano   | 70 fps | 80 fps | 90 fps |
    | small  | 60 fps | 70 fps | 80 fps |
    | medium | 50 fps | 60 fps | 70 fps |

2. CPU
    | Type   | FHD    | HD     | qHD    |
    |--------|--------|--------|--------|
    | nano   | 30 fps | 35 fps | 40 fps |
    | small  | 25 fps | 30 fps | 35 fps |
    | medium | 20 fps | 25 fps | 30 fps |

### 2. 탐지 모델 설정

- `config.py`를 텍스트 에디터로 열고 `model_type`과 `model_version`을 작성 후 저장합니다.

- `model_type`: `nano`, `small`, `medium`

- `model_version`: 1 이상의 양의 정수

- 최신 모델 버전은 아래 명령어로 확인 가능합니다.
    ```shell
    python setup.py --check_model_latest_version
    ```

- `nano`, `small`, `medium` 순으로 모델은 더 정확한 추론을 제공하지만 더 많은 자원을 소모하므로 추론 속도가 저하됩니다. 모델에 관련된 자세한 내용은 <https://docs.ultralytics.com/tasks/pose/#models>에서 확인 가능합니다.

- 위의 퍼포먼스 표를 참고하여 적절한 `model_type`을 선택해야 합니다. 또는 실행 후 표시되는 FPS를 참고하여 적절한 `model_type`을 찾습니다.

- 학습된 모델의 데이터가 당신의 트랙 환경에 잘 작동되지 않을 수 있습니다. 이 경우 추가 훈련을 통한 커스텀 모델을 사용해야 할 수도 있습니다. 자세한 내용은 하단의 [커스텀 모델 사용하기]를 참고하세요.

### 3. 카메라 인덱스 설정

- 웹캠을 사용하는 경우 아래 명령어를 통해 사용 가능한 로컬 웹캠 목록 및 인덱스를 확인합니다. 화면에 표시되는 웹캠의 인덱스를 기억합니다.
    ```shell
    python setup.py --local_camera_check
    ```

- `config.py`를 열고 `camera_idx`에 위에서 찾은 인덱스 (정수형) 또는 테스트해볼 비디오 파일 경로 (문자형) 등을 작성한 후 저장합니다.

- 예시
    - 웹캠: `camera_idx = 1`
    - 테스트 비디오 파일: `camera_idx = "source/test/test_video.mp4"`
    - IP카메라 (실험적): `camera_idx = "rtsp://admin:12345@192.168.1.10:554/stream1"`

### 4. 카메라 해상도 설정

- 아래 명령어를 통해 위에서 지정한 카메라 인덱스에 해당하는 카메라의 해상도 지원 여부를 체크합니다.
    ```shell
    python setup.py --check_resolution
    ```

- `config.py`를 열고 `resolution`에 지원되는 해상도 중 하나 (`FHD`, `HD`, `qHD`)를 입력합니다.

- 예시
    - FHD(1920x1080): `resolution = "FHD"`
    - HD(1280x720): `resolution = "HD"`
    - qHD(960x540): `resolution = "qHD"`

### 5. 카메라 이미지 캡처

- 아래 명령어를 통해 카메라 캡처를 수행합니다.
    ```shell
    python setup.py --capture_image
    ```

- 만약 카메라가 180도 회전이 필요하다면 `config.py`에서 `camera_rotation_180`를 `True`로 설정한 후 저장하여 위 명령어를 다시 실행합니다.

- 위 명령어를 실행하면 캡처된 이미지는 `source/capture.jpg`, `source/off_track.temp.jpg`, `source/off_fence.temp.jpg`에 저장됩니다. 특히, `source/off_track.temp.jpg`, `source/off_fence.temp.jpg`는 뒤에서 트랙 및 펜스의 아웃라인을 설정하는데 사용될 것입니다.

### 6. 출발선 양끝점 설정

- `source/capture.jpg`를 이미지 뷰어를 통해서 열거나 `python setup.py --capture_image`를 실행하여 이미지 뷰어를 엽니다. 그다음 마우스 포인트의 위치를 통해 출발선 (결승선)의 두 개의 양끝점의 좌표를 찾아 기록합니다.

- `config.py`를 열고 `starting_line_endpoints`에 양끝점을 설정한 후 저장합니다.

- 예시: 양끝점이 (1050,662), (1238,897)인 경우
    ```python
    ######## starting line ########
    starting_line_endpoints = [
        [1050, 662],    # point1 : [x1, y1]
        [1238, 897]     # point2 : [x2, y2]
    ]
    ```

- `starting_line_endpoints`에 입력된 값을 기준으로 출발선 통과를 감지합니다.


### 7. 펜스 외곽 설정

- `source/off_fence.temp.jpg`를 그림판과 같은 툴을 이용하여 펜스 외곽 부분을 검정색으로 칠하여 `source/off_fence.jpg`에 저장합니다. 아래의 예시를 참고하세요.
    ![off_fence.jpg](source/sample/off_fence.jpg)

- 검게 칠해진 부분은 모델이 딥레이서 탐지에서 제외됩니다. 이를 통해 펜스 외곽의 관련성이 없는 차량을 탐지하여 오류가 발생하는 것을 방지할 수 있습니다.

- 검정색은 매우 진하게 칠해야 합니다. 일부 이미지 에디터의 경우 브러쉬가 연하게 칠해지는 경우가 있으니 이를 주의하세요.

### 8. 트랙 외곽 설정

- `source/off_track.temp.jpg`를 그림판과 같은 툴을 이용하여 트랙 외곽 부분을 검정색으로 칠하여 `source/off_track.jpg`에 저장합니다. 아래의 예시를 참고하세요.
    ![off_track.jpg](source/sample/off_track.jpg)

- 검게 칠해진 부분은 모델이 추론한 차량의 바퀴가 이탈했는지 여부를 판단하는데 사용됩니다.

- 검정색은 매우 진하게 칠해야 합니다. 일부 이미지 에디터의 경우 브러쉬가 연하게 칠해지는 경우가 있으니 이를 주의하세요.

### 9. 게임 규칙 설정

- `config.py`를 열고 `total_laps` 및 `offtrack_penalty`를 수정 후 저장합니다.
    - `total_laps`: 차량이 완주하기 위한 총 바퀴 수입니다.
    - `offtrack_penalty`: 차량이 트랙을 이탈 시 부여되는 패널티 (초)입니다.

### 10. 타임존 설정

- `config.py`를 열고 당신의 지역에 맞는 `time_zone`을 수정 후 저장합니다.

- 타임존 리스트는 <https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568>에서 확인 가능합니다.

- 여기서 설정된 타임존을 기준으로 시간과 관련된 작업들을 수행합니다.

### 11. 팀 설정

- `config.py`를 열고 대회에 참가하는 팀 이름 목록을 `team_name_list`에 설정 후 저장합니다.

- 예시
    ```python
    ######## team name ########
    team_name_list = [
        "team1",
        "team2",
        "team3",
        "team4",
        "team5",
        "team6",
    ]
    ```

- 팀 이름 목록을 기반으로 팀의 기록 및 타이머 뷰어 등을 표시합니다.


## [4] 실행

### 1. 서버 실행

- 아래 명령어를 통해 DeepRacer Vision Timer 서버를 실행합니다.
    ```shell
    python main.py
    ```

- 실행 후 아래와 같이 터미널에 표시되면 정상적으로 실행된 것입니다.

    ```shell
    ========== start =========
    (start) All ready. Start!!
    (Timer View)
    - http://localhost:5000/
    - http://172.30.1.54:5000/
    (Remote Control)
    - http://localhost:5000/remote
    - http://172.30.1.54:5000/remote
    (Logs Control)
    - http://localhost:5000/logs
    - http://172.30.1.54:5000/logs
    ```

- 터미널에 아래와 같이 FPS 정보가 10초 간격으로 실시간으로 확인 가능합니다.
    ```shell
    [FPS] camera : 30 < detect model : 40 < display : 164
    ```
    - 여기서 주목할 점은 detect model의 FPS가 camera의 FPS보다 높게 나오는 것이 좋습니다.
    - detect model의 FPS가 지나치게 높게 나올 경우 더 높은 모델 (예: medium) 또는 더 높은 해상도 (예: FHD)를 사용하세요. detect model의 FPS가 낮게 나오는 경우 더 낮은 모델 (예: nano) 또는 더 낮은 해상도 (예: qHD)를 사용하세요.

### 2. Timer Viewer 접속

- http://localhost:5000 또는 http://{ip}:5000로 접속하면 아래와 같은 타이머 뷰어를 볼 수 있습니다.
    ![timer-view](source/etc/timer-view.jpg)

- 실시간 스트리밍을 하는 경우 [OBS](https://obsproject.com)와 같은 툴을 활용하여 송신하세요.

### 3. Remote Control 접속

- http://localhost:5000/remote 또는 http://{ip}:5000/remote로 접속하면 아래와 같은 원격 컨트롤이 가능합니다.
    ![remote-view](source/etc/remote-view.jpg)

- Observe: 기계 추론을 할지 여부를 결정합니다.
    - on: 기계 추론을 활성화하여 자동화합니다. 동시에 사람도 수동으로 컨트롤 가능합니다.
    - off: 기계 추론을 비활성화합니다. 오직 사람이 수동으로 컨트롤 가능합니다.

- Auto Start: 차량이 완주 (Complete) 후에 다음 랩 기록 측정을 자동으로 시작할지 여부를 결정합니다.
    - on: 현재 측정 기록이 완주가 되면 다음 랩 타임 기록이 자동으로 시작됩니다.
    - off: 현재 측정 기록이 완주가 되면 다음 랩 타임 기록이 자동으로 시작되지 않습니다.

- Finish Time: 현재 주행 중인 팀의 마감 시간입니다. 해당 시간에 도달하면 주행 중인 기록이 취소되고 팀의 주행이 마감됩니다.

- Status: 현재 주행 상태를 나타냅니다.
    - Status 종류: Waiting, Ready, Driving, Paused, Finished
    - Action 종류: Ready, Start, Stop, Out, Complete
    - Action 단축키: Enter (Start, Complete), Spacebar(Out), Esc(Stop)

- Status 및 Action에 대한 Workflow는 아래의 다이어그램을 참고하세요.
    ![status-workflow](source/etc/status-workflow.jpg)

### 4. Logs Control 접속

- http://localhost:5000/logs 또는 http://{ip}:5000/logs로 접속하면 아래와 같은 원격 컨트롤이 가능합니다.
    ![logs-view](source/etc/logs-view.jpg)

- 잘못 측정된 기록을 삭제 및 수정 가능합니다.


## [5] 기타 (선택적)

### 커스텀 모델 사용하기

- 학습된 모델의 데이터가 당신의 트랙 환경에 잘 작동되지 않을 수 있습니다. 이 경우 추가 훈련을 통한 커스텀 모델을 사용해야 할 수도 있습니다.

- 이 프로젝트에서 사용되는 모델은 YOLOv8-pose입니다. 모델 훈련에 관한 내용은 <https://docs.ultralytics.com/tasks/pose/#train>을 참고하세요.

- 모델 훈련을 위한 데이터셋 포맷은 다음 형식을 따라야 합니다.
    ```
    0 <x> <y> <width> <height> <px1> <py1> <px2> <py2> <px3> <py3> <px4> <py4>
    ```
    - `0`: 딥레이서의 class-index입니다. 이 값은 항상 0입니다.
    - `<x>`: 딥레이서 차량 바운딩 박스의 중심 x좌표입니다. (범위: 0 ~ 1)
    - `<y>`: 딥레이서 차량 바운딩 박스의 중심 y좌표입니다. (범위: 0 ~ 1)
    - `<width>`: 딥레이서 차량 바운딩 박스의 가로 길이입니다. (범위: 0 ~ 1)
    - `<height>`: 딥레이서 차량 바운딩 박스의 세로 길이입니다. (범위: 0 ~ 1)
    - `<px1>`: 딥레이서 차량 왼쪽 앞바퀴의 x좌표 (범위: 0 ~ 1)
    - `<py1>`: 딥레이서 차량 왼쪽 앞바퀴의 y좌표 (범위: 0 ~ 1)
    - `<px2>`: 딥레이서 차량 오른쪽 앞바퀴의 x좌표 (범위: 0 ~ 1)
    - `<py2>`: 딥레이서 차량 오른쪽 앞바퀴의 y좌표 (범위: 0 ~ 1)
    - `<px3>`: 딥레이서 차량 왼쪽 뒷바퀴의 x좌표 (범위: 0 ~ 1)
    - `<py3>`: 딥레이서 차량 왼쪽 뒷바퀴의 y좌표 (범위: 0 ~ 1)
    - `<px4>`: 딥레이서 차량 오른쪽 뒷바퀴의 x좌표 (범위: 0 ~ 1)
    - `<py4>`: 딥레이서 차량 오른쪽 뒷바퀴의 y좌표 (범위: 0 ~ 1)

- YOLOv8-pose의 데이터 형식에 관한 설명은 <https://docs.ultralytics.com/datasets/pose/#ultralytics-yolo-format>에서 확인 가능합니다.

- 모델을 추가 훈련하기 위한 베이스 모델은 아래 링크를 통해 다운 가능합니다. 다운받은 모델로 추가 훈련을 진행하세요.

    | Version | nano   | small  | medium |
    |---------|--------|--------|--------|
    | 1       | [download](https://pub-a06c1fb0ad3e476f910f5ce72aff9f9b.r2.dev/shared_resources/models/v1_nano.pt) | [download](https://pub-a06c1fb0ad3e476f910f5ce72aff9f9b.r2.dev/shared_resources/models/v1_small.pt) | [download](https://pub-a06c1fb0ad3e476f910f5ce72aff9f9b.r2.dev/shared_resources/models/v1_medium.pt) |

- 커스텀 모델 훈련이 모두 완료되면 `data/saved_model` 폴더에 `v{model_version}_{model_type}.pt` 형식으로 저장합니다. 예를 들어, `config.py`에서 `model_type="small"`, `model_version="custom"`으로 한 경우, `vcustom_small.pt`가 됩니다.

### 로그 데이터 삭제

- 실행에 따라 지속적으로 로그 기록이 쌓이게 됩니다.
    - `data/saved_logs/df_logs.pickle`: 시스템에서 사용하는 로그 파일입니다. 약 1분 간격으로 저장되며 만약 타이머가 예상치 못하게 중단된 경우 이 파일을 기반으로 이전 기록을 복원하여 진행합니다.
    - `data/saved_logs/df_logs.xlsx`: 엑셀 형식으로 저장된 로그 기록 파일입니다. 약 1분 간격으로 저장됩니다.
    - `data/saved_logs/pickle_history`: 이전의 `df_logs.pickle` 기록이 모두 저장되는 폴더입니다.
    - `data/saved_logs/excel_history`: 이전의 `df_logs.xlsx` 기록이 모두 저장되는 폴더입니다.

- 아래 명령어를 통해 모든 로그 기록을 삭제하고 초기화할 수 있습니다.
    ```shell
    python setup.py --clear_logs
    ```

### 시스템 관련 변수 설정 (hparams.py)

- 타이머 시스템과 관련된 변수 값을 `hparams.py`에서 설정 가능합니다.