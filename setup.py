import argparse

parser = argparse.ArgumentParser(description="This script performs various operations related to device setup and testing, such as checking CUDA availability, checking connected cameras, and capturing images.")
parser.add_argument('--check_cuda_available', action='store_true', help='Check if CUDA is available on this machine.')
parser.add_argument('--check_model_latest_version', action='store_true', help='Check the latest version of the model.')
parser.add_argument('--local_camera_check', action='store_true', help='List all local cameras and check their availability.')
parser.add_argument('--check_resolution', action='store_true', help='Check the resolution settings of the specified camera.')
parser.add_argument('--capture_image', action='store_true', help='Capture an image using the specified camera settings.')
parser.add_argument('--clear_logs', action='store_true', help='Clear log files and directories.')
args = parser.parse_args()

if args.check_cuda_available:
    print("\n\n======== Cuda check ======== ")
    print("checking...")
    import torch

    if torch.cuda.is_available():
        print("(info) Cuda : True, Cuda is available !")
    else:
        print("(info) Cuda : False, Cuda is not available.")
        print("(Warning) Model inference using the CPU may have slower inference speed.")

elif args.check_model_latest_version:
    print("\n\n======== Model Version ======== ")
    print("checking...")
    from utils.util_functions import check_model

    latest_version = check_model()
    print(f"- latest model version : {latest_version}")

elif args.local_camera_check:
    print("\n\n======== Local Camera List ======== ")
    print("checking...")
    from matplotlib import pyplot as plt
    import cv2

    camera_indices = [0, 1, 2, 3, 4, 5]

    num_cameras = len(camera_indices)
    cols = 3
    rows = (num_cameras + 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axs = axs.flatten()

    valid_camera_index = []
    print(f"- checking camera index ... ", end=" ")
    for index, ax in zip(camera_indices, axs):
        print(index, end=" ")
        cap = cv2.VideoCapture(index)
        ret, frame = cap.read()
        if not ret:
            ax.axis("off")
        else:
            ax.set_title(f"Camera at index {index}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_frame)
            ax.axis("off")
            valid_camera_index.append(index)
        cap.release()

    for ax in axs[num_cameras:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

elif args.check_resolution:
    import config as cf
    from utils.util_functions import get_resolution_config
    print("\n\n======== Camera Resolution Check ======== ")
    print("Camera information")
    print(f"- camera_idx : {cf.camera_idx}")
    print(f"- camera_rotation_180 : {cf.camera_rotation_180}")
    print("------------------- \n")
    print("checking...")

    import time
    import cv2

    cap = cv2.VideoCapture(cf.camera_idx)
    resolutions = ["FHD", "HD", "qHD"]
    for idx, target_resolution in enumerate(resolutions):
        width, height, _, __ = get_resolution_config(target_resolution)
        print(f"- {target_resolution} ({width}x{height}) : ", end="")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        valid_resolution = (actual_width == width) and (actual_height == height)
        if valid_resolution:
            print(f"O / FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
        else:
            print("X")
    cap.release()

elif args.capture_image:
    import config as cf
    from utils.util_functions import get_resolution_config
    print("\n\n======== Image Capture ======== ")
    print("Camera information")
    print(f"- camera_idx : {cf.camera_idx}")
    print(f"- resolution : {cf.resolution}")
    print(f"- camera_rotation_180 : {cf.camera_rotation_180}")
    print("------------------- \n")
    print("capturing...")

    import cv2
    from matplotlib import pyplot as plt

    capture_img_path = "source/capture.jpg"
    off_track_temp_path = "source/off_track.temp.jpg"
    off_fence_temp_path = "source/off_fence.temp.jpg"

    cap = cv2.VideoCapture(cf.camera_idx)
    width, height, _, __ = get_resolution_config(cf.resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, frame = cap.read()
    if cf.camera_rotation_180:
        frame = frame[::-1, ::-1]
    cap.release()

    cv2.imwrite(capture_img_path, frame)
    print(f"(info) The captured image has been saved to {capture_img_path}")
    cv2.imwrite(off_track_temp_path, frame)
    print(f"(info) The captured image has been saved to {off_track_temp_path}")
    cv2.imwrite(off_fence_temp_path, frame)
    print(f"(info) The captured image has been saved to {off_fence_temp_path}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_frame)
    plt.show()

elif args.clear_logs:
    print("clearing...")
    import hparams as hp
    import shutil, os

    def remove_folder(folder):
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print(f"(info) {folder} has been cleared")

    def remove_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"(info) {file_path} has been cleared")

    remove_file(hp.df_logs_path)
    remove_file(hp.excel_logs_path)
    remove_folder(hp.logs_pickle_path)
    remove_folder(hp.logs_excel_path)
