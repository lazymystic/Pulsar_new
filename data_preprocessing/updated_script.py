import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os

CURRENT_DIR = os.path.dirname(__file__)
print("Working Directory:", CURRENT_DIR)

# The new API requires the explicit path to the downloaded model file
MODEL_PATH = os.path.join(CURRENT_DIR, "hand_landmarker.task")

hand_landmarks_dict = {
    "WRIST": 0, "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5, "INDEX_FINGER_PIP": 6, "INDEX_FINGER_DIP": 7, "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9, "MIDDLE_FINGER_PIP": 10, "MIDDLE_FINGER_DIP": 11, "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13, "RING_FINGER_PIP": 14, "RING_FINGER_DIP": 15, "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
}

# Connections for manual drawing (replaces mp.solutions.drawing_utils)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

INPUT_PATH = os.path.join(CURRENT_DIR, "input")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "output")

# Note: Your original script had task_list = [""], which would skip all files. 
# I've commented out the filter below, but you can re-enable it if needed.
task_list = [""] 

augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]

# Initialize the Tasks API Options
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Using VIDEO mode for sequential tracking
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_tracking_confidence=0.9
)

if not os.path.exists(INPUT_PATH):
    print(f"Error: Input path does not exist: {INPUT_PATH}")
    exit()

folder_names = os.listdir(INPUT_PATH)

for folder in folder_names:
    print("PROCESSING FOLDER:", folder)
    folder_path = os.path.join(INPUT_PATH, folder)
    
    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    for file in files:
        # if file not in task_list:
        #    continue

        for aug in augmentations:
            # Create a fresh landmarker instance per augmentation to reset the tracking state
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                video_path = os.path.join(folder_path, file)
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Get FPS to calculate exact millisecond timestamps for the Tasks API
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Processing {file} with augmentation '{aug}' at {fps} FPS")
                if fps == 0: fps = 30 # Fallback just in case OpenCV can't read it
                
                landmarks_dict_frames = {}
                frame_index = 0

                while cap.isOpened():
                    ret, image = cap.read()
                    if not ret:
                        break

                    # Calculate the timestamp (required by VIDEO running mode)
                    timestamp_ms = int((frame_index / fps) * 1000)

                    # Apply Augmentations
                    if aug == "original":
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif aug == "flip-vert":
                        image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
                    elif aug == "flip-hor":
                        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    elif aug == "flip-hor-vert":
                        image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)

                    # Convert the numpy array to a MediaPipe Image object
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

                    # Run inference using the Tasks API
                    results = landmarker.detect_for_video(mp_image, timestamp_ms)

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    h, w, c = image.shape
                    landmarks_dict_points = {}

                    if results.hand_landmarks:
                        # Extract the first hand detected
                        hand_landmarks = results.hand_landmarks[0]

                        # Extract Coordinates
                        for landmark_pos, landmark_name in zip(hand_landmarks, hand_landmarks_dict.keys()):
                            landmarks_dict_points[landmark_name] = (landmark_pos.x, landmark_pos.y, landmark_pos.z)

                        # Manual Visualization (Replaces drawing_utils)
                        for connection in HAND_CONNECTIONS:
                            start_idx, end_idx = connection[0], connection[1]
                            pt1, pt2 = hand_landmarks[start_idx], hand_landmarks[end_idx]
                            x1, y1 = int(pt1.x * w), int(pt1.y * h)
                            x2, y2 = int(pt2.x * w), int(pt2.y * h)
                            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for pt in hand_landmarks:
                            cx, cy = int(pt.x * w), int(pt.y * h)
                            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
                    else:
                        for landmark_name in hand_landmarks_dict.keys():
                            landmarks_dict_points[landmark_name] = (0, 0, 0)

                    landmarks_dict_frames[frame_index] = landmarks_dict_points
                    frame_index += 1
                    
                    cv2.imshow(f'Processing {file} ({aug})', image)
                    if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit early
                        break

                cap.release()

            # Export to CSV
            df_landmarks = pd.DataFrame(landmarks_dict_frames).transpose()
            removed_ext = file.split(".")[0]
            
            out_folder_path = os.path.join(OUTPUT_PATH, folder)
            if not os.path.exists(out_folder_path):
                os.makedirs(out_folder_path)
                
            csv_file_path = os.path.join(out_folder_path, f"{folder}_{removed_ext}_out_{aug}.csv")
            df_landmarks.to_csv(csv_file_path)

cv2.destroyAllWindows()