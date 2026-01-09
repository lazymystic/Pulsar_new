import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

CURRENT_DIR = os.path.dirname(__file__)

MP_DRAWING = mp.solutions.drawing_utils
MP_HANDS = mp.solutions.hands

hand_landmarks_dict = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_MCP": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_MCP": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_MCP": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}

"""
SPECIFY THE INPUT AND OUTPUT FOLDER HERE
"""
INPUT_PATH = os.path.join(CURRENT_DIR, "input")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "output")

task_list = [""]

"""
SPECIFY HERE THE LIST OF AUGMENTATIONS TO APPLY
"""
augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]

folder_names = os.listdir(INPUT_PATH)

for folder in folder_names:
    print("PROCESSING FOLDER ", folder)

    files = os.listdir(os.path.join(INPUT_PATH, folder))

    for file in files:
        if file not in task_list:
            continue

        landmarks_dict_frames = {}
        landmarks_dict_points = {}

        for aug in augmentations:
            hands = MP_HANDS.Hands(
                min_detection_confidence=0.8,
                min_tracking_confidence=0.9,
                max_num_hands=1
			)

            cap = cv2.VideoCapture(f"{INPUT_PATH}/{folder}/{file}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            i = 0
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                landmarks_dict_points = {}

                if aug == "original":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif aug == "flip-vert":
                    image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
                elif aug == "flip-hor":
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                elif aug == "flip-hor-vert":
                    image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                draw_blank = np.full((720, 1280, 3), 255, np.uint8)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark_pos, landmark_name in zip(hand_landmarks.landmark, hand_landmarks_dict.keys()):
                            landmarks_dict_points[landmark_name] = (
                                landmark_pos.x,
                                landmark_pos.y,
                                landmark_pos.z
							)
                        MP_DRAWING.draw_landmarks(
                            image, hand_landmarks,
                            MP_HANDS.HAND_CONNECTIONS
						)
                else:  # if no landmarks found
                    for landmark_name in (hand_landmarks_dict.keys()):
                        landmarks_dict_points[landmark_name] = (0, 0, 0)

                landmarks_dict_frames[i] = landmarks_dict_points
                i += 1
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            df_landmarks = pd.DataFrame(landmarks_dict_frames).transpose()

            removed_ext = file.split(".")[0]

            if not os.path.exists(f"{OUTPUT_PATH}/{folder}"):
                os.makedirs(f"{OUTPUT_PATH}/{folder}")
            df_landmarks.to_csv(
                f"{OUTPUT_PATH}/{folder}/{folder}_{removed_ext}_out_{aug}.csv")
