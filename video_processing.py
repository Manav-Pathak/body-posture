import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# Load your pretrained model
model = joblib.load('model/body_language_handa.pkl')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def process_video(input_path, output_path):
    """Process the video to overlay landmarks AND classification results."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + input_path)

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (for MediaPipe) and disable writing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Re-enable writing and convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ---------------------------
            # 1. Draw Landmarks
            # ---------------------------
            # Face
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            # Right Hand
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )
            # Left Hand
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
            # Pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # ---------------------------
            # 2. Classification (try/except in case of any missing data)
            # ---------------------------
            try:
                # -- Pose Landmarks (33 x 4 = 132) --
                if results.pose_landmarks:
                    pose_row = list(np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                    ).flatten())
                else:
                    pose_row = [0] * (33 * 4)

                # -- Face Landmarks (468 x 4 = 1872) --
                if results.face_landmarks:
                    face_row = list(np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.face_landmarks.landmark]
                    ).flatten())
                else:
                    face_row = [0] * (468 * 4)

                # -- Right Hand Landmarks (21 x 3 = 63) --
                if results.right_hand_landmarks:
                    right_hand_row = list(np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                    ).flatten())
                else:
                    right_hand_row = [0] * (21 * 3)

                # -- Left Hand Landmarks (21 x 3 = 63) --
                if results.left_hand_landmarks:
                    left_hand_row = list(np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                    ).flatten())
                else:
                    left_hand_row = [0] * (21 * 3)

                # Concatenate all landmark data
                row = pose_row + face_row + right_hand_row + left_hand_row

                # Create DataFrame with same columns used in training
                X = pd.DataFrame([row], columns=model.feature_names_in_)

                # Predict Class & Probability
                body_language_class = model.predict(X)[0]
                body_language_prob  = model.predict_proba(X)[0]

                # 2.1. Get a coordinate (like left ear) for text placement
                if results.pose_landmarks:
                    left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
                    coords = (int(left_ear.x * width), int(left_ear.y * height))
                else:
                    coords = (50, 50)  # Default if no pose

                # 2.2. Draw rectangle & text for classification near left ear
                cv2.rectangle(
                    image,
                    (coords[0], coords[1] + 5),
                    (coords[0] + len(body_language_class)*20, coords[1] - 30),
                    (245, 117, 16),
                    -1
                )
                cv2.putText(
                    image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA
                )

                # 2.3. Draw status box (top-left corner)
                cv2.rectangle(image, (0,0), (250,60), (245,117,16), -1)

                # Display Class
                cv2.putText(image, 'CLASS', (95,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(
                    image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA
                )

            except Exception as e:
                # If any error (e.g., mismatch in columns), skip classification on this frame
                print("Classification Error:", e)

            # ---------------------------
            # 3. Write Processed Frame
            # ---------------------------
            out.write(image)

    cap.release()
    out.release()
