import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import pickle


#with open('../models/uploaded_dataset.pkl', 'rb') as f:    ----->ALONE
#   model = pickle.load(f)

with open('models/uploaded_dataset.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def process_video(input_path, output_path):
    """
    Process the video to overlay landmarks and classification results,
    and accumulate posture predictions for final report.
    Returns a dictionary with posture report details.
    """
    
    posture_counts = {"closed": 0, "fear": 0, "confident": 0}
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file: " + input_path)

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


            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ---------------------------
            # 1. Draw Landmarks
       
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # ---------------------------
            # 2. Classification & Aggregation
            
            try:
                if results.pose_landmarks:
                    pose_row = list(np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                    ).flatten())
                else:
                    pose_row = [0] * (33 * 4)
                    
                if results.face_landmarks:
                    face_row = list(np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.face_landmarks.landmark]
                    ).flatten())
                else:
                    face_row = [0] * (468 * 4)
                    
                if results.right_hand_landmarks:
                    right_hand_row = list(np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                    ).flatten())
                else:
                    right_hand_row = [0] * (21 * 3)
                    
                if results.left_hand_landmarks:
                    left_hand_row = list(np.array(
                        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                    ).flatten())
                else:
                    left_hand_row = [0] * (21 * 3)
                
                row = pose_row + face_row + right_hand_row + left_hand_row

                X = pd.DataFrame([row], columns=model.feature_names_in_)  #--------------*********----
                #X = pd.DataFrame([row])
                
                body_language_class = model.predict(X)[0]
                body_language_prob  = model.predict_proba(X)[0]

             
                if body_language_prob[np.argmax(body_language_prob)] > 0.60:
                    if body_language_class in posture_counts:
                        posture_counts[body_language_class] += 1

                # Text placement 
                if results.pose_landmarks:
                    left_ear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
                    coords = (int(left_ear.x * width), int(left_ear.y * height))
                else:
                    coords = (50, 50)

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
                cv2.rectangle(image, (0,0), (250,60), (245,117,16), -1)
                cv2.putText(image, 'CLASS', (95,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(
                    image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                    (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA
                )

            except Exception as e:
                print("Classification Error:", e)

            # ---------------------------
            # 3. Write Processed Frame
            # ---------------------------
            out.write(image)

    cap.release()
    out.release()

    # ---------------------------
    # 4. Compute Final Posture Report
    # ---------------------------
    total_count = sum(posture_counts.values())
    if total_count == 0:
        final_posture = "reasonable posture"
    else:
       
        max_class = max(posture_counts, key=posture_counts.get)
        
        max_val = posture_counts[max_class]
        ties = [cls for cls, cnt in posture_counts.items() if cnt == max_val]
        if len(ties) > 1:
            final_posture = "reasonable posture"
        else:
            if max_class == "closed":
                final_posture = "Defensive/Nervous posture. Try maintaining eye contact and open body language."
            elif max_class == "fear":
                final_posture = "Nervous; needs to be calmer. This may indicate stress or anxiety"
            elif max_class == "confident":
                final_posture = "Confident posture. Can improve eye contact more"
            else:
                final_posture = "reasonable posture"

    posture_report = {
        #"counts": posture_counts,
        "final_posture": final_posture,
        "total_frames_counted": total_count
    }
    
    return posture_report

# ---------------------------------------------------------------------------
# Uncomment the following block to test video_processing independently.
#
# if __name__ == "__main__":
#     input_vid = "closed_test1.mp4"
#     output_vid = "processed_closed_test1.mp4"
#     report = process_video(input_vid, output_vid)
#     print("Posture Report:", report)
