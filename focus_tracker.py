import cv2
import mediapipe as mp

#go into the mediapipe module to grab the face_mesh module 
mp_face_mesh = mp.solutions.face_mesh
# face_mesh is a more detailed face detection compared to the facebox version

face_mesh = mp_face_mesh.FaceMesh(
  # added frame refresh
  max_num_faces=1,
  refine_landmarks=True,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
)


#basic command to open laptop camera
cap = cv2.VideoCapture(0)

calibration_frames = 60
frame_count = 0
calibrated_head_drop = None
calibrated_gaze_ratio = None
head_drop_total = 0
gaze_ratio_total = 0
bad_frame_count = 0
bad_frame_limit = 8

while True:
  focus = False
  
  ret, frame = cap.read()
  if not ret:
    break
  
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(rgb) #ask mediapip to check if there is a face
  
  
  
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      h,w,_ = frame.shape
      
      # key points for head angle
      nose = face_landmarks.landmark[1]
      left_eye_corner = face_landmarks.landmark[33]
      right_eye_corner = face_landmarks.landmark[263]
      
      nose_y = nose.y
      eye_avg_y = (left_eye_corner.y + right_eye_corner.y)/2
      
      # bigger difference means the nose drop further below the eyes == looking down!
      head_drop = nose_y - eye_avg_y
      
      
      # iris + eyelid points for gaze direction
      left_iris = face_landmarks.landmark[468]
      right_iris = face_landmarks.landmark[473]

      left_lid_top = face_landmarks.landmark[159]
      left_lid_bottom = face_landmarks.landmark[145]
      right_lid_top = face_landmarks.landmark[386]
      right_lid_bottom = face_landmarks.landmark[374]

      left_eye_height = abs(left_lid_bottom.y - left_lid_top.y)
      right_eye_height = abs(right_lid_bottom.y - right_lid_top.y)

      if left_eye_height > 0 and right_eye_height > 0:
        left_gaze_ratio = (left_iris.y - left_lid_top.y) / left_eye_height
        right_gaze_ratio = (right_iris.y - right_lid_top.y) / right_eye_height
        gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
      else:
        gaze_ratio = 0.5
      
      # first few frames are used to learn your normal "looking at screen" position
      if frame_count < calibration_frames:
        head_drop_total += head_drop
        gaze_ratio_total += gaze_ratio
        frame_count += 1
        focus = True

        if frame_count == calibration_frames:
          calibrated_head_drop = head_drop_total / calibration_frames
          calibrated_gaze_ratio = gaze_ratio_total / calibration_frames
      else:
        head_change = head_drop - calibrated_head_drop
        gaze_change = gaze_ratio - calibrated_gaze_ratio

        looks_away = head_change > 0.04 or gaze_change > 0.04

        if looks_away:
          bad_frame_count += 1
        else:
          bad_frame_count = 0

        if bad_frame_count >= bad_frame_limit:
          focus = False
        else:
          focus = True
        
      # for testing purpose
      
      
      cv2.putText(
        frame,
        f"head_drop: {head_drop:.3f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
      )

      if frame_count < calibration_frames:
        cv2.putText(
          frame,
          "CALIBRATING: look at screen",
          (30, 140),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2
        )
      
      cv2.putText(
        frame,
        f"gaze_ratio: {gaze_ratio:.3f}",
        (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
      )

      if calibrated_head_drop is not None and calibrated_gaze_ratio is not None:
        head_change = head_drop - calibrated_head_drop
        gaze_change = gaze_ratio - calibrated_gaze_ratio

        cv2.putText(
          frame,
          f"head_change: {head_change:.3f}",
          (30, 170),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2
        )

        cv2.putText(
          frame,
          f"gaze_change: {gaze_change:.3f}",
          (30, 200),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2
        )

      cv2.putText(
        frame,
        f"bad_frames: {bad_frame_count}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
      )
      
      eye_points = [33, 133, 159, 145, 362, 263, 386, 374, 468, 473]
      
      for point_id in eye_points:
        # shorten this whole function into "lm" for easier re-usability 
        lm = face_landmarks.landmark[point_id] 
        x = int(lm.x * w)
        y = int(lm.y * h)
        
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        
  if focus:
    cv2.putText(frame,"FOCUSED", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  else:
    cv2.putText(frame, "DISTRACTED", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
  
  cv2.imshow("Live Camera", frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()