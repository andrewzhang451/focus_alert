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
      
      # give coordinates for (nose, left eye, right eye)
      nose = face_landmarks.landmark[1]
      left_eye = face_landmarks.landmark[33]
      right_eye = face_landmarks.landmark[263]
      
      nose_y = nose.y
      eye_avg_y = (left_eye.y + right_eye.y)/2
      
      # bigger difference means the nose drop further below the eyes == looking down!
      head_drop = nose_y - eye_avg_y
      
      if head_drop > 0.18:
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
      
      eye_points = [33, 133, 159, 145, 362, 263, 386, 374]
      
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