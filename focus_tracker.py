import cv2
import mediapipe as mp

#go into the mediapipe module to grab the face_mesh module 
mp_face_mesh = mp.solutions.face_mesh
# face_mesh is a more detailed face detection compared to the facebox version
face_mesh = mp_face_mesh.FaceMesh()


#basic command to open laptop camera
cap = cv2.VideoCapture(0)

while True:
  
  ret, frame = cap.read()
  if not ret:
    break
  
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(rgb) #ask mediapip to check if there is a face
  
  
  focus = False
  
  if results.multi_face_landmarks:
    focus = True
  
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      h,w,_ = frame.shape
      
      eye_point = [33,133,159,145]
      
      for point_id in eye_point:
        # shorten this whole function into "lm" for easier re-usability 
        lm = face_landmarks.landmark[point_id] 
        x = int(lm.x * w)
        y = int(lm.y * h)
        
        cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        
  if focus:
    cv2.putText(frame,"YOU ARE FOCUS", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  else:
    cv2.putText(frame, "DISTRACTED", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
  
  cv2.imshow("Live Camera", frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()