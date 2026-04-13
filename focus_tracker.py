import cv2
import mediapipe as mp

#go into the mediapipe module to grab the face_detection module 
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

#basic command to open laptop camera
cap = cv2.VideoCapture(0)

while True:
  
  ret, frame = cap.read()
  if not ret:
    break
  
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_detection.process(rgb) #ask mediapip to check if there is a face
  
  if results.detections:
    #for each of the face we found
    for detection in results.detections:
      #bouding box is just the box for your face. this will 
      bbox = detection.location_data.relative_bounding_box
      
      #mediapipe gives percentages for box values, not pixel locations. so need box size to convert later
      h,w, _ = frame.shape
      
      #based on where the bbox is on the screen, this will convert that locaiton into coordinates.
      x = int(bbox.xmin * w)
      y = int(bbox.ymin * h)
      width = int(bbox.width * w)
      height = int(bbox.height * h)
      
      # this will draw the green box for user to see
      cv2.rectangle(frame, (x,y), (x + width, y + height), (0,255,0), 2)
      
      
  cv2.imshow("Live Camera", frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()