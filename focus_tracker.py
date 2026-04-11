import cv2
import mediapipe as md



#basic command to open laptop camera
cap = cv2.VideoCapture(0)

while True:
  
  ret, frame = cap.read()
  
  if not ret:
    break
  
  cv2.imshow("Live Camera", frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv2.destroyAllWindows()