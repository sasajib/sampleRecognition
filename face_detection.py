import cv2
import face_recognition
from time import time
import datetime


start = time()
image = cv2.imread('dataset/aishwarya_rai_bachchan/24.\ aishwarya_rai_bachchan.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

print('located')
for (top, right, bottom, left) in boxes:
    cv2.rectangle(rgb, (left, top), (right, bottom), (0, 0, 255), 2)

end = time()
print((end-start))

print(encodings)    


cv2.imshow('Image', rgb)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


