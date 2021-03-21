from mtcnn import MTCNN
from src import face_detect_image
import cv2 

img = cv2.cvtColor(cv2.imread("data/demo.jpg"), cv2.COLOR_BGR2RGB)
DETECTOR = MTCNN()

list_bbox = face_detect_image(img, DETECTOR)
print(list_bbox)
for bbox in list_bbox:
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)

cv2.imshow("img", img)
cv2.waitKey(0)