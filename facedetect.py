import cv2
import os
img = cv2.imread("t.jpg")

abspath = os.getcwd()
haarfile = os.path.join(abspath,"haarcascades/haarcascade_frontalface_default.xml")
print(haarfile)
cascade = cv2.CascadeClassifier(haarfile)#(cv2.data.haarcascades + haarfile)
face = cascade.detectMultiScale(img)
face[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# if len(face) > 1:
#     max_width = 0
#     for index in range(len(face)):
#         if max_width < face[index][3]:
#             max_width = face[index][3]

# for x, y, w, h in face:
#     print(x,y,w,h)
#     cv2.rectangle(gray, (x,y), (x+w,y+h), (0,0,255),1)
#     face_image = img[y:y+h, x:x+w]
x = face[0][0]
y = face[0][1]
w = face[0][2]
h = face[0][3]
cv2.rectangle(gray, (x,y), (x+w,y+h), (0,0,255),1)
face_image = img[y:y+h, x:x+w]
print(face)
cv2.imshow("",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

cv2.imshow("",face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.imwrite("face.jpg",face_image)

