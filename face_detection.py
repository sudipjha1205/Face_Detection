import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

image = cv2.imread('sudip.jpeg')

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=10)
for (x,y,width,height) in detected_faces:
    cv2.rectangle(image,(x,y),(x+width,y+height),(255,0,0),5)


plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()