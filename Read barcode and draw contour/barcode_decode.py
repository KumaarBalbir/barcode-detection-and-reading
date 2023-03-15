import cv2
import  numpy as np
from pyzbar.pyzbar import decode

img=cv2.imread('images/1.jpg')

# printing all information decoded from bar or qr code
# print(decode(img))
# print(len(decode(img))==0)

for barcode in decode(img):
    # print(barcode)
    # get the data from detected barcode
    myData=barcode.data.decode('utf-8')

    # get the coordinate for the bounding box
    (left, top, width, height) = barcode.rect

    # draw a bbox around the detected barcode
    cv2.rectangle(img,(left,top),(left+width,top+height),(255,0,0),10)
    cv2.putText(img, myData, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 8)


# Show the image
img=cv2.resize(img,(1080,840))
cv2.imshow("result..",img)
cv2.waitKey(0)
