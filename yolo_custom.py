# Import the essential libraries for this project

# Read more about the YOLO and its different functions at: https://docs.ultralytics.com/ (ctrl + click)

from ultralytics import  YOLO
import  cv2
import  cvzone
from pyzbar.pyzbar import decode
import math

# HelperFunctions.py as module of necessary and frequent functions
import HelperFunctions as helpfn

# Load the custom trained weight from the YOLOv8
model= YOLO('weights/best.pt')

# read the image file
img=cv2.imread('images/1.jpg')
img1=img

# since the image size is 4000x3000 which is out of my screen resolution, resizing it
img1 = cv2.resize(img1, (1080, 840))

# show the input image
cv2.imshow("Input image..",img1)


# Run inference on from custom trained model, make show=True to show the YOLO defined bounding boxes
results=model(img,show=False,conf=0.80,iou=0.70,line_thickness=2)

# Make a list of our classes, used while training
classNames=["Item","QR_code","Bar_code"]

# key -value counter (dictionary) for counting frequency of each barcode
barcode_cnt={}

# iterate over results and target the bounding boxes to modify them as per our need
for result in results:
    boxes=result.boxes

    # Iterate over the all bounding boxes and check the class associated with bounding boxes and according take action
    for box in boxes:

        # print(box)
        # Get the diagonal (top-left and bottom-right) coordinates of the current bbox
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        w, h = x2 - x1, y2-y1



        # method 1 to draw bounding box
        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        # cv2.putText(img, f'{classNames[clsId] {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (240, 7, 7), 2)

        # method 2 to draw bounding box
        # cvzone.cornerRect(img, (x1, y1, w, h))
        # cvzone.putTextRect(img,f'{classNames[clsId]} {conf} ',(max(0,x1),max(0,y1)),scale=0.8,thickness=2)

        #class ID and class name
        clsId= int(box.cls[0])
        clsName=classNames[clsId]
        # print(clsId,clsName)

        # confidence of predicted class
        conf=box.conf[0]
        # take upto 2 decimal places
        conf="{:.2f}".format(conf)

        # If the current bbox is for Item class
        if clsName=="Item":
            contain_barcode=False
            # loop over all bbox again and check if the ith_bbox is inside the item region & is ith_bbox a barcode?
            for ith_bbox in boxes:
                clsId_inner_bbox= int(ith_bbox.cls[0])
                if classNames[clsId_inner_bbox]=="Bar_code" or classNames[clsId_inner_bbox]=="QR_code" :

                    # get the percentage of intersection area of the barcode bbox, inside the item bbox
                    perc_area_intr=helpfn.overlap_area(box,ith_bbox)

                    # If the ith_bbox has >=60% area inside the Item bbox
                    if  perc_area_intr>=60:
                        contain_barcode=True

                        # draw a black bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 8)
                        cv2.putText(img,f'Item {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (240,7,7), 2)


                        break

            # check if the barcode is not present inside the Item bbox
            if contain_barcode==False:
                # draw a red bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 8)
                cv2.putText(img,f'Item {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (240,7,7), 2)

        # the current bbox has class either barcode or QR code
        else:

            # crop the region of bbox for barcode or QR code
            roi = img[y1:y1 + h, x1:x1 + w]
            # cv2.imshow("cropped region",roi)

            # detect the barcode or QR code using the decode function of pyzbar
            detections = decode(roi)
            # print(detections)

            # the barcode is not detected, draw a yellow bounding box
            if(len(detections)==0):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 8)
                if classNames[clsId]=="Bar_code":
                    cv2.putText(img,f'Barcode {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (240,7,7), 5)
                else:
                    cv2.putText(img,f'QR {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (240,7,7), 5)

            # the barcode is detected, draw a blue bbox
            else:

                # get the data from the detected bar code
                bar_data=detections[0].data.decode('utf-8')

                # update the count of barcode value
                if bar_data in barcode_cnt:
                    barcode_cnt[bar_data]+=1
                else:
                    barcode_cnt[bar_data]=1

                # draw a blue bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 8)
                if classNames[clsId] == "Bar_code":
                    cv2.putText(img, f'Barcode {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 7, 7), 5)
                else:
                    cv2.putText(img, f'QR {conf} ', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 7, 7), 5)



# print the frequency of each barcode detected
for code in barcode_cnt:
    print(f'The barcode value: {code} and the count is: {barcode_cnt[code]}')

# Finally, show the resized image, having bboxes
img = cv2.resize(img, (1080, 860))
cv2.imshow("Output Image..",img)
cv2.waitKey(0)
cv2.destroyWindow()





