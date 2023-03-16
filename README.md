# barcode-detection-and-reading--Mowito
#### Note: Checkout the master branch. As of now, all the code is committed in that branch and is not merged to main.
The following repository is a result of assignment provided by the Robotics and Warehouse automation company Mowito. 

## Detects the products/items and barcode or QR code in images using opencv,pyzbar & YOLOv8.
For detailed approach, please visit here:
<a href="https://docs.google.com/document/d/1AV1Bz4Qp4_dcAgq6n8fidlKm4TV8mltx9AX2zrdbciQ/edit?usp=sharing" target="_blank">Google doc</a>

### a. Key results 
* I tried to make bbox using only opencv. But it was very difficult to approximate the contours and make a convex hull around each item.
  ![Screenshot1](https://github.com/KumaarBalbir/barcode-detection-and-reading--Mowito/blob/master/supporting%20screenshots/draw%20contours.png) 
  
  ![Screenshot2](https://github.com/KumaarBalbir/barcode-detection-and-reading--Mowito/blob/master/supporting%20screenshots/draw%20convex%20hull.png) 
  
* I used pre-trained model YOLO version 8 but that model was not trained for custom object detection. So that didn't work.
* Augmented the sample images using labelImg library as there were very less number of images to train the YOLO model.
![Screenshot3](https://github.com/KumaarBalbir/barcode-detection-and-reading--Mowito/blob/master/supporting%20screenshots/labelImg_annotation.png)
* Trained the YOLO model on augmented images, and made inference:
![Screenshot4](https://github.com/KumaarBalbir/barcode-detection-and-reading--Mowito/blob/master/supporting%20screenshots/detected%20bbox%20custom%20yolo.png) 
* Barcode frequency with its value:
![Screenshot5](https://github.com/KumaarBalbir/barcode-detection-and-reading--Mowito/blob/master/supporting%20screenshots/barcode%20cnt%20ss.png) 

### b. For the remaining results do check the 'results custom yolov8' and  'supporting screenshots'  directory 

### c. To run this repository on your local system
* Clone this repository or download it. 
* make a virtual environment in python 3.7
* install the all libraries reqirements.txt
* Run yolo_custom.py file.
* You will see the 2 window as outcome: one will be input image.. another one will be output image..


  

