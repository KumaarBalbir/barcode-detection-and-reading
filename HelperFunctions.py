import cv2
import  cvzone
import  numpy
from ultralytics import  YOLO
from pyzbar.pyzbar import decode

# function to resize the image to the original ratio
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# function to check if the one bounding box is another bounding box
def overlap_area(bbox_outer,bbox_inner):

    # get the diagonal coordinate of the item bounding box
    x1_outer,y1_outer, x2_outer, y2_outer = bbox_outer.xyxy[0]
    x1_outer, y1_outer, x2_outer, y2_outer = int(x1_outer), int(y1_outer), int(x2_outer), int(y2_outer)

    # get the diagonal coordinate of the ith_bbox bounding box
    x1_inner,y1_inner, x2_inner, y2_inner = bbox_inner.xyxy[0]
    x1_inner, y1_inner, x2_inner, y2_inner = int(x1_inner), int(y1_inner), int(x2_inner), int(y2_inner)


    # get the diagonal coordinate of the intersected area
    xA = max(x1_outer, x1_inner)
    yA = max(y1_outer, y1_inner)
    xB = min(x2_outer, x2_inner)
    yB = min(y2_outer, y2_inner)

    # compute the area of intersection rectangle
    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    if interArea == 0:
        return 0
    else:
        area_ith_bbox = (x2_inner-x1_inner) * (y2_inner-y1_inner)
        perc_area_overlap = int((interArea / area_ith_bbox) * 100)
        return perc_area_overlap










