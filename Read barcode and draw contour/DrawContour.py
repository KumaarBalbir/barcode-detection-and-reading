import cv2
img = cv2.imread('images/1.jpg')

# convert the image into gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold the gray image
ret,thresh = cv2.threshold(gray,50,150,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
hierarchy=hierarchy[0]

print("Number of contours detected:", len(contours))

# iterate over the contours and hierarchy
for cnt,hier in zip(contours,hierarchy):
    # print the contour having area greater than 1000
   if cv2.contourArea(cnt) > 1000:
       hull = cv2.convexHull(cnt)

       # draw the contours
       # img = cv2.drawContours(img,[cnt],0,(0,255,0),5)

       # draw the contours as hull
       img = cv2.drawContours(img,[hull],0,(0,0,255),8)

# Display the image
img=cv2.resize(img,(1080,840))
cv2.imshow("Convex Hull", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

