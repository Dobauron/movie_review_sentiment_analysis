import cv2

"""Recognize shape on picture"""
image_3chan = cv2.imread("img/bolt.png")
image_3chan_copy = image_3chan.copy()

#konwert to gray
gray_image= cv2.cvtColor(image_3chan, cv2.COLOR_BGR2GRAY)


#convert to binary image
_, binary_img = cv2.threshold(gray_image,250,255, cv2.THRESH_BINARY)


#search for contours
contours_list, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(0, len(contours_list)):
    contour_info = hierarchy[0][i,:]
    if contour_info[2]== -1 and contour_info[3] == -1:
        with_contours = cv2.drawContours(image_3chan_copy, contours_list,i,[0,255,0],3)
        print("contours finded")
    if contour_info[2]== -1 and contour_info[3] != -1:
        with_contours = cv2.drawContours(with_contours, contours_list,i,[0,0,255],3)


#show image as a gray
cv2.imshow("gray", with_contours)
cv2.waitKey(0)
cv2.destroyWindow()
