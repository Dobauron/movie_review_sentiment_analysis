import pytesseract
import cv2
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

"""Recognize shape on picture"""
image = cv2.imread("img/shapes.jpg")
#konwert to gray
gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#convert to binary image
_, binary_img = cv2.threshold(gray_image,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


#invert binary img
inverted_binary_img = ~binary_img

#search for contours
contours, hierarchy = cv2.findContours(inverted_binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#draw contours
with_contours = cv2.drawContours(image, contours,-1,(0,255,0),1)

#show image as a gray
cv2.imshow("gray", with_contours)
cv2.waitKey(0)
cv2.destroyWindow()


"""read text from picture"""
#wczytanie obrazu
image_to_ocr = cv2.imread("img/images.jpg")




#przeprocesowanie obrazu

preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)
_, preprocessed_img= cv2.threshold(preprocessed_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
preprocessed_img= cv2.medianBlur(preprocessed_img, 3)

cv2.imwrite('temp_img.jpg', preprocessed_img)

preprocessed_pil_img = Image.open('temp_img.jpg')
text_extract = pytesseract.image_to_string(preprocessed_pil_img)

print(text_extract)