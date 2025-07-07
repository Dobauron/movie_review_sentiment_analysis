import pytesseract
import cv2
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


#wczytanie obrazu
image_to_ocr = cv2.imread("img/Dobromir Matuszak - CV .pdf")

#przeprocesowanie obrazu

preprocessed_img = cv2.cvtColor(image_to_ocr, cv2.COLOR_BGR2GRAY)
_, preprocessed_img= cv2.threshold(preprocessed_img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
preprocessed_img= cv2.medianBlur(preprocessed_img, 3)

cv2.imwrite('temp_img.jpg', preprocessed_img)

preprocessed_pil_img = Image.open('temp_img.jpg')
text_extract = pytesseract.image_to_string(preprocessed_pil_img)

print(text_extract)