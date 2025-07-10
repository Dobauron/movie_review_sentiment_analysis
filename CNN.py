from keras.applications import VGG19, imagenet_utils
from keras.utils import img_to_array, load_img
import numpy as np
import cv2

img_path = "img/frog.jpg"
img = load_img(img_path)


#resize obrazu, dzielenie na małe kwadraty, które są konwertowane na tablice
img = img.resize((224,224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

#preprocessing
img_array = imagenet_utils.preprocess_input(img_array)

#load VGG
pretrained_model = VGG19(weights="imagenet")


prediction = pretrained_model.predict(img_array)
actual_prediction= imagenet_utils.decode_predictions(prediction)
print(f"Rozpoznane obiekty - {actual_prediction[0][0][1]} "
      f"\nDokładnośc - {actual_prediction[0][0][2]*100}")
disp_img = cv2.imread(img_path)
cv2.putText(disp_img,actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX,0.8,(0,0,0))
cv2.imshow(disp_img)
