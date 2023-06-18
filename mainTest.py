import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model("BrainTumor10EpochsCategorical.h5")

image=cv2.imread('C:\\Users\\jeeva\\Downloads\\BrainTumor Classification2\\BrainTumor Classification DL\\datasets\\yes\\y9.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img, axis=0)
result=model.predict_step(input_img)
predict_class=np.argmax(result, axis=1)
print(predict_class)




