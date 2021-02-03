import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub 
import tensorflow as tf
import cv2 
import numpy as np
import time

img_size = 128
cap = cv2.VideoCapture(0)
# load_model = tf.keras.models.load_model("model.h5",custom_objects={'KerasLayer': hub.KerasLayer})
load_model = tf.keras.models.load_model("model.h5")
labels = {0:"No Mask", 1:"Mask"}

while True:
    ret, frame = cap.read()
    resize_img = cv2.resize(frame,(img_size,img_size))
    img_for_model = resize_img / 255.0
    prediction = load_model.predict(np.array(img_for_model).reshape(1,img_size,img_size,3))
    cv2.imshow("frame",resize_img)
    print(f"Predicted result:{prediction}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()