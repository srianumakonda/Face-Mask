import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_hub as hub
from PIL import Image
import time
from sklearn.metrics import confusion_matrix


train_dir = ".\\dataset\\Train\\"
val_dir = ".\\dataset\\Validation\\"
test_dir= ".\\dataset\\Test\\"

img_size = 128
batch_size = 64
epochs = 10
tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")

augment_train_data = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.15,
                                        zoom_range=0.15,
                                        horizontal_flip=True)
augment_val_data = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.25,
                                      zoom_range=0.25,
                                      horizontal_flip=True)
augment_test_data = ImageDataGenerator(rescale=1./255,)

train_set = augment_train_data.flow_from_directory(train_dir,
                                                   target_size=(img_size,img_size),
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   class_mode='categorical')
val_set = augment_val_data.flow_from_directory(val_dir,
                                               target_size=(img_size,img_size),
                                               batch_size=batch_size,
                                               shuffle=False,
                                               class_mode='categorical')
test_set = augment_val_data.flow_from_directory(test_dir,
                                                target_size=(img_size,img_size),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                class_mode='categorical')

inception_model = tf.keras.applications.mobilenet.MobileNet()


inception_model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("Model summary: ")
inception_model.summary()

# print("Model training: ")
# inception_model.fit(train_set,
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_data=val_set,
#                     callbacks=[tensorboard])

# print("Model save: ")
# inception_model.save("transfer-learning-model.h5")
# print("Done")

# print("Model evaluation: ")
# inception_model.evaluate(test_set)

# predictions = inception_model.predict(test_set)
# y_pred = np.argmax(predictions, axis=1)
# print('Confusion Matrix: ')
# print(confusion_matrix(test_set.classes, y_pred))

imported_model = tf.keras.models.load_model("transfer-learning-model.h5",custom_objects={'KerasLayer': hub.KerasLayer})
labels = {0:"Mask", 1:"No mask"}

fig = plt.figure(figsize=(20,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    test_img = np.array(test_set[3][0][i])
    test_img = np.array(test_img).reshape(1,img_size,img_size,3)
    plt.imshow(np.squeeze(test_img))
    plt.title(f"Predicted result:{labels[np.argmax(imported_model.predict(test_img))]}")

plt.show()