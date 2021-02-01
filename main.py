import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_addons as tfa
from PIL import Image, ImageOps
import time

from sklearn.metrics import confusion_matrix


train_dir = ".\\dataset\\Train\\"
val_dir = ".\\dataset\\Validation\\"
test_dir= ".\\dataset\\Test\\"

img_size = 128
batch_size = 64
epochs = 10
tensorboard = TensorBoard(log_dir=f"logs/{time.time()}")

# plt.bar(["Train", "Validation", "Test"], [sum([len(files) for r, d, files in os.walk(train_dir)]), sum([len(files) for r, d, files in os.walk(val_dir)]), sum([len(files) for r, d, files in os.walk(test_dir)])])

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

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape=(img_size,img_size,3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("Model train: ")
model.fit(train_set,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=val_set,
          callbacks=[tensorboard])

print("Model save: ")
model.save("model.h5")
print("Done")

print("Model evaluation: ")
model.evaluate(test_set)

predictions = model.predict(test_set)
y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix: ')
print(confusion_matrix(test_set.classes, y_pred))

imported_model = tf.keras.models.load_model("model.h5")
labels = {0:"Mask", 1:"No mask"}

fig = plt.figure(figsize=(20,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    test_img = np.array(test_set[i+2][0][i+15])
    test_img = np.array(test_img).reshape(1,img_size,img_size,3)
    prediction = imported_model.predict(test_img)
    plt.imshow(np.squeeze(test_img),cmap="gray")
    plt.title(f"Predicted result:{prediction}")

plt.show()