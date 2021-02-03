#Put code below if u don't want tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import time
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

img_size = 224
batch_size = 32
epochs = 10

def create_data(data_subset,img_size=128,rot_range=None,flip=True):
    classes = ["WithMask", "WithoutMask"]
    if data_subset == "train":
        subset_path = ".\\dataset\\Train\\"
    elif data_subset == "val":
        subset_path = ".\\dataset\\Validation\\"
    elif data_subset == "test":
        subset_path = ".\\dataset\\Test\\"
    else:
        raise KeyError("Input is now one of the following categories: train, val, and/or test.")

    X = []
    y = []
    for class_option in classes:
        for filename in os.listdir(os.path.join(subset_path,class_option)):
            counter = 0
            img_path = os.path.join(os.path.join(subset_path,class_option),filename)
            open_img = Image.open(img_path).resize((img_size,img_size))
            X.append(np.asarray(open_img))
            counter += 1

            if flip == True:
                flip_img = ImageOps.flip(open_img)
                mirror_img = ImageOps.mirror(open_img)
                
                X.append(np.asarray(flip_img))
                X.append(np.asarray(mirror_img))
                counter += 2
            
            if rot_range != None:
                if rot_range%10 != 0:
                    raise ValueError("Number must be divisable by 10. Please input a number that is divisable by 10.")
                    for i in range(10,rot_range+10,10):
                        rot_img = open_img.rotate(i)
                        X.append(np.asarray(rot_img))
                        counter += 1

            for i in range(counter):
                if classes.index(class_option) == 0:
                    y.append([1,0])
                else:
                    y.append([0,1])

    normalize_X = np.asarray(X) / 255.0
    return normalize_X,np.asarray(y)


X_train, y_train = create_data("train",img_size=img_size,rot_range=70,flip=True)
X_val, y_val = create_data("val",img_size=img_size,rot_range=40,flip=True)
X_test, y_test = create_data("test",img_size=img_size,rot_range=None,flip=False)

model = Sequential()
model.add(Conv2D(16,(3,3), padding='same',input_shape=(img_size,img_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=1e-5),
              metrics=["accuracy"])

print("Model train: ")
model.fit(x=X_train,y=y_train,
          validation_data=(X_val,y_val),
          epochs=epochs,
          batch_size=batch_size)

print("Model save: ")
model.save("model.h5")
print("Done")

print("Model evaluation: ")
model.evaluate(x=X_test,y=y_test)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix: ')
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

imported_model = tf.keras.models.load_model("model.h5")
labels = {0:"Mask", 1:"No mask"}

fig = plt.figure(figsize=(20,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    test_img = Image.fromarray(np.uint8(X_test[i+585]*255.0))
    test_img = np.array(test_img).reshape(1,img_size,img_size,3)
    prediction = imported_model.predict(test_img)
    plt.imshow(np.squeeze(test_img),cmap="gray")
    plt.title(f"Predicted result:{prediction}")

plt.show()