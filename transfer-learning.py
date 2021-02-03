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
from PIL import Image, ImageOps
import time
from sklearn.metrics import confusion_matrix


train_dir = ".\\dataset\\Train\\"
val_dir = ".\\dataset\\Validation\\"
test_dir= ".\\dataset\\Test\\"

img_size = 224
batch_size = 32
epochs = 10
def create_data(data_subset,img_size=128,rot_range=None,flip=True):
    classes = ["Mask", "Non Mask"]
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

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
download_model = hub.KerasLayer(url,input_shape=(img_size,img_size,3))
download_model.trainable = False
inception_model = Sequential([
    download_model,
    Dense(2,activation="softmax")
])

inception_model.compile(optimizer="adam",
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# print("Model summary: ")
print(inception_model.summary())

print("Model training: ")
inception_model.fit(train_set,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=val_set,
                    callbacks=[tensorboard])

print("Model save: ")
inception_model.save("transfer-learning-model.h5")
print("Done")

print("Model evaluation: ")
inception_model.evaluate(test_set)

predictions = inception_model.predict(test_set)
y_pred = np.argmax(predictions, axis=1)
print('Confusion Matrix: ')
print(confusion_matrix(test_set.classes, y_pred))

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
