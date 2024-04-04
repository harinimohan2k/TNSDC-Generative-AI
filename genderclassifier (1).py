#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
os.chdir(os.getcwd())


# In[10]:


#limit VRAM usage
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[11]:


#check and remove dodgy images from dataset
import cv2
import imghdr

data_dir = 'datasets/dataset1/data'
image_exts = ['jpg', 'jpeg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        if image_path.endswith(('.csv', '.py')):
            continue
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image extension not supported: ', format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image: ', format(image_path))


# In[12]:


#creating a dataset using keras and all the images in the data folder
import numpy as np
from matplotlib import pyplot as plt

#building a data pipeline
data = tf.keras.utils.image_dataset_from_directory('datasets/dataset1/data')
validation = tf.keras.utils.image_dataset_from_directory('datasets/dataset1/validation')

#scaling the data to be between 0 and 1 from 0 to 255
#logic - scaled = batch[0] / 255 but we apply this directly to the pipeline
data = data.map(lambda x, y: (x / 255, y)) # type: ignore
validation = validation.map(lambda x, y: (x / 255, y)) # type: ignore


# In[13]:


#making it an interator to allow us to loop through the data
data_iterator = data.as_numpy_iterator() # type: ignore
validation_iterator = validation.as_numpy_iterator() # type: ignore
#grab a batch of data
batch = data_iterator.next()
batch1 = validation_iterator.next()
#batch[0].shape #shape of the images
print(batch[1]) 
print(batch1[1])
#Classification labels: 0 - Female, 1 - Male.


# In[14]:


fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    #ax[idx].imshow(img.astype(int))
    ax[idx].set_title(batch[1][idx])
    #ax[idx].title.set_text(batch[1][idx])


# In[15]:


print('Data batches: ',len(data)) #number of batches of data
print('Validation data batches: ',len(validation)) #number of batches of validation data


# In[16]:


#making training, validation and test sets.
train_size = len(data)
val_size = int(0.8 * len(validation))
test_size = int(0.2 * len(validation))
print(val_size)
train = data.take(train_size) # type: ignore
val = validation.take(val_size) # type: ignore
test = validation.skip(val_size).take(test_size) # type: ignore


# In[17]:


#building the model
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()


# In[19]:


#training the model
logdir = 'logs'
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
#hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])
hist = model.fit(train, epochs=40, validation_data=val, callbacks=[early_stopping])


# In[20]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['accuracy'], color='green', label='accuracy')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[21]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[22]:


#Evaluate the model performance
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy # type: ignore
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
len(test)


# In[23]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[24]:


print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')


# In[25]:


#Test the model
img = cv2.imread('ftest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[26]:


resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int)) # type: ignore
plt.show()


# In[27]:


np.expand_dims(resize, 0).shape
yhat = model.predict(np.expand_dims(resize/255, 0)) # type: ignore
print(yhat)
if yhat>0.5:
    print('prediction: Male')
else:
    print('prediction: Female')


# In[28]:


#Save the model for future use.
from tensorflow.keras.models import load_model  # type: ignore
model.save(os.path.join('models','gender_classifier.h5'))


# In[29]:


#Load the saved model
new_model = load_model(os.path.join('models','gender_classifier.h5'))
yhatnew = new_model.predict(np.expand_dims(resize/255, 0)) # type: ignore
if yhat>0.5:
    print('prediction: Male')
else:
    print('prediction: Female')

