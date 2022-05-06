#TechVidvan load all required libraries
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

#specify number
num_feeling = 7
im_size = 224
batch_size = 64
encoder = LabelEncoder()

#read the csv file
df_labels = pd.read_csv("languageLabels.csv")
#store training and testing images folder location
train_file = 'train/'
test_file = 'test/'

#check the total number of unique breed in our dataset file
print("Total number of unique Dog body languages :", len(df_labels.feeling.unique()))

#get only 60 unique breeds record
feeling_dict = list(df_labels['feeling'].value_counts().keys())
# new_list = sorted(feeling_dict, reverse=True)[:num_feeling*2+1:2]
new_list = sorted(feeling_dict, reverse=True)
#change the dataset to have only those 60 unique breed records
# df_labels = df_labels.query('feeling in @new_list')
#create new column which will contain image name with the image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")

#create a numpy array of the shape
#(number of dataset records, image size , image size, 3 for rgb channel ayer)
#this will be input for model
train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')

#iterate over img_file column of our dataset
for i, img_id in enumerate(df_labels['img_file']):
  #read the image file and convert into numeric format
  #resize all images to one dimension i.e. 224x224
  #we will get array with the shape of
  #(224,224,3) where 3 is the RGB channels layers
  img = cv2.resize(cv2.imread(train_file+img_id,
                   cv2.IMREAD_COLOR), ((im_size, im_size)))
  #scale array into the range of -1 to 1.
  #preprocess the array and expand its dimension on the axis 0
  img_array = preprocess_input(np.expand_dims(
    np.array(img[..., ::-1].astype(np.float32)).copy(), axis=0))
  #update the train_x variable with new element
  train_x[i] = img_array

#this will be target for model.
#convert breed names into numerical format
train_y = encoder.fit_transform(df_labels["feeling"].values)

#split the dataset in the ratio of 80:20.
#80% for training and 20% for testing purpose
x_train, x_test, y_train, y_test = train_test_split(
  train_x, train_y, test_size=0.2, random_state=42)

#Image augmentation using ImageDataGenerator class
train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

#generate images for training sets
train_generator = train_datagen.flow(x_train,
                                     y_train,
                                     batch_size=batch_size)

#same process for Testing sets also by declaring the instance
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(x_test,
                                   y_test,
                                   batch_size=batch_size)

#building the model using ResNet50V2 with input shape of our image array
#weights for our network will be from of imagenet dataset
#we will not include the first Dense layer
resnet = ResNet50V2(
  input_shape=[im_size, im_size, 3], weights='imagenet', include_top=False)
#freeze all trainable layers and train only top layers
for layer in resnet.layers:
    layer.trainable = False

#add global average pooling layer and Batch Normalization layer
x = resnet.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
#add fully connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

#add output layer having the shape equal to number of breeds
predictions = Dense(num_feeling, activation='softmax')(x)

#create model class with inputs and outputs
model = Model(inputs=resnet.input, outputs=predictions)
#model.summary()

#epochs for model training and learning rate for optimizer
epochs = 25
learning_rate = 1e-3

#using RMSprop optimizer compile or build the model
optimizer = RMSprop(learning_rate=learning_rate, rho=0.9)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

#fit the training generator data and train the model
model.fit(train_generator,
          steps_per_epoch=x_train.shape[0] // batch_size,
          epochs=epochs,
          validation_data=test_generator,
          validation_steps=x_test.shape[0] // batch_size)


saved_model_dir = './liteModel/BodyLanguageModel'

#Save the model for prediction
model.save(saved_model_dir)
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('LanguageModel.tflite', 'wb') as f:
  f.write(tflite_model)

#load the model
model = load_model(saved_model_dir)

#get the image of the dog for prediction
pred_img_path = 'DogImages/anxious2.jpg'
#read the image file and convert into numeric format
#resize all images to one dimension i.e. 224x224
pred_img_array = cv2.resize(cv2.imread(
  pred_img_path, cv2.IMREAD_COLOR), ((im_size, im_size)))
#scale array into the range of -1 to 1.
#expand the dimesion on the axis 0 and normalize the array values
pred_img_array = preprocess_input(np.expand_dims(
  np.array(pred_img_array[..., ::-1].astype(np.float32)).copy(), axis=0))

#feed the model with the image array for prediction
pred_val = model.predict(np.array(pred_img_array, dtype="float32"))
#display the image of dog
cv2.imshow("TechVidvan", cv2.resize(cv2.imread(
  pred_img_path, cv2.IMREAD_COLOR), ((im_size, im_size))))
#display the predicted breed of dog
pred_feeling = sorted(new_list)[np.argmax(pred_val)]
print("Emotion for this Dog is :", pred_feeling)
