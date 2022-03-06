from fileinput import filename
from genericpath import isfile
from os.path import join
from os import listdir 
import csv
import scipy.io as sio
import cv2
import random
import numpy as np
import sys

random.seed()

print("Preraring the data...")

filename_list = [f for f in listdir('caltech') if isfile(join('caltech',f))]

with open('caltech/caltech_labels.csv') as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))

    count=[]
    for row in csv_reader:
        if int(row[0]) == len(count):
            count[int(row[0])-1]=count[int(row[0])-1]+1
        else:
            count.append(1)

sio.loadmat('caltech/ImageData.mat')

train_images=[]
train_labels=[]
test_images=[]
test_labels=[]


img_data=sio.loadmat("caltech/ImageData.mat")

for filename in filename_list:
    if filename.startswith('i'):

        im_num=int(filename[6:10])

        if count[int(csv_reader[im_num-1][0])-1] > 19 :

            image = cv2.imread('caltech/'+filename)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            start_row=int(img_data['SubDir_Data'][3][im_num-1])
            end_row=int(img_data['SubDir_Data'][7][im_num-1])
            start_col=int(img_data['SubDir_Data'][2][im_num-1])
            end_col=int(img_data['SubDir_Data'][6][im_num-1])

            scrop_img = gray[start_row:end_row, start_col:end_col]

            resized_image = cv2.resize(scrop_img, (70, 100),)

            if random.random() > 0.25:
                train_images.append(resized_image)
                train_labels.append(int(csv_reader[im_num-1][0]))
            else:
                test_images.append(resized_image)
                test_labels.append(int(csv_reader[im_num-1][0]))

print("Train images: ", len(train_images))

#Train the eigen face recognition model
EIFR_model = cv2.face.EigenFaceRecognizer_create()
EIFR_model.train(np.array(train_images), np.array(train_labels))

#Train the Fisher face recognition model
Fisher_model = cv2.face.FisherFaceRecognizer_create()
Fisher_model.train(np.array(train_images), np.array(train_labels))

#Train the LBPH face recognition model
LBPH_model = cv2.face.LBPHFaceRecognizer_create()
LBPH_model.train(np.array(train_images), np.array(train_labels))

#Test the models
EIFR_predicted = []

print("Eigen Face Recognition Model prediction")

for img in test_images:

    EIFR_predicted.append(EIFR_model.predict(img))

Fisher_model_predicted = []

print("Fisher Face Recognition Model prediction")

for img in test_images:

    Fisher_model_predicted.append(Fisher_model.predict(img))


LBPH_model_predicted = []

print("LBPH Face Recognition Model prediction")

for img in test_images:

    LBPH_model_predicted.append(LBPH_model.predict(img))

#Evaluate the models
EIFR_correct = 0
Fisher_correct = 0
LBPH_correct = 0

for i in range(len(test_images)):

    if test_labels[i] == EIFR_predicted[i][0]:
        EIFR_correct += 1

    if test_labels[i] == Fisher_model_predicted[i][0]:
        Fisher_correct

    if test_labels[i] == LBPH_model_predicted[i][0]:
        LBPH_correct += 1

print("Eigen Face Recognition Model accuracy: ", EIFR_correct/len(test_images))
print("Fisher Face Recognition Model accuracy: ", Fisher_correct/len(test_images))
print("LBPH Face Recognition Model accuracy: ", LBPH_correct/len(test_images))
