from fileinput import filename
from genericpath import isfile
from os.path import join
from os import listdir 
import csv
import scipy.io as sio
import cv2
import random
import numpy as np

random.seed()

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
EIFR_predicted_label = []
EIFR_predicted_confidence = []

EIFR_predicted_label = EIFR_model.predict(np.array(test_images))
EIFR_predicted_confidence = EIFR_model.predict(np.array(test_images))

print("Eigen Face Recognition Model")
print("Predicted labels: " + str(EIFR_predicted_label))
print("Predicted confidences: " + str(EIFR_predicted_confidence))

