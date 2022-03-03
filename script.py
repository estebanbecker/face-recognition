from fileinput import filename
from genericpath import isfile
from os.path import join
from os import listdir 
import csv
import scipy.io as sio
import cv2
import random

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

train_image=[]
train_label=[]
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

            resized_image = cv2.resize(scrop_img, (70, 100))

            if random.random() > 0.25:
                train_image.append(resized_image)
                train_label.append(int(csv_reader[im_num-1][0]))
            else:
                test_images.append(resized_image)
                test_labels.append(int(csv_reader[im_num-1][0]))


            
