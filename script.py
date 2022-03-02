from fileinput import filename
from genericpath import isfile
from os.path import join
from os import listdir 
import csv
from scipy.io import loadmat

filename_list = [f for f in listdir('caltech') if isfile(join('caltech',f))]

with open('caltech/caltech_labels.csv') as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))

    count=[]
    for row in csv_reader:
        if int(row[0]) == len(count):
            count[int(row[0])-1]=count[int(row[0])-1]+1
        else:
            count.append(1)
    print(count)

loadmat('caltech/ImageData.mat')

train_image=[]
train_label=[]
test_images=[]
test_labels=[]


for filename in filename_list:
    if filename.startswith('i'):
        if count[int(csv_reader[int(filename[6:10])-1][0])-1] > 19 :
            print(int(filename[6:10]))
