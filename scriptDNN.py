from fileinput import filename
from genericpath import isfile
from os.path import join
from os import listdir
import csv, cv2 , random, torch
import scipy.io as sio
import numpy as np
import sys


from torch import tensor

from facenet_pytorch import MTCNN, InceptionResnetV1

random.seed()

print("Preparing the data...")

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

            start_row=int(img_data['SubDir_Data'][3][im_num-1])
            end_row=int(img_data['SubDir_Data'][7][im_num-1])
            start_col=int(img_data['SubDir_Data'][2][im_num-1])
            end_col=int(img_data['SubDir_Data'][6][im_num-1])

            scrop_img = image[start_row:end_row, start_col:end_col]

            resized_image = cv2.resize(scrop_img, (160, 160),)

            float_image=resized_image/255.0

            tensor_image = torch.from_numpy(float_image).permute(2, 0, 1).float()

            if random.random() > 0.25:
                train_images.append(tensor_image)
                train_labels.append(int(csv_reader[im_num-1][0]))
            else:
                test_images.append(tensor_image)
                test_labels.append(int(csv_reader[im_num-1][0]))

print("Train images: ", len(train_images))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu'
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

train_images = torch.stack(train_images).to('cpu')
embeddings_train = resnet(train_images).detach().cpu().numpy()

test_images = torch.stack(test_images).to('cpu')
embeddings_test = resnet(test_images).detach().cpu().numpy()

#Create a SVM

print('Train SVM')

svm = cv2.ml.SVM_create()

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

svm.train(np.array(embeddings_train), cv2.ml.ROW_SAMPLE, np.array(train_labels))

svm_predicted = []

for image in embeddings_test:
    image = image.reshape(1, -1)

    svm_predicted.append(svm.predict(np.array(image)))

svm_correct = 0

for i in range(len(test_images)):

    if test_labels[i] == svm_predicted[i][1][0][0]:
        svm_correct += 1

print("DNN+SVM Face Recognition Model accuracy: ", svm_correct/len(test_images))

print("Done")

