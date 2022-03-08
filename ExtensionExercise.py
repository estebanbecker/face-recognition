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

filename_list = [f for f in listdir('celba_limited') if isfile(join('celba_limited',f))]
count=[]

for filename in filename_list:
    label=int(filename.split('_')[0])

    while label > len(count):
        count.append(0)
    
    count[label-1] += 1


train_images=[]
train_labels=[]
test_images=[]
test_labels=[]

t_train_images=[]
t_test_images=[]


for filename in filename_list:
    label=int(filename.split('_')[0])

    if count[label-1] > 19 :

        image = cv2.imread('celba_limited/'+filename)

        resized_image = cv2.resize(image, (160, 160),)


        #Preparing data for DNN

        float_image=resized_image/255.0

        tensor_image = torch.from_numpy(float_image).permute(2, 0, 1).float()

        #Preparing data for LBPH

        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        if random.random() > 0.25:
            t_train_images.append(tensor_image)

            train_images.append(gray)
            train_labels.append(label)
        else:
            t_test_images.append(tensor_image)

            test_images.append(gray)
            test_labels.append(label)

print("Train images: ", len(train_images))
print("Processing images to the DNN...")

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device='cpu'
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

t_train_images = torch.stack(t_train_images).to('cpu')
embeddings_train = resnet(t_train_images).detach().cpu().numpy()

t_test_images = torch.stack(t_test_images).to('cpu')
embeddings_test = resnet(t_test_images).detach().cpu().numpy()

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

#Train the LBPH face recognition model
LBPH_model = cv2.face.LBPHFaceRecognizer_create()
LBPH_model.train(np.array(train_images), np.array(train_labels))

LBPH_model_predicted = []

print("LBPH Face Recognition Model prediction")

for img in test_images:

    LBPH_model_predicted.append(LBPH_model.predict(img))

LBPH_correct = 0

for i in range(len(test_images)):

    if test_labels[i] == LBPH_model_predicted[i][0]:
        LBPH_correct += 1

print("LBPH Face Recognition Model accuracy: ", LBPH_correct/len(test_images))
print("Done")

