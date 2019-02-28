import glob
import re
import h5py
import numpy as np
import cv2
from datetime import datetime
import csv

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

train_x1_img_path_list = []
train_y_img_path_list = []

test_x1_img_path_list = []
test_y_img_path_list = []

validation_x1_img_path_list = []
validation_y_img_path_list = []


train_X2 = []
test_X2 = []
validation_X2 = []

train_parent_directory = sorted(glob.glob('../../../data/train/*/'), key=numericalSort)
test_parent_directory = sorted(glob.glob('../../../data/test/*/'), key=numericalSort)
validation_parent_directory = sorted(glob.glob('../../../data/validation/*/'), key=numericalSort)

#TRAIN DATASET
for i in train_parent_directory:
        directory = i
        img_list = sorted(glob.glob(directory + '*.png'), key=numericalSort)
        
        train_x1_img_path = img_list[0:-1]
        train_y_img_path = img_list[1:]
        train_x1_img_path_list.extend(train_x1_img_path)
        train_y_img_path_list.extend(train_y_img_path)
        
        #calculate time from image file
        f=open("../../../data/train_X2.txt", "a")
        for k, img in enumerate(train_x1_img_path):
            first_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', train_x1_img_path[k])
            first_utc_time = datetime.strptime(first_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            second_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', train_y_img_path[k])
            second_utc_time = datetime.strptime(second_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            
            time = second_utc_time - first_utc_time
            int_time = time.seconds
            train_X2.append(int_time)
            f.write(str(int_time) + '\n')
            data = [[k, int_time, first_utc_time, second_utc_time,train_x1_img_path[k], train_y_img_path[k]]]
            with open('../../../data/train_X2.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['serial', 'time_in_second', 'first_image_time', 'second_image_time', 'first_image_path', 'second_image_path'])
                writer.writerows(data)
        csvFile.close()    
        f.close()
        
#TEST DATASET        
for i in test_parent_directory:
        directory = i
        img_list = sorted(glob.glob(directory + '*.png'), key=numericalSort)
        
        test_x1_img_path = img_list[0:-1]
        test_y_img_path = img_list[1:]
        test_x1_img_path_list.extend(test_x1_img_path)
        test_y_img_path_list.extend(test_y_img_path)
        
        #calculate time from image file
        f=open("../../../data/test_X2.txt", "a")
        for k, img in enumerate(test_x1_img_path):
            first_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', test_x1_img_path[k])
            first_utc_time = datetime.strptime(first_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            second_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', test_y_img_path[k])
            second_utc_time = datetime.strptime(second_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            
            time = second_utc_time - first_utc_time
            int_time = time.seconds
            test_X2.append(int_time)
            f.write(str(int_time) + '\n')
            data = [[k, int_time, first_utc_time, second_utc_time,test_x1_img_path[k], test_y_img_path[k]]]
            with open('../../../data/test_X2.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['serial', 'time_in_second', 'first_image_time', 'second_image_time', 'first_image_path', 'second_image_path'])
                writer.writerows(data)
        csvFile.close()    
        f.close()
        
#Validation DATASET        
for i in validation_parent_directory:
        directory = i
        img_list = sorted(glob.glob(directory + '*.png'), key=numericalSort)
        
        validation_x1_img_path = img_list[0:-1]
        validation_y_img_path = img_list[1:]
        validation_x1_img_path_list.extend(validation_x1_img_path)
        validation_y_img_path_list.extend(validation_y_img_path)
        
        #calculate time from image file
        f=open("../../../data/validation_X2.txt", "a")
        for k, img in enumerate(validation_x1_img_path):
            first_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', validation_x1_img_path[k])
            first_utc_time = datetime.strptime(first_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            second_match = re.search(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_\d{2}', validation_y_img_path[k])
            second_utc_time = datetime.strptime(second_match.group(), "%Y_%m_%d__%H_%M_%S_%f")
            
            time = second_utc_time - first_utc_time
            int_time = time.seconds
            validation_X2.append(int_time)
            f.write(str(int_time) + '\n')
            data = [[k, int_time, first_utc_time, second_utc_time, validation_x1_img_path[k], validation_y_img_path[k]]]
            with open('../../../data/validation_X2.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(['serial', 'time_in_second', 'first_image_time', 'second_image_time', 'first_image_path', 'second_image_path'])
                writer.writerows(data)
        csvFile.close()    
        f.close()        
        
       
train_X1 = train_x1_img_path_list
train_Y = train_y_img_path_list
test_X1 = test_x1_img_path_list
test_Y = test_y_img_path_list  
validation_X1 = validation_x1_img_path_list
validation_Y = validation_y_img_path_list     
        
train_X1_shape = (len(train_X1), 240, 240, 1)

train_Y_shape = (len(train_Y), 240, 240, 1)

test_X1_shape = (len(test_X1), 240, 240, 1)

test_Y_shape = (len(test_Y), 240, 240, 1)

validation_X1_shape = (len(test_X1), 240, 240, 1)

validation_Y_shape = (len(test_Y), 240, 240, 1)



hdf5_path = '../../../data/dataset.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')

#hdf5_file.create_dataset("train_X1", train_X1_shape, dtype="f", compression="gzip", compression_opts=4)

hdf5_file.create_dataset("train_X1", train_X1_shape, dtype="f")
hdf5_file.create_dataset("train_Y", train_Y_shape, dtype="f")
hdf5_file.create_dataset("test_X1", test_X1_shape, dtype="f")
hdf5_file.create_dataset("test_Y", test_Y_shape, dtype="f")
hdf5_file.create_dataset("validation_X1", test_X1_shape, dtype="f")
hdf5_file.create_dataset("validation_Y", test_Y_shape, dtype="f")
hdf5_file.create_dataset("train_x2", data=train_X2)
hdf5_file.create_dataset("test_x2", data=test_X2)
hdf5_file.create_dataset("validation_x2", data=test_X2)


# loop over train_x1 addresses
for i in range(len(train_X1)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_X1)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_X1[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    # save the image 
    hdf5_file["train_X1"][i, ...] = img[None]
    
    
    # loop over train_Y addresses
for i in range(len(train_Y)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_Y)))
    # read an image and resize to (120, 120)
    # cv2 load images as BGR, convert it to RGB
    addr = train_Y[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.reshape(240, 240, 1)
    
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    
    # save the image
    hdf5_file["train_Y"][i, ...] = img[None]
    
    
# loop over test_X1 addresses
for i in range(len(test_X1)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_X1)))
    # read an image and resize to (120, 120)
    # cv2 load images as BGR, convert it to RGB
    addr = test_X1[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.reshape(240, 240, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    # save the image
    hdf5_file["test_X1"][i, ...] = img[None]


# loop over test_Y addresses
for i in range(len(test_Y)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_Y)))
    # read an image and resize to (120, 120)
    # cv2 load images as BGR, convert it to RGB
    addr = test_Y[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.reshape(240, 240, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    # save the image
    hdf5_file["test_Y"][i, ...] = img[None]
    
    
    
        
# loop over validation_X1 addresses
for i in range(len(validation_X1)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Validation data: {}/{}'.format(i, len(validation_X1)))
    # read an image and resize to (120, 120)
    # cv2 load images as BGR, convert it to RGB
    addr = validation_X1[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.reshape(240, 240, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    # save the image
    hdf5_file["validation_X1"][i, ...] = img[None]


# loop over validation_Y addresses
for i in range(len(validation_Y)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Validation data: {}/{}'.format(i, len(validation_Y)))
    # read an image and resize to (120, 120)
    # cv2 load images as BGR, convert it to RGB
    addr = validation_Y[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (240, 240), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.reshape(240, 240, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img = img.reshape(240, 240, 1)
    # save the image
    hdf5_file["validation_Y"][i, ...] = img[None]
    
    
hdf5_file.close()
