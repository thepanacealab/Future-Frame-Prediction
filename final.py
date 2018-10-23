from random import shuffle
import glob
import cv2
import numpy as np
import h5py


import tensorflow as tf
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.InteractiveSession()

def CV_to_NN(x):
    return np.array( x / 255.0, dtype=np.float32) 

def NN_to_CV(x):
    return np.array(255 * x, dtype=np.uint8)


hdf5_path = 'dataset.hdf5'

train_img_x1_path = 'walking/nobg/train_x1/*.jpg'
train_img_y_path = 'walking/nobg/train_y/*.jpg'
test_img_x1_path = 'walking/nobg/test_x1/*.jpg'
test_img_y_path = 'walking/nobg/test_y/*.jpg'

train_x1_addrs = glob.glob(train_img_x1_path)
train_y_addrs = glob.glob(train_img_y_path)
test_x1_addrs = glob.glob(test_img_x1_path)
test_y_addrs = glob.glob(test_img_y_path)

train_X1 = train_x1_addrs[0:int(len(train_x1_addrs))]
train_Y = train_y_addrs[0:int(len(train_y_addrs))]
test_X1 = test_x1_addrs[0:int(len(test_x1_addrs))]
test_Y = test_y_addrs[0:int(len(test_y_addrs))]

train_X1_shape = (len(train_X1), 120, 120, 1)
train_X2_shape = (len(train_X1))
train_Y_shape = (len(train_Y), 120, 120, 1)

test_X1_shape = (len(test_X1), 120, 120, 1)
test_X2_shape = (len(test_X1))
test_Y_shape = (len(test_Y), 120, 120, 1)

hdf5_file = h5py.File(hdf5_path, mode='w')


hdf5_file.create_dataset("train_X1", train_X1_shape, np.int8)
hdf5_file.create_dataset("train_Y", train_Y_shape, np.int8)
hdf5_file.create_dataset("test_X1", test_X1_shape, np.int8)
hdf5_file.create_dataset("test_Y", test_Y_shape, np.int8)

#Create train_X2
train_X2 = []
for x in range(len(train_X1)):
    num = x*40
    train_X2.append(num) 


hdf5_file.create_dataset("train_x2", data=train_X2)

#Create test_X2
test_X2 = []
for x in range(len(test_X1)):
    num = x*40
    test_X2.append(num) 
    
hdf5_file.create_dataset("test_x2", data=test_X2)



# loop over train_x1 addresses
for i in range(len(train_X1)):
    # print how many images are saved every 1000 images
    if i % 10 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_X1)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_X1[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(120, 120, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
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
    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(120, 120, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
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
    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(120, 120, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
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
    img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(120, 120, 1)
    # add any image pre-processing here
    #img= img.astype(np.float32)/255
    # save the image
    hdf5_file["test_Y"][i, ...] = img[None]
    
hdf5_file.close()

#Read The Dataset.h5
fname_in = "dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    train_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    train_x2 = np.array(hf.get('train_x2'), dtype=np.float32)
    train_Y  = np.array(hf.get('train_Y'), dtype=np.float32)

    test_X1 = np.array(hf.get('test_X1'), dtype=np.float32)
    test_x2 = np.array(hf.get('test_x2'), dtype=np.float32)
    test_Y  = np.array(hf.get('test_Y'), dtype=np.float32)

batch_size = 16 #16


def data_iterator():
    assert train_X1.shape[0] == train_Y.shape[0]
    assert train_X1.shape[0] == train_x2.shape[0]
    while True:
        idxs = np.arange(0, train_X1.shape[0])
        np.random.shuffle(idxs)
        shuf_X1 = train_X1[idxs]
        shuf_x2 = train_x2[idxs]
        shuf_x2 = shuf_x2.reshape([-1,1])
        shuf_Y  = train_Y[idxs]
        for batch_idx in range(0, train_X1.shape[0], batch_size):
            if batch_idx+batch_size > train_X1.shape[0]:
                break
            yield shuf_X1[batch_idx:batch_idx+batch_size], shuf_x2[batch_idx:batch_idx+batch_size], shuf_Y[batch_idx:batch_idx+batch_size]

            
data_iter = data_iterator()
data_iter


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def pool_2x2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


data_iter = data_iterator()


input_X = tf.placeholder(dtype=tf.float32, shape=[None, 120, 120, 1])
#input_x2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
input_x2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

input_Y = tf.placeholder(dtype=tf.float32, shape=[None, 120, 120, 1])
keep_prob = tf.placeholder("float")


#  INPUT 2 (separate branch for input 2)
# --------
#embedding_size = 64
#vocab_size = 6

# embedding
#emb_W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
#emb_x2 = tf.nn.embedding_lookup(emb_W, input_x2)

# fc emb weights
#W_emb_fc1 = weight_variable([embedding_size, 64])
W_emb_fc1 = weight_variable([1, 64])
b_emb_fc1 = bias_variable([64])

W_emb_fc2 = weight_variable([64, 64])
b_emb_fc2 = bias_variable([64])

W_emb_fc3 = weight_variable([64, 64])
b_emb_fc3 = bias_variable([64])

# layers for input 2
#h_emb_x2 = tf.reshape(emb_x2, [-1, embedding_size])
#h_emb_fc1 = tf.nn.relu(tf.add(tf.matmul(h_emb_x2, W_emb_fc1), b_emb_fc1))
h_emb_fc1 = tf.nn.relu(tf.add(tf.matmul(input_x2, W_emb_fc1), b_emb_fc1))

h_emb_fc2 = tf.nn.relu(tf.add(tf.matmul(h_emb_fc1, W_emb_fc2), b_emb_fc2))
h_emb_fc3 = tf.nn.relu(tf.add(tf.matmul(h_emb_fc2, W_emb_fc3), b_emb_fc3))



#  ENCODER
# ---------

# encoder weights
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_pool1 = weight_variable([2, 2, 32, 32])
b_pool1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])  # 32->64
b_conv2 = bias_variable([64])

W_pool2 = weight_variable([2, 2, 64, 64])
b_pool2 = bias_variable([64])

W_conv3 = weight_variable([1, 1, 64, 128])
b_conv3 = bias_variable([128])

W_pool3 = weight_variable([2, 2, 128, 128])
b_pool3 = bias_variable([128])

W_conv4 = weight_variable([1, 1, 128, 32])
b_conv4 = bias_variable([32])

W_fc1 = weight_variable([15*15*32, 4096])
b_fc1 = bias_variable([4096])


# encoder layers
h_conv1 = tf.nn.relu(conv2d(input_X, W_conv1) + b_conv1)
h_pool1 = tf.nn.relu(pool_2x2(h_conv1, W_pool1) + b_pool1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf.nn.relu(pool_2x2(h_conv2, W_pool2) + b_pool2)
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

h_pool3 = tf.nn.relu(pool_2x2(h_conv3, W_pool3) + b_pool3)
#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

# fc part of encoder
h_fc0 = tf.reshape(h_conv4, [-1, 15*15*32])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc0, W_fc1), b_fc1))

# merge
h_fcM = tf.concat([h_emb_fc3, h_fc1], 1)



#  DECODER
# ---------

# fc decoder weights
W_fc2 = weight_variable([4160, 15*15*32])
#W_fc2 = weight_variable([4096, 15*15*32])
b_fc2 = bias_variable([ 15*15*32])

# conv decoder weights
W_conv5 = weight_variable([1, 1, 128, 32])
b_conv5 = bias_variable([128])
deconv_shape_conv5 = tf.stack([batch_size, 15, 15, 128])

W_pool4 = weight_variable([2, 2, 128, 128])
b_pool4 = bias_variable([128])
deconv_shape_pool4 = tf.stack([batch_size, 30, 30, 128])

W_conv6 = weight_variable([1, 1, 64, 128])
b_conv6 = bias_variable([64])
deconv_shape_conv6 = tf.stack([batch_size, 30, 30, 64])

W_pool5 = weight_variable([2, 2, 64, 64])
b_pool5 = bias_variable([64])
deconv_shape_pool5 = tf.stack([batch_size, 60, 60, 64])

W_conv7 = weight_variable([5, 5, 32, 64])
b_conv7 = bias_variable([32])
deconv_shape_conv7 = tf.stack([batch_size, 60, 60, 32])

W_pool6 = weight_variable([2, 2, 32, 32])
b_pool6 = bias_variable([32])
deconv_shape_pool6 = tf.stack([batch_size, 120, 120, 32])

W_conv8 = weight_variable([5, 5, 1, 32])
b_conv8 = bias_variable([1])
deconv_shape_conv8 = tf.stack([batch_size, 120, 120, 1])

# decoder layers

# fc decoder part
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fcM, W_fc2), b_fc2))
#h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
h_fc3 = tf.reshape(h_fc2, [-1, 15, 15, 32])

# conv decoder part
h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(h_fc3, W_conv5, output_shape = deconv_shape_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
#h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv4, W_conv5, output_shape = deconv_shape_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
h_pool4 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W_pool4, output_shape = deconv_shape_pool4, strides=[1,2,2,1], padding='SAME') + b_pool4)
h_conv6 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool4, W_conv6, output_shape = deconv_shape_conv6, strides=[1,1,1,1], padding='SAME') + b_conv6)
h_pool5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W_pool5, output_shape = deconv_shape_pool5, strides=[1,2,2,1], padding='SAME') + b_pool5)
h_conv7 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool5, W_conv7, output_shape = deconv_shape_conv7, strides=[1,1,1,1], padding='SAME') + b_conv7)
h_pool6 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv7, W_pool6, output_shape = deconv_shape_pool6, strides=[1,2,2,1], padding='SAME') + b_pool6)
h_conv8 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool6, W_conv8, output_shape = deconv_shape_conv8, strides=[1,1,1,1], padding='SAME') + b_conv8)

beta = 0.00001
error = tf.nn.l2_loss(input_Y - h_conv8) # \
#+ beta*tf.nn.l2_loss(W_conv1) + beta*tf.nn.l2_loss(b_conv1) \
#+ beta*tf.nn.l2_loss(W_pool1) + beta*tf.nn.l2_loss(b_pool1) \
#+ beta*tf.nn.l2_loss(W_conv2) + beta*tf.nn.l2_loss(b_conv2) \
#+ beta*tf.nn.l2_loss(W_pool2) + beta*tf.nn.l2_loss(b_pool2) \
#+ beta*tf.nn.l2_loss(W_conv3) + beta*tf.nn.l2_loss(b_conv3) \
#+ beta*tf.nn.l2_loss(W_conv4) + beta*tf.nn.l2_loss(b_conv4) \
#+ beta*tf.nn.l2_loss(W_conv5) + beta*tf.nn.l2_loss(b_conv5) \
#+ beta*tf.nn.l2_loss(W_pool3) + beta*tf.nn.l2_loss(b_pool3) \
#+ beta*tf.nn.l2_loss(W_conv6) + beta*tf.nn.l2_loss(b_conv6) \
#+ beta*tf.nn.l2_loss(W_pool4) + beta*tf.nn.l2_loss(b_pool4) \
#+ beta*tf.nn.l2_loss(W_conv7) + beta*tf.nn.l2_loss(b_conv7)


train_step = tf.train.AdamOptimizer(0.0001).minimize(error)
accuracy = tf.nn.l2_loss(input_Y - h_conv8)

font = cv2.FONT_HERSHEY_SIMPLEX

sess.run(tf.initialize_all_variables())
for i in range(1000000):
    X, x2, Y = next(data_iter)
    #X = X[:,:,:,1].reshape([batch_size,200,200,1])

    #print X.shape, Y.shape
    #print X[:, :,:,1].reshape(batch_size, 200, 200, 1).shape
    if i%100 == 0:
        train_accuracy = accuracy.eval( feed_dict={
        input_X:X, input_x2:x2, input_Y:Y, keep_prob: 1.0})
        #input_X:X, input_Y:X, keep_prob: 1.0})
        print ("\nIter", i, "training accuray", train_accuracy)

        #new_img = h_conv8.eval(feed_dict={input_X: X, input_x2: x2, keep_prob: 1.0})

        test_img = h_conv8.eval(feed_dict={input_X: test_X1[0:batch_size], input_x2: test_x2[0:batch_size].reshape([-1,1]), keep_prob: 1.0})
       
        idx_img2 = 9 # cool for walking
        #idx_img2 = 14000 # for handwaving
        #idx_img2 = 3450 # for handclapping
        #idx_img2 = 14000 # for boxing
        #idx_img2 = 1000 # for jogging
        #idx_img2 = 300 # for running
        #idx_img2 = 8020 # handwaving white actor

        test_img2 = h_conv8.eval(feed_dict={input_X: test_X1[idx_img2:idx_img2+batch_size], input_x2: test_x2[idx_img2:idx_img2+batch_size].reshape([-1,1]), keep_prob: 1.0})


        line = np.zeros([1, 120, 1])
        # generate test images

        # 1st test image
        #row1 = np.concatenate((NN_to_CV(test_X1[0]), line, NN_to_CV(test_Y[0]), line, NN_to_CV(test_img[0]), line, NN_to_CV(test_img[0]-test_Y[0])), axis=0)
        #row2 = np.concatenate((NN_to_CV(test_X1[1]), line, NN_to_CV(test_Y[1]), line, NN_to_CV(test_img[1]), line, NN_to_CV(test_img[1]-test_Y[1])), axis=0)
        #row3 = np.concatenate((NN_to_CV(test_X1[2]), line, NN_to_CV(test_Y[2]), line, NN_to_CV(test_img[2]), line, NN_to_CV(test_img[2]-test_Y[2])), axis=0)
        #row4 = np.concatenate((NN_to_CV(test_X1[3]), line, NN_to_CV(test_Y[3]), line, NN_to_CV(test_img[3]), line, NN_to_CV(test_img[3]-test_Y[3])), axis=0)
        #row5 = np.concatenate((NN_to_CV(test_X1[4]), line, NN_to_CV(test_Y[4]), line, NN_to_CV(test_img[4]), line, NN_to_CV(test_img[4]-test_Y[4])), axis=0)

        
        # 1st test image
        row1 = np.concatenate((test_X1[0], line, test_Y[0], line, test_img[0], line, test_img[0]-test_Y[0]), axis=0)
        row2 = np.concatenate((test_X1[1], line, test_Y[1], line, test_img[1], line, test_img[1]-test_Y[1]), axis=0)
        row3 = np.concatenate((test_X1[2], line, test_Y[2], line, test_img[2], line, test_img[2]-test_Y[2]), axis=0)
        row4 = np.concatenate((test_X1[3], line, test_Y[3], line, test_img[3], line, test_img[3]-test_Y[3]), axis=0)
        row5 = np.concatenate((test_X1[4], line, test_Y[4], line, test_img[4], line, test_img[4]-test_Y[4]), axis=0)

        
        all_imgs = np.concatenate((row1, row2, row3, row4, row5), axis=1)
        cv2.imwrite('out_img.png', all_imgs)

        # 2nd test image
        #row1 = np.concatenate((NN_to_CV(test_X1[idx_img2+0]), line, NN_to_CV(test_Y[idx_img2+0]), line, NN_to_CV(test_img2[0]), line, NN_to_CV(test_img2[0]-test_Y[idx_img2+0])), axis=0)
        #row2 = np.concatenate((NN_to_CV(test_X1[idx_img2+1]), line, NN_to_CV(test_Y[idx_img2+1]), line, NN_to_CV(test_img2[1]), line, NN_to_CV(test_img2[1]-test_Y[idx_img2+1])), axis=0)
        #row3 = np.concatenate((NN_to_CV(test_X1[idx_img2+2]), line, NN_to_CV(test_Y[idx_img2+2]), line, NN_to_CV(test_img2[2]), line, NN_to_CV(test_img2[2]-test_Y[idx_img2+2])), axis=0)
        #row4 = np.concatenate((NN_to_CV(test_X1[idx_img2+3]), line, NN_to_CV(test_Y[idx_img2+3]), line, NN_to_CV(test_img2[3]), line, NN_to_CV(test_img2[3]-test_Y[idx_img2+3])), axis=0)
        #row5 = np.concatenate((NN_to_CV(test_X1[idx_img2+4]), line, NN_to_CV(test_Y[idx_img2+4]), line, NN_to_CV(test_img2[4]), line, NN_to_CV(test_img2[4]-test_Y[idx_img2+4])), axis=0)


        row1 = np.concatenate((test_X1[idx_img2+0], line, test_Y[idx_img2+0], line, test_img2[0], line, test_img2[0]-test_Y[idx_img2+0]), axis=0)
        row2 = np.concatenate((test_X1[idx_img2+1], line, test_Y[idx_img2+1], line, test_img2[1], line, test_img2[1]-test_Y[idx_img2+1]), axis=0)
        row3 = np.concatenate((test_X1[idx_img2+2], line, test_Y[idx_img2+2], line, test_img2[2], line, test_img2[2]-test_Y[idx_img2+2]), axis=0)
        row4 = np.concatenate((test_X1[idx_img2+3], line, test_Y[idx_img2+3], line, test_img2[3], line, test_img2[3]-test_Y[idx_img2+3]), axis=0)
        row5 = np.concatenate((test_X1[idx_img2+4], line, test_Y[idx_img2+4], line, test_img2[4], line, test_img2[4]-test_Y[idx_img2+4]), axis=0)



        all_imgs = np.concatenate((row1, row2, row3, row4, row5), axis=1)
        cv2.imwrite('out_img2.png', all_imgs)


    print (".",)
    #print X.shape,
    train_step.run(feed_dict={input_X:X, input_x2:x2, input_Y:Y, keep_prob: 0.8})
    #train_step.run(feed_dict={input_X:X, input_Y:X, keep_prob: 0.8})