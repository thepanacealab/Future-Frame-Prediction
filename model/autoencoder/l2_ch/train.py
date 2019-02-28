from random import shuffle
import cv2
import numpy as np
import h5py
import os
import tensorflow as tf
import time
import csv
from datetime import timedelta
import matplotlib.pyplot as plt

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.InteractiveSession()




def CV_to_NN(x):
    return np.array( x / 255.0, dtype=np.float32) 

def NN_to_CV(x):
    return np.array(255 * x, dtype=np.uint8)




#Read The Dataset.h5
fname_in = "../../../data/dataset.hdf5"


with h5py.File(fname_in,'r') as hf:
    train_X1 = np.array(hf.get('train_X1'), dtype=np.float32)
    train_x2 = np.array(hf.get('train_x2'), dtype=np.float32)
    train_Y  = np.array(hf.get('train_Y'), dtype=np.float32)

    test_X1 = np.array(hf.get('test_X1'), dtype=np.float32)
    test_x2 = np.array(hf.get('test_x2'), dtype=np.float32)
    test_Y  = np.array(hf.get('test_Y'), dtype=np.float32)
    
    validation_X1 = np.array(hf.get('validation_X1'), dtype=np.float32)
    validation_x2 = np.array(hf.get('validation_x2'), dtype=np.float32)
    validation_Y  = np.array(hf.get('validation_Y'), dtype=np.float32)

batch_size = 8 #16


def data_iterator(train_X1, train_x2, train_Y):
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

            
train_data_iter = data_iterator(train_X1, train_x2, train_Y)
test_data_iter = data_iterator(test_X1, test_x2, test_Y)
validation_data_iter = data_iterator(validation_X1, validation_x2, validation_Y)
#data_iter


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


#data_iter = data_iterator()


input_X = tf.placeholder(dtype=tf.float32, shape=[None, 240, 240, 1])
#input_x2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
input_x2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

input_Y = tf.placeholder(dtype=tf.float32, shape=[None, 240, 240, 1])
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
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

W_pool1 = weight_variable([2, 2, 16, 16])
b_pool1 = bias_variable([16])

W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])

W_pool2 = weight_variable([2, 2, 32, 32])
b_pool2 = bias_variable([32])

W_conv3 = weight_variable([5, 5, 32, 64])  # 32->64
b_conv3 = bias_variable([64])

W_pool3 = weight_variable([2, 2, 64, 64])
b_pool3 = bias_variable([64])

W_conv4 = weight_variable([1, 1, 64, 128])
b_conv4 = bias_variable([128])

W_pool4 = weight_variable([2, 2, 128, 128])
b_pool4 = bias_variable([128])

W_conv5 = weight_variable([1, 1, 128, 64])
b_conv5 = bias_variable([64])


W_fc1 = weight_variable([15*15*64, 4096])
#W_fc1 = weight_variable([15*15*32, 4096])
#b_fc1 = bias_variable([4096])
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
h_pool4 = tf.nn.relu(pool_2x2(h_conv4, W_pool4) + b_pool4)

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
# fc part of encoder
h_fc0 = tf.reshape(h_conv5, [-1, 15*15*64])
#h_fc0 = tf.reshape(h_conv5, [-1, 15*15*32])
h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_fc0, W_fc1), b_fc1))
print(h_fc1.shape)
# merge
h_fcM = tf.concat([h_emb_fc3, h_fc1], 1)

print(h_fcM.shape)



#  DECODER
# ---------

# fc decoder weights
W_fc2 = weight_variable([4160, 15*15*64])
#W_fc2 = weight_variable([4160, 15*15*32])
#W_fc2 = weight_variable([4096, 15*15*32])
b_fc2 = bias_variable([ 15*15*64])
#b_fc2 = bias_variable([ 15*15*128])


# conv decoder weights
W_conv6 = weight_variable([1, 1, 128, 64])
b_conv6 = bias_variable([128])
deconv_shape_conv6 = tf.stack([batch_size, 15, 15, 128])

W_pool5 = weight_variable([2, 2, 128, 128])
b_pool5 = bias_variable([128])
deconv_shape_pool5 = tf.stack([batch_size, 30, 30, 128])

W_conv7 = weight_variable([1, 1, 64, 128])
b_conv7 = bias_variable([64])
deconv_shape_conv7 = tf.stack([batch_size, 30, 30, 64])

W_pool6 = weight_variable([2, 2, 64, 64])
b_pool6 = bias_variable([64])
deconv_shape_pool6 = tf.stack([batch_size, 60, 60, 64])

W_conv8 = weight_variable([5, 5, 32, 64])
b_conv8 = bias_variable([32])
deconv_shape_conv8 = tf.stack([batch_size, 60, 60, 32])

W_pool7 = weight_variable([2, 2, 32, 32])
b_pool7 = bias_variable([32])
deconv_shape_pool7 = tf.stack([batch_size, 120, 120, 32])

W_conv9 = weight_variable([5, 5, 16, 32])
b_conv9 = bias_variable([16]) #this part may has some 
deconv_shape_conv9 = tf.stack([batch_size, 120, 120, 16])


W_pool8 = weight_variable([2, 2, 16, 16])
b_pool8 = bias_variable([16])
deconv_shape_pool8 = tf.stack([batch_size, 240, 240, 16])

W_conv10 = weight_variable([5, 5, 1, 16])
b_conv10 = bias_variable([1])
deconv_shape_conv10 = tf.stack([batch_size, 240, 240, 1])

# decoder layers

# fc decoder part
h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fcM, W_fc2), b_fc2))
print(h_fc2.shape)
#h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
#h_fc3 = tf.reshape(h_fc2, [-1, 15, 15, 32])
h_fc3 = tf.reshape(h_fc2, [-1, 15, 15, 64])
print(h_fc3.shape)

# conv decoder part
h_conv6 = tf.nn.relu(tf.nn.conv2d_transpose(h_fc3, W_conv6, output_shape = deconv_shape_conv6, strides=[1,1,1,1], padding='SAME') + b_conv6)
#h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv4, W_conv5, output_shape = deconv_shape_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
h_pool5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W_pool5, output_shape = deconv_shape_pool5, strides=[1,2,2,1], padding='SAME') + b_pool5)
h_conv7 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool5, W_conv7, output_shape = deconv_shape_conv7, strides=[1,1,1,1], padding='SAME') + b_conv7)
h_pool6 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv7, W_pool6, output_shape = deconv_shape_pool6, strides=[1,2,2,1], padding='SAME') + b_pool6)
h_conv8 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool6, W_conv8, output_shape = deconv_shape_conv8, strides=[1,1,1,1], padding='SAME') + b_conv8)
h_pool7 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv8, W_pool7, output_shape = deconv_shape_pool7, strides=[1,2,2,1], padding='SAME') + b_pool7)
h_conv9 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool7, W_conv9, output_shape = deconv_shape_conv9, strides=[1,1,1,1], padding='SAME') + b_conv9)
h_pool8 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv9, W_pool8, output_shape = deconv_shape_pool8, strides=[1,2,2,1], padding='SAME') + b_pool8)
h_conv10 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool8, W_conv10, output_shape = deconv_shape_conv10, strides=[1,1,1,1], padding='SAME') + b_conv10)

beta = 0.00001
error = tf.nn.l2_loss(input_Y - h_conv10) 
#+ beta*tf.nn.l2_loss(W_conv1) + beta*tf.nn.l2_loss(b_conv1) 
#+ beta*tf.nn.l2_loss(W_pool1) + beta*tf.nn.l2_loss(b_pool1) 
#+ beta*tf.nn.l2_loss(W_conv2) + beta*tf.nn.l2_loss(b_conv2) 
#+ beta*tf.nn.l2_loss(W_pool2) + beta*tf.nn.l2_loss(b_pool2) 
#+ beta*tf.nn.l2_loss(W_conv3) + beta*tf.nn.l2_loss(b_conv3)
#+ beta*tf.nn.l2_loss(W_conv4) + beta*tf.nn.l2_loss(b_conv4) 
#+ beta*tf.nn.l2_loss(W_conv5) + beta*tf.nn.l2_loss(b_conv5) 
#+ beta*tf.nn.l2_loss(W_pool3) + beta*tf.nn.l2_loss(b_pool3) 
#+ beta*tf.nn.l2_loss(W_conv6) + beta*tf.nn.l2_loss(b_conv6) 
#+ beta*tf.nn.l2_loss(W_pool4) + beta*tf.nn.l2_loss(b_pool4) 
#+ beta*tf.nn.l2_loss(W_conv7) + beta*tf.nn.l2_loss(b_conv7)


optimizer = tf.train.AdamOptimizer(0.001).minimize(error)

l2_loss = tf.nn.l2_loss(input_Y - h_conv10)

font = cv2.FONT_HERSHEY_SIMPLEX



train_start_time = time.time()


saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'best_validation_loss')

sess = tf.Session()
def init_variables():
    sess.run(tf.global_variables_initializer())

init_variables()

# Best validation accuracy seen so far.
best_validation_loss = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000



# Counter for total number of iterations performed so far.
total_iterations = 0

epoch_list = []
train_loss = []
validation_loss = []
train_time = []
validation_time = []
mini_batch_train_loss = []

def calculate_test_loss():
    total_batch = int(test_X1.shape[0] / batch_size)
    # Start-time used for printing time-usage below.
    start_time = time.time()
    avg_cost = 0
    
    for iteration in range(total_batch):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        X, x2, Y = next(test_data_iter)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_test = {input_X:X, input_x2:x2, input_Y:Y, keep_prob: 1.}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        _, c = sess.run([optimizer, l2_loss], feed_dict=feed_dict_test)
            
        # Compute average loss
        avg_cost += c / total_batch
    
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage in test data: " + str(timedelta(seconds=int(round(time_dif)))))
    
    return avg_cost
            
            
def calculate_validation_loss():
    global validation_time
    total_batch = int(validation_X1.shape[0] / batch_size)
    # Start-time used for printing time-usage below.
    start_time = time.time()
    avg_cost = 0
    
    for iteration in range(total_batch):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        X, x2, Y = next(validation_data_iter)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_validation = {input_X:X, input_x2:x2, input_Y:Y, keep_prob: 1.}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        _, c = sess.run([optimizer, l2_loss], feed_dict=feed_dict_validation)
        
        # Compute average loss
        avg_cost += c / total_batch
        
    # Ending time.
    end_time = time.time()
    
    # Difference between start and end-times.
    time_dif = end_time - start_time
    validation_time.append(time_dif)
    # Print the time-usage.
    print("Time usage in validation data: " + str(timedelta(seconds=int(round(time_dif)))))
    return avg_cost            

# Best validation accuracy seen so far.
best_validation_loss = 1000000.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
require_improvement = 1000

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(epoch):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_loss
    global last_improvement
    global epoch_list
    global train_loss
    global mini_batch_train_loss
    global validation_loss
    global train_time
    
    total_batch = int(train_X1.shape[0] / batch_size)
    display_freq = 100
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(epoch):
        avg_cost = 0
        print('Training epoch: {}'.format(i + 1))
        
        total_iterations += 1
        
        for iteration in range(total_batch):
            # Increase the total number of iterations performed.
            # It is easier to update it in each iteration because
            # we need this number several times in the following.
            

            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            X, x2, Y = next(train_data_iter)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {input_X:X, input_x2:x2, input_Y:Y, keep_prob: 0.8}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            _, c = sess.run([optimizer, l2_loss], feed_dict=feed_dict_train)
            
            #save mini-batch loss c to list
            mini_batch_train_loss.append(c)
            
            # Compute average loss
            avg_cost += c / total_batch
            print(avg_cost)
            
            if iteration % display_freq == 0:
                print("Epoch {:6d}:\t step {0:5d}:\t Mini-Batch Loss={1:.6f}".format(i, iteration, c))
        
        epoch_end_time = time.time() 
        epoch_time_dif = epoch_end_time - start_time
        train_time.append(epoch_time_dif)
        train_loss.append(avg_cost)
        # Print status every 100 iterations and after last iteration.
        #if (total_iterations % 100 == 0) or (i == (epoch - 1)):

        # Calculate the loss on the validation-set
        loss_validation = calculate_validation_loss();
        validation_loss.append(loss_validation)
        
        

        # If validation loss is an improvement over best-known.
        if loss_validation < best_validation_loss:
            
            # Update the best-known validation loss.
            best_validation_loss = loss_validation
                
            # Set the iteration for the last improvement to current.
            last_improvement = total_iterations

            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=sess, save_path=save_path)

            # A string to be printed below, shows improvement found.
            improved_str = '*'
        else:
            # An empty string to be printed below.
            # Shows that no improvement was found.
            improved_str = ''
                
        # Status-message for printing.
        msg = "Iter: {0:>6}, Train Loss: {1:.6f}, Validation Loss: {2:.6f} {3}"
        
        # Print it.
        print(msg.format(i + 1, avg_cost, loss_validation, improved_str))
        
        epoch_list.append(i)   
        # If no improvement found in the required number of iterations.
        #if total_iterations - last_improvement > require_improvement:
        #    print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
        #    break
        

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time
    
    # Print the time-usage.
    print("Time usage in optimize function: " + str(timedelta(seconds=int(round(time_dif)))))



def plot_train_loss(train, validation): 
    
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
	#eval_indices = range(0, training_iters, display_steps)
    plt.plot(train, 'k-', label='Train set Loss')
    plt.plot(validation, 'r--', label='Validation set Loss')
    plt.title('Train and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Generation')           
    plt.legend(loc='upper right')
    fig1 = plt.gcf()
    plt.show()
    
    if not os.path.exists('graph'):
        os.makedirs('graph')
    fig1.savefig('graph/train_validation_loss.png', dpi=2000)

def save_loss_as_csv(epoch_list, train_loss, validation_loss, train_time, validation_time, name):
    data = zip(epoch_list, train_loss, validation_loss, train_time, validation_time)
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['epoch_list', 'train_loss', 'validation_loss', 'train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()
    
def save_mini_batch_train_loss_as_csv(mini_batch_train_loss, name):
    data = [[mini_batch_train_loss]]
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['mini_batch_train_loss'])
        writer.writerows(data)
    
    csvFile.close()
    


def save_test_loss_as_csv(test_loss):
    data = [[test_loss]]
    with open("output/accuracy.txt", "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['test loss'])
        writer.writerows(data)
    csvFile.close() 

epoch = 3
optimize(epoch)

plot_train_loss(train_loss, validation_loss)
save_loss_as_csv(epoch_list, train_loss, validation_loss, train_time, validation_time, 'training_data')
save_mini_batch_train_loss_as_csv(mini_batch_train_loss, 'mini_batch_train_loss')
# Running a new session
print("Starting 2nd session...")

init_variables()
saver.restore(sess=sess, save_path=save_path)
print("Model restored from file: %s" % save_path)


test_loss = calculate_test_loss()
save_test_loss_as_csv(test_loss)
print("Test loss: {:.6f}".format(test_loss))
   

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()


'''
#1000000
for i in range(100000):
    X, x2, Y = next(data_iter)
    #X = X[:,:,:,1].reshape([batch_size,200,200,1])
    
    #print X.shape, Y.shape
    #print X[:, :,:,1].reshape(batch_size, 200, 200, 1).shape
    
    
    
    if i%100 == 0:
        train_accuracy = accuracy.eval( feed_dict={
        input_X:X, input_x2:x2, input_Y:Y, keep_prob: 1.0})
        #input_X:X, input_Y:X, keep_prob: 1.0})
        print ("\nIter", i, "training accuray", train_accuracy)
        
        with open("output/accuracy.txt", "a") as myfile:
            myfile.write('Iter ' + str(i) + ', Accuracy: ' + str(train_accuracy))

        #new_img = h_conv8.eval(feed_dict={input_X: X, input_x2: x2, keep_prob: 1.0})

        test_img = h_conv10.eval(feed_dict={input_X: test_X1[0:batch_size], input_x2: test_x2[0:batch_size].reshape([-1,1]), keep_prob: 1.0})
       
        idx_img2 = 101 # cool for walking
        #idx_img2 = 14000 # for handwaving
        #idx_img2 = 3450 # for handclapping
        #idx_img2 = 14000 # for boxing
        #idx_img2 = 1000 # for jogging
        #idx_img2 = 300 # for running
        #idx_img2 = 8020 # handwaving white actor

        test_img2 = h_conv10.eval(feed_dict={input_X: test_X1[idx_img2:idx_img2+batch_size], input_x2: test_x2[idx_img2:idx_img2+batch_size].reshape([-1,1]), keep_prob: 1.0})


        line = np.zeros([1, 240, 1])
        # generate test images

        # 1st test image
        row1 = np.concatenate((NN_to_CV(test_X1[0]), line, NN_to_CV(test_Y[0]), line, NN_to_CV(test_img[0]), line, NN_to_CV(test_img[0]-test_Y[0])), axis=0)
        row2 = np.concatenate((NN_to_CV(test_X1[1]), line, NN_to_CV(test_Y[1]), line, NN_to_CV(test_img[1]), line, NN_to_CV(test_img[1]-test_Y[1])), axis=0)
        row3 = np.concatenate((NN_to_CV(test_X1[2]), line, NN_to_CV(test_Y[2]), line, NN_to_CV(test_img[2]), line, NN_to_CV(test_img[2]-test_Y[2])), axis=0)
        row4 = np.concatenate((NN_to_CV(test_X1[3]), line, NN_to_CV(test_Y[3]), line, NN_to_CV(test_img[3]), line, NN_to_CV(test_img[3]-test_Y[3])), axis=0)
        row5 = np.concatenate((NN_to_CV(test_X1[4]), line, NN_to_CV(test_Y[4]), line, NN_to_CV(test_img[4]), line, NN_to_CV(test_img[4]-test_Y[4])), axis=0)

        
        

        
        all_imgs = np.concatenate((row1, row2, row3, row4, row5), axis=1)
        #cv2.imwrite('output/out_img.{i:02d}-{idx_img2:02d}.png', all_imgs)
        path = 'output'
        if not os.path.exists(path):
            os.makedirs(path)
        
        path = 'output/output1/'
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path , 'out_img.%05d.png' % (i)), all_imgs)
        

        # 2nd test image
        row1 = np.concatenate((NN_to_CV(test_X1[idx_img2+0]), line, NN_to_CV(test_Y[idx_img2+0]), line, NN_to_CV(test_img2[0]), line, NN_to_CV(test_img2[0]-test_Y[idx_img2+0])), axis=0)
        row2 = np.concatenate((NN_to_CV(test_X1[idx_img2+1]), line, NN_to_CV(test_Y[idx_img2+1]), line, NN_to_CV(test_img2[1]), line, NN_to_CV(test_img2[1]-test_Y[idx_img2+1])), axis=0)
        row3 = np.concatenate((NN_to_CV(test_X1[idx_img2+2]), line, NN_to_CV(test_Y[idx_img2+2]), line, NN_to_CV(test_img2[2]), line, NN_to_CV(test_img2[2]-test_Y[idx_img2+2])), axis=0)
        row4 = np.concatenate((NN_to_CV(test_X1[idx_img2+3]), line, NN_to_CV(test_Y[idx_img2+3]), line, NN_to_CV(test_img2[3]), line, NN_to_CV(test_img2[3]-test_Y[idx_img2+3])), axis=0)
        row5 = np.concatenate((NN_to_CV(test_X1[idx_img2+4]), line, NN_to_CV(test_Y[idx_img2+4]), line, NN_to_CV(test_img2[4]), line, NN_to_CV(test_img2[4]-test_Y[idx_img2+4])), axis=0)

        




        all_imgs = np.concatenate((row1, row2, row3, row4, row5), axis=1)
        path = 'output/output2/'
        
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path , 'out_img2.%05d.png' % (i)), all_imgs)
        #cv2.imwrite('output/out_img.png', all_imgs)
    
    #print (".",)
    #print X.shape,
    start_time = time.time()
    train_step.run(feed_dict={input_X:X, input_x2:x2, input_Y:Y, keep_prob: 0.8})
    #train_step.run(feed_dict={input_X:X, input_Y:X, keep_prob: 0.8})
    
    #count individual epoch time
    elapsed_time = time.time() - start_time
    data = [[i , elapsed_time]]
    with open('epoch_time.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
        
        
        
    #count total epoch  time      
    total_elapsed_time = time.time() - train_start_time
    epoch_data = [[i , total_elapsed_time]]
    with open('total_epoch_time.csv', 'a') as epochFile:
        writer = csv.writer(epochFile)
        writer.writerows(epoch_data)

csvFile.close()
epochFile.close()
f=open("total_train_time.txt", "a+")
train_elapsed_time = time.time() - train_start_time
f.write(time.strftime("%H:%M:%S", time.gmtime(train_elapsed_time)))
f.close()
'''
