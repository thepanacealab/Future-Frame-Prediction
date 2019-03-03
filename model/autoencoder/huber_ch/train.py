from random import shuffle
import cv2
import numpy as np
import h5py
import os
import tensorflow as tf
import time
import csv
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
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

def huber_loss(x, y):
    return tf.losses.huber_loss(x, y)



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
#print(h_fc1.shape)
# merge
h_fcM = tf.concat([h_emb_fc3, h_fc1], 1)

#print(h_fcM.shape)



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
#print(h_fc2.shape)
#h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))
#h_fc3 = tf.reshape(h_fc2, [-1, 15, 15, 32])
h_fc3 = tf.reshape(h_fc2, [-1, 15, 15, 64])
#print(h_fc3.shape)

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
error = huber_loss(input_Y, h_conv10)


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

#l2_loss = tf.nn.l2_loss(input_Y - h_conv10)
huber_loss = huber_loss(input_Y, h_conv10)

mse = tf.losses.mean_squared_error(input_Y, h_conv10)

font = cv2.FONT_HERSHEY_SIMPLEX



train_start_time = time.time()


saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'best_validation_loss')


config = tf.ConfigProto()
config.intra_op_parallelism_threads = 20
config.inter_op_parallelism_threads = 20
#tf.session(config=config)
sess = tf.Session(config=config)

#sess = tf.Session()
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
epcoh_step_train_list = []
epcoh_step_validation_list = []
train_loss = []
train_loss_mse = []
validation_loss = []
validation_loss_mse = []
train_time = []
validation_time = []
mini_batch_train_loss = []
mini_batch_train_loss_mse = []
mini_batch_test_loss = []
mini_batch_test_loss_mse = []
mini_batch_validation_loss = []
mini_batch_validation_loss_mse = []


train_data_iter = data_iterator(train_X1, train_x2, train_Y)
validation_data_iter = data_iterator(validation_X1, validation_x2, validation_Y)
test_data_iter = data_iterator(test_X1, test_x2, test_Y)

def calculate_test_loss():
    total_batch = int(test_X1.shape[0] / batch_size)
    # Start-time used for printing time-usage below.
    start_time = time.time()
    avg_cost = 0
    avg_cost_mse = 0
    
    
    
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
        c, c_mse = sess.run([huber_loss, mse], feed_dict=feed_dict_test)
            
        
        mini_batch_test_loss.append(c)
        mini_batch_test_loss_mse.append(c_mse)
            
        # Compute average loss
        avg_cost += c / total_batch
        avg_cost_mse += c_mse / total_batch
    
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage in test data: " + str(timedelta(seconds=int(round(time_dif)))))
    
    return avg_cost, avg_cost_mse
            
            
def calculate_validation_loss():
    global validation_time
    total_batch = int(validation_X1.shape[0] / batch_size)
    # Start-time used for printing time-usage below.
    start_time = time.time()
    avg_cost = 0
    avg_cost_mse = 0
    
    
    
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
        c, c_mse = sess.run([huber_loss, mse], feed_dict=feed_dict_validation)
        
        mini_batch_validation_loss.append(c)
        mini_batch_validation_loss_mse.append(c_mse)
        
        # Compute average loss
        avg_cost += c / total_batch
        avg_cost_mse += c_mse / total_batch
        
    # Ending time.
    end_time = time.time()
    
    # Difference between start and end-times.
    time_dif = end_time - start_time
    validation_time.append(time_dif)
    # Print the time-usage.
    print("Time usage in validation data: " + str(timedelta(seconds=int(round(time_dif)))))
    return avg_cost, avg_cost_mse

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
    global train_loss_mse
    global mini_batch_train_loss
    global mini_batch_train_loss_mse
    global validation_loss
    global validation_loss_mse
    global train_time
    
    total_batch = int(train_X1.shape[0] / batch_size)
    #display_freq = 100
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(epoch):
        mini_batch_start_time = time.time()
        
        avg_cost = 0
        avg_cost_mse = 0
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
            _, c, c_mse = sess.run([optimizer, huber_loss, mse], feed_dict=feed_dict_train)
            
            #save mini-batch loss c to list
            mini_batch_train_loss.append(c)
            mini_batch_train_loss_mse.append(c_mse)
            
            # Compute average loss
            avg_cost += c / total_batch
            avg_cost_mse += c_mse / total_batch
            #print(avg_cost)
            
            #if iteration % display_freq == 0:
            #    print("Epoch {0:4d}\t step {1:5d}:\t Mini-Batch Loss={2:.6f} \t Mini-Batch MSE Loss={3:.6f}".format(i, iteration, c, c_mse))
                
            
        
        epoch_end_time = time.time() 
        epoch_time_dif = epoch_end_time - mini_batch_start_time
        train_time.append(epoch_time_dif)
        print("Time usage in train data: " + str(timedelta(seconds=int(round(epoch_time_dif)))))
        
        train_loss.append(avg_cost)
        train_loss_mse.append(avg_cost_mse)
        # Print status every 100 iterations and after last iteration.
        #if (total_iterations % 100 == 0) or (i == (epoch - 1)):

        # Calculate the loss on the validation-set
        loss_validation, loss_mse_validation = calculate_validation_loss();
        validation_loss.append(loss_validation)
        validation_loss_mse.append(loss_mse_validation)
        
        

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
        msg = "Iter: {0:>6}, Train Loss: {1:.6f}, Train Loss MSE: {2:.6f}, Validation Loss: {3:.6f}, Validation Loss MSE: {4:.6f} {5}"
        
        # Print it.
        print(msg.format(i + 1, avg_cost, avg_cost_mse, loss_validation, loss_mse_validation, improved_str))
        
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



def plot_train_loss(train, validation, name): 
    
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
    #plt.show()
    
    if not os.path.exists('graph'):
        os.makedirs('graph')
    filename = name+'.png'
    fig1.savefig('graph/' + filename, dpi=2000)

def save_loss_as_csv(epoch_list, train_loss, train_loss_mse, validation_loss, validation_loss_mse, train_time, validation_time, name):
    data = zip(epoch_list, train_loss, train_loss_mse, validation_loss, validation_loss_mse, train_time, validation_time)
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['epoch_list', 'train_loss', 'train_loss_mse', 'validation_loss', 'validation_loss_mse','train_time', 'validation_time'])
        writer.writerows(data)
    
    csvFile.close()
    
def save_mini_batch_loss_as_csv(mini_batch_train_loss, mini_batch_train_loss_mse, name):
    data = zip(mini_batch_train_loss, mini_batch_train_loss_mse)
    file_name = str(name)+'.csv'
    save_dir = 'csv'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    header_1 = str(name)
    header_2 = str(name) + '_mse'
    
    with open(save_path, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([header_1, header_2])
        writer.writerows(data)
    
    csvFile.close()
    


def save_test_loss_as_csv(test_loss, test_loss_mse):
    data = [[test_loss, test_loss_mse]]
    with open("csv/test_loss.csv", "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['test_loss', 'test_loss_mse'])
        writer.writerows(data)
    csvFile.close() 

epoch = 2
optimize(epoch)


#plot_train_loss(mini_batch_train_loss, mini_batch_validation_loss, 'minibatch_l2')
#plot_train_loss(mini_batch_train_loss_mse, mini_batch_validation_loss_mse, 'minibatch_mse')

save_loss_as_csv(epoch_list, train_loss, train_loss_mse, validation_loss, validation_loss_mse, train_time, validation_time, 'training_data')
save_mini_batch_loss_as_csv(mini_batch_train_loss, mini_batch_train_loss_mse, 'mini_batch_train_loss')
save_mini_batch_loss_as_csv(mini_batch_validation_loss, mini_batch_validation_loss_mse, 'mini_batch_validation_loss')
save_mini_batch_loss_as_csv(mini_batch_test_loss, mini_batch_test_loss_mse, 'mini_batch_test_loss')

# Running a new session
print("Starting 2nd session...")

init_variables()
saver.restore(sess=sess, save_path=save_path)
print("Model restored from file: %s" % save_path)


test_loss, test_loss_mse = calculate_test_loss()
save_test_loss_as_csv(test_loss, test_loss_mse)
print("Test loss: {:.6f}\t Test loss MSE: {:.6f}".format(test_loss, test_loss_mse))
   
plot_train_loss(train_loss, validation_loss, 'huber')
plot_train_loss(train_loss_mse, validation_loss_mse, 'mse')
# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()



