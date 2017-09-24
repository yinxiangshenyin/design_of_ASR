import error

import sys
import tensorflow as tf
import data
import pdb
import network
import CTC
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mynetwork=network.Neural_network()
mytrain_data=data .Train_data()
myctc=CTC.ctc()
mytrain_data.Data_process()
word_class_len=len(mytrain_data.label_class_dict)
#word_class_len=2229
batch_size=16
X=tf.placeholder(tf.float32,[batch_size,None,20])
X_=tf.expand_dims(X,-1)
Y=tf.placeholder(tf.int32,[batch_size,None])
sequence_length=tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(tf.squeeze(X_,[3])),2)),1),tf.int32)
initial_learning_rate = 1e-2

print("Building neural network..")

output=mynetwork.build_network(X_,word_class_len)

print("Building neural network sucessfully!")

print("Design the CTC lost function and the optimizer..")

loss=myctc.ctc_loss(output,Y,sequence_length)
cost = tf.reduce_mean(loss)
acc=myctc.ctc_accuracy(output,Y,sequence_length)
optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)

print("Design the CTC lost function and the optimizer sucessfully!")
n_batch = mytrain_data.wav_max_len // batch_size
saver=tf.train.Saver()
with tf.Session() as sess:
    print("Start Training")
    sess.run(tf.global_variables_initializer())
    for epoch in range(16):
        for batch in range(n_batch):
            now_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(now_time)
            print('Batch:'+str(batch))
            batches_wavs, batches_labels = mytrain_data.Get_next_batch(batch_size)
            train_loss,train_acc, _ = sess.run([cost,acc, optimizer,], feed_dict={X: batches_wavs, Y: batches_labels})
            print('Cost:'+str(train_loss))
            print("Accuracy:"+str(train_acc))

    save_path=saver.save(sess,r'..\data for ASR\save\save.ckpt')











