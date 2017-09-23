import tensorflow as tf
import data
import pdb
import network
import CTC
import os
import numpy as np
import pickle
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mynetwork=network.Neural_network()
mytrain_data=data .Train_data()
myctc=CTC.ctc()
mytrain_data.Data_process()
word_class_len=len(mytrain_data.label_class_dict)
batch_size=10
X=tf.placeholder(tf.float32,[batch_size,None,20])
X_=tf.expand_dims(X,-1)
Y=tf.placeholder(tf.int32,[batch_size,None])
sequence_length=tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(tf.squeeze(X_,[3])),2)),1),tf.int32)
initial_learning_rate = 1e-2

print("Building neural network..")

output=mynetwork.build_network(X_,word_class_len)

print("Building neural network sucessfully!")

print("Design the CTC lost function and the optimizer..")

def decode(label,output):
    print('-------------------------------------------------')
    for j in range(len(label)):
        s=""
        for i in range(len(output[j])):
            if (output[j][i] != -1):
                 s = s + word_class[output[j][i]]
        print(s)
        s = ""
        for i in range(len(batches_labels[j])):
            if (batches_labels[j][i] != -1):
                s = s + word_class[batches_labels[j][i]]
        print(s)



def Savedata(filemane, data):
    f = open(filemane, 'wb')
    pickle.dump(data, f, 0)
    f.close()

def Loaddata(filename):
    f=open(filename,'rb')
    data=pickle.load(f)
    return data
save_word_class_dict=r'..\data for ASR\save_word_class_dict.txt'
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
word_class = Loaddata(save_word_class_dict)
word_class = {v: k for k, v in word_class.items()}
with tf.Session() as sess:
    batches_wavs, batches_labels = mytrain_data.Get_next_batch(batch_size)
    Savedata("test.txt",batches_wavs)


    print(np.shape(batches_wavs))
    saver.restore(sess, r'save\save.ckpt') # 即将固化到硬盘中的Session从保存路径再读取出来
    decoded, _ = tf.nn.ctc_greedy_decoder(output, sequence_length, merge_repeated=True)
    #predict = tf.sparse_to_dense(decoded[0].indices,decoded[0].shape, decoded[0].values)
    predict = tf.sparse_tensor_to_dense(decoded[0],default_value=-1)
    output,xx= sess.run([predict,sequence_length], feed_dict={X: batches_wavs})

    decode(batches_labels,output)