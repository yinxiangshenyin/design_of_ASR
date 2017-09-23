import tensorflow as tf
import numpy as np
class ctc:
    def SimpleSparseTensorFrom(self,x):
        indices=tf.where(tf.not_equal(x,-1))
        value=tf.gather_nd(x,indices)
        shape = tf.cast(tf.shape(x), tf.int64)
        return tf.SparseTensor(indices, value, shape)

    def ctc_loss(self,input,label,sequence_length):
        #the input shape is [batch,time_stamp,depth] the depth is the num_class+1 ,the num_class+1 is reserver for bank_label
        #the lable is encoded by num_class lik[0,45,9...] the '0' is real_class not bank_label ,the label is not contain the class for bank_label,this work is down by tf.nn.ctc_loss
        #the sequrnce_length mean length eatch squence of  batch_input for the network_input [376,560,....]
        #the time_stamp is the define by the out_put from the network,and the time_stamp is large than the each squence_length
        target=self.SimpleSparseTensorFrom(label)
        loss = tf.nn.ctc_loss(time_major=True, labels=target, inputs=input, sequence_length=sequence_length)
        return loss

    def ctc_accuracy(self,input,label,sequence_length):

        decoded, log_prob = tf.nn.ctc_greedy_decoder(input, sequence_length)
        target = self.SimpleSparseTensorFrom(label)
        # Inaccuracy: label error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              target))
        return acc