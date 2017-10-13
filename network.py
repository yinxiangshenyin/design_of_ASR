import tensorflow as tf
layer_num=0
class Neural_network:
    def convolution_2D_layer(self,input,filter_size,output_channel,stride_len,activation_function=True,bisa=False):
        # for the first layer the input shape is [batch_size,time_step,mfcc,channel=1]
        # for the first layer filter_shape=[filter_size_x,filter_size_y,input_channel,output_channel]
        global layer_num
        with tf.variable_scope("Layer_"+str(layer_num)):
            input_channel=(input.get_shape()).as_list()[-1]
            filter=[filter_size,filter_size,input_channel,output_channel]
            stride=[1,1,stride_len,1]
            W=tf.get_variable(shape=filter,initializer=tf.truncated_normal_initializer(stddev=0.1),name='W')
            cnn_output = tf.nn.conv2d(input=input, filter=W, strides=stride, padding="SAME")
            if(~bisa):
                mean,var=tf.nn.moments(cnn_output,axes=[0,1,2])
                gamma = tf.get_variable(shape=[output_channel],initializer=tf.truncated_normal_initializer( stddev=0.1), name='gamma')
                beta=tf.get_variable(shape=[output_channel],initializer=tf.constant_initializer(0),name='beta')

                output = tf.nn.batch_norm_with_global_normalization(
                    cnn_output, mean, var, beta, gamma, 0.001,
                    scale_after_normalization=True)
            else:
                b=tf.get_variable(tf.float32,[output_channel],tf.constant_initializer(0),name='bisa')
                output=cnn_output+b

            if(activation_function):
                out = tf.nn.relu(output)
            else:
                out=output

            layer_num+=1
        return out

    def residual_neural_network(self,input,filter_size,output_channel,stride=1,max_pooling=False,projection=True):
        input_channel=input.get_shape().as_list()[-1]
        if max_pooling:
            filter=[1,2,2,1]
            input = tf.nn.max_pool(input, ksize=filter, strides=filter, padding='SAME')
        conv1 = self.convolution_2D_layer(input=input, filter_size=filter_size,output_channel=input_channel, stride_len=stride)
        conv2 = self.convolution_2D_layer(input=conv1, filter_size=filter_size,output_channel=output_channel,stride_len=1)

        if stride != 1 or output_channel!=input_channel:
            if projection:
                # Option B: Projection shortcut
                input_layer = self.convolution_2D_layer(input,  1, output_channel, stride)
            else:
                # Option A: Zero-padding
                input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, input_channel - output_channel]])
        else:
            input_layer = input

        res = conv2 + input_layer
        return res

    def build_network(self,input,word_class_len):

        first_layer_output = self.convolution_2D_layer(input=input, filter_size=5, output_channel=32, stride_len=1)

        firt_residual_layer_output = self.residual_neural_network(input=first_layer_output, filter_size=3,
                                                                       output_channel=64, stride=1, max_pooling=False)

        residual1_layer_output = self.residual_neural_network(input=firt_residual_layer_output, filter_size=3,
                                                                   output_channel=64, stride=1, max_pooling=False)

        residual1_layer_output = self.residual_neural_network(input=residual1_layer_output, filter_size=3,
                                                                   output_channel=64, stride=1, max_pooling=False)

        residual1_layer_output = self.residual_neural_network(input=residual1_layer_output, filter_size=3,
                                                                   output_channel=64, stride=1, max_pooling=False)

        residual2_layer_output = self.residual_neural_network(input=residual1_layer_output, filter_size=3,
                                                                   output_channel=128, stride=2, max_pooling=False)

        residual2_layer_output = self.residual_neural_network(input=residual2_layer_output, filter_size=3,
                                                                   output_channel=128, stride=1, max_pooling=False)

        residual2_layer_output = self.residual_neural_network(input=residual2_layer_output, filter_size=3,
                                                                   output_channel=128, stride=1, max_pooling=False)

        residual2_layer_output = self.residual_neural_network(input=residual2_layer_output, filter_size=3,
                                                                   output_channel=128, stride=1, max_pooling=False)

        residual3_layer_output = self.residual_neural_network(input=residual2_layer_output, filter_size=3,
                                                                   output_channel=256, stride=2, max_pooling=False)

        residual3_layer_output = self.residual_neural_network(input=residual3_layer_output, filter_size=3,
                                                                   output_channel=256, stride=1, max_pooling=False)

        residual4_layer_output = self.residual_neural_network(input=residual3_layer_output, filter_size=3,
                                                                   output_channel=256, stride=2, max_pooling=False)
        residual4_layer_output = self.residual_neural_network(input=residual4_layer_output, filter_size=3,
                                                                   output_channel=256, stride=1, max_pooling=False)
        residual4_layer_output = self.residual_neural_network(input=residual4_layer_output, filter_size=3,
                                                                   output_channel=256, stride=3, max_pooling=False)

        # residual4_layer_output = self.residual_neural_network(input=residual3_layer_output, filter_size=3,
        #                                                            output_channel=512, stride=2, max_pooling=False)
        #
        # residual4_layer_output = self.residual_neural_network(input=residual4_layer_output, filter_size=3,
        #                                                            output_channel=512, stride=1, max_pooling=False)
        #
        # residual5_layer_output = self.residual_neural_network(input=residual4_layer_output, filter_size=3,
        #                                                            output_channel=1024, stride=3, max_pooling=False)

        last_layer_output = self.convolution_2D_layer(input=residual4_layer_output, filter_size=3,
                                                           output_channel=word_class_len + 1, stride_len=1,
                                                           activation_function=False, bisa=True)

        output = tf.transpose(tf.squeeze(last_layer_output, [2]),(1,0,2))

        return output

