# Antonio Busson - https://github.com/Busson/Inception-ResNet-v2
# NVIDIA DIGITS template

# [IMPORTANT!] Set initial Learning Rate with = 0.001

from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits

class UserModel(Tower):

    #residual learning 
    residual_scale = 0.2

    def inception_resnet_a(self, input, name, scale=1.0):
        
        #conv2d_1ira_1x1
        conv2d_1ira_1x1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_1ira'+name+'_1x1')

        #conv2d_2ira_1x1
        conv2d_2ira_1x1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_2ira'+name+'_1x1')

        #conv2d_3ira_1x1
        conv2d_3ira_1x1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_3ira'+name+'_1x1')

        #conv2d_4ira_3x3
        conv2d_4ra_3x3 = tf.layers.conv2d(inputs=conv2d_2ira_1x1, filters=32, kernel_size=[
            3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_4ira'+name+'_3x3')

        #conv2d_5ira_3x3
        conv2d_5ra_3x3 = tf.layers.conv2d(inputs=conv2d_3ira_1x1, filters=48, kernel_size=[
            3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_5ira'+name+'_3x3')

        #conv2d_6ira_3x3
        conv2d_6ra_3x3 = tf.layers.conv2d(inputs=conv2d_5ra_3x3, filters=64, kernel_size=[
            3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_6ira'+name+'_3x3')
    
        filter_concat_1ra = tf.concat([conv2d_1ira_1x1,conv2d_4ra_3x3,conv2d_6ra_3x3], 3)
        
        #conv2d_7ra_3x3
        conv2d_7ra_1x1 = tf.layers.conv2d(inputs=filter_concat_1ra, filters=384, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=None, name='Conv2d_7ira'+name+'_1x1')
    
        #res
        res = input + conv2d_7ra_1x1 * scale

        #relu1 activation
        relu1 = tf.nn.relu(res, name='Relu1_incept_res_a'+name)

        return relu1

    def inception_resnet_b(self, input, name, scale=1.0):
        
        #conv2d_1irb_1x1
        conv2d_1irb_1x1 = tf.layers.conv2d(inputs=input, filters=192, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_1irb'+name+'_1x1')

        #conv2d_2irb_1x1
        conv2d_2irb_1x1 = tf.layers.conv2d(inputs=input, filters=128, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_2irb'+name+'_1x1')

        #conv2d_3irb_1x7
        conv2d_3irb_1x7 = tf.layers.conv2d(inputs=conv2d_2irb_1x1, filters=160, kernel_size=[
            1, 7], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_3irb'+name+'_1x7')

        #conv2d_4irb_7x1
        conv2d_4irb_7x1 = tf.layers.conv2d(inputs=conv2d_3irb_1x7, filters=192, kernel_size=[
            7, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_4irb'+name+'_7x1')

        concat = tf.concat([conv2d_1irb_1x1, conv2d_4irb_7x1], 3)

        #conv2d_7irb_1x1
        conv2d_7irb_1x1 = tf.layers.conv2d(inputs=concat, filters=1152, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=None, name='conv2d_7irb'+name+'_1x1')

        #res
        res = input + conv2d_7irb_1x1 * scale

        #relu1 activation
        relu1 = tf.nn.relu(res, name='Relu1_incept_res_b'+name)

        return relu1

    def inception_resnet_c(self, input, name, scale=1.0):
        
        #conv2d_1irc_1x1
        conv2d_1irc_1x1 = tf.layers.conv2d(inputs=input, filters=192, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_1irc'+name+'_1x1')

        #conv2d_2irc_1x1
        conv2d_2irc_1x1 = tf.layers.conv2d(inputs=input, filters=192, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_2irc'+name+'_1x1')

        #conv2d_3irc_1x3
        conv2d_3irc_1x3 = tf.layers.conv2d(inputs=conv2d_2irc_1x1, filters=224, kernel_size=[
            1, 3], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_3irc'+name+'_1x3')

        #conv2d_4irc_3x1
        conv2d_4irc_3x1 = tf.layers.conv2d(inputs=conv2d_3irc_1x3, filters=256, kernel_size=[
            3, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d_4irc'+name+'_3x1')

        concat = tf.concat([conv2d_1irc_1x1, conv2d_4irc_3x1], 3)

        #conv2d_7irc_1x1
        conv2d_7irc_1x1 = tf.layers.conv2d(inputs=concat, filters=2144, kernel_size=[
            1, 1], strides=1, padding='SAME', activation=None, name='conv2d_7irc'+name+'_1x1')

        #res
        res = input + conv2d_7irc_1x1 * scale

        #relu1 activation
        relu1 = tf.nn.relu(res, name='Relu1_incept_res_c'+name)

        return relu1        

    def reduction_a(self, input, k, l, m, n):
        #max_pool_ra
        max_pool_ra = tf.layers.max_pooling2d(inputs=input, pool_size=[
            3, 3], strides=2)

        #conv2d_1ra_3x3
        conv2d_1ra_3x3 = tf.layers.conv2d(inputs=input, filters=n, kernel_size=[
            3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_1reda_3x3')

        #conv2d_2ra_1x1
        conv2d_2ra_1x1 = tf.layers.conv2d(inputs=input, filters=k, kernel_size=[
            1, 1], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_2reda_1x1')

        #conv2d_3ra_3x3
        conv2d_3ra_3x3 = tf.layers.conv2d(inputs=conv2d_2ra_1x1, filters=l, kernel_size=[
            3, 3], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_3reda_3x3')

        #conv2d_4ra_3x3
        conv2d_4ra_3x3 = tf.layers.conv2d(inputs=conv2d_3ra_3x3, filters=m, kernel_size=[
            3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_4reda_3x3')

        filter_concat = tf.concat([max_pool_ra, conv2d_1ra_3x3, conv2d_4ra_3x3], 3)

        return filter_concat

    def reduction_b(self, input):

        #max_pool_rb_3x3
        max_pool_rb_3x3 = tf.layers.max_pooling2d(inputs=input, pool_size=[
            3, 3], strides=2)

        #conv2d_1rb_1x1
        conv2d_1rb_1x1 = tf.layers.conv2d(inputs=input, filters=256, kernel_size=[
            1, 1], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_1rb_1x1')

        #conv2d_2rb_1x1
        conv2d_2rb_1x1 = tf.layers.conv2d(inputs=input, filters=256, kernel_size=[
            1, 1], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_2rb_1x1')
    
        #conv2d_3rb_1x1
        conv2d_3rb_1x1 = tf.layers.conv2d(inputs=input, filters=256, kernel_size=[
            1, 1], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_3rb_1x1')

        #conv2d_4rb_3x3
        conv2d_4rb_3x3 = tf.layers.conv2d(inputs=conv2d_1rb_1x1, filters=384, kernel_size=[
            3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_4rb_3x3')

        #conv2d_5rb_3x3
        conv2d_5rb_3x3 = tf.layers.conv2d(inputs=conv2d_2rb_1x1, filters=288, kernel_size=[
            3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_5rb_3x3')

        #conv2d_6rb_3x3
        conv2d_6rb_3x3 = tf.layers.conv2d(inputs=conv2d_3rb_1x1, filters=288, kernel_size=[
            3, 3], strides=1, activation=tf.nn.relu, padding='SAME', name='Conv2d_6rb_3x3')    

        #conv2d_7rb_3x3
        conv2d_7rb_3x3 = tf.layers.conv2d(inputs=conv2d_6rb_3x3, filters=320, kernel_size=[
            3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_7rb_3x3')    

        filter_concat = tf.concat([max_pool_rb_3x3, conv2d_4rb_3x3, conv2d_5rb_3x3, conv2d_7rb_3x3], 3)

        return filter_concat 

    @model_property
    def inference(self):
        #input 299 x 299 x 3 
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        #conv2d_1a_3x3 (299x299x3) -> (149x149x32)
        conv2d_1a_3x3 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_1a_3x3')
        #tf.add_to_collection('inception_convs', conv2d_1a_3x3) 

        #conv2d_2a_3x3 (149x149x32) -> (147x147x32)
        conv2d_2a_3x3 = tf.layers.conv2d(inputs=conv2d_1a_3x3, filters=32, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, name='Conv2d_2a_3x3')
        #tf.add_to_collection('inception_convs', conv2d_2a_3x3)

        #conv2d_3a_3x3 (147x147x32) -> (147x147x64)
        conv2d_3a_3x3 = tf.layers.conv2d(inputs=conv2d_2a_3x3, filters=64, kernel_size=[3, 3], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_3a_3x3')
        #tf.add_to_collection('inception_convs', conv2d_3a_3x3) 

        #max_pool1 (147x147x64) -> (73x73x64)
        max_pool1 = tf.layers.max_pooling2d(inputs=conv2d_3a_3x3, pool_size=[3, 3], strides=2)

        #conv2d_4a_3x3 (147x147x64) -> (73x73x96)
        conv2d_4a_3x3 = tf.layers.conv2d(inputs=conv2d_3a_3x3, filters=96, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_4a_3x3')
            
        #filter_concat_1a  (73x73x64) + (73x73x96) -> (73x73x160)
        filter_concat_1a = tf.concat([max_pool1, conv2d_4a_3x3], 3) 

        #conv2d_5a_1x1 (73x73x160) -> ?
        conv2d_5a_1x1 = tf.layers.conv2d(inputs=filter_concat_1a, filters=64, kernel_size=[1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_5a_1x1')

        #conv2d_6a_3x3 ? -> (71x71x96)
        conv2d_6a_3x3 = tf.layers.conv2d(inputs=conv2d_5a_1x1, filters=96, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, name='Conv2d_6a_3x3')

        #conv2d_7a_1x1 ? -> ?
        conv2d_7a_1x1 = tf.layers.conv2d(inputs=filter_concat_1a, filters=64, kernel_size=[1, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_7a_1x1')

        #conv2d_8a_7x1 ? -> ?
        conv2d_8a_7x1 = tf.layers.conv2d(inputs=conv2d_7a_1x1, filters=64, kernel_size=[7, 1], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_8a_7x1')

        #conv2d_9a_1x7 ? -> ?
        conv2d_9a_1x7 = tf.layers.conv2d(inputs=conv2d_8a_7x1, filters=64, kernel_size=[1, 7], strides=1, padding='SAME', activation=tf.nn.relu, name='Conv2d_9a_1x7')

        #conv2d_10a_3x3 ? -> (71x71x96)
        conv2d_10a_3x3 = tf.layers.conv2d(inputs=conv2d_9a_1x7, filters=96, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, name='Conv2d_10a_3x3')

        #filter_concat_2a (71x71x96) + (71x71x96) -> (71x71x192)
        filter_concat_2a = tf.concat([conv2d_6a_3x3, conv2d_10a_3x3], 3)
            
        #conv2d_11a_3x3 (71x71x192) -> (35x35x192)
        conv2d_11a_3x3 = tf.layers.conv2d(inputs=filter_concat_2a, filters=192, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, name='Conv2d_11a_3x3')

        #max_pool3 (71x71x192) -> (35x35x192)
        max_pool3 = tf.layers.max_pooling2d(inputs=filter_concat_2a, pool_size=[3, 3], strides=2)

        #filter_concat_3a (35x35x192) + (35x35x192) -> (35x35x384)
        filter_concat_3a = tf.concat([conv2d_11a_3x3, max_pool3], 3)

        #inception-resnet-a-1 (35x35x384) -> (35x35x384)
        inception_resnet_a_1 = self.inception_resnet_a(filter_concat_3a, '1', self.residual_scale)

        #inception-resnet-a-2 (35x35x384) -> (35x35x384)
        inception_resnet_a_2 = self.inception_resnet_a(inception_resnet_a_1, '2', self.residual_scale)

        #inception-resnet-a-3 (35x35x384) -> (35x35x384)
        inception_resnet_a_3 = self.inception_resnet_a(inception_resnet_a_2, '3', self.residual_scale)

        #inception-resnet-a-4 (35x35x384) -> (35x35x384)
        inception_resnet_a_4 = self.inception_resnet_a(inception_resnet_a_3, '4', self.residual_scale)

        #inception-resnet-a-5 (35x35x384) -> (35x35x384)
        inception_resnet_a_5 = self.inception_resnet_a(inception_resnet_a_4, '5', self.residual_scale)

        #reduction-a (35x35x384) -> (17x17x1152)
        reduction_a = self.reduction_a(inception_resnet_a_5, 256, 256, 384, 384)

        #inception-resnet-b-1 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_1 = self.inception_resnet_b(reduction_a, '1', self.residual_scale)

        #inception-resnet-b-2 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_2 = self.inception_resnet_b(inception_resnet_b_1, '2', self.residual_scale)

        #inception-resnet-b-3 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_3 = self.inception_resnet_b(inception_resnet_b_2, '3', self.residual_scale)

        #inception-resnet-b-4 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_4 = self.inception_resnet_b(inception_resnet_b_3, '4', self.residual_scale)

        #inception-resnet-b-5 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_5 = self.inception_resnet_b(inception_resnet_b_4, '5', self.residual_scale)

        #inception-resnet-b-6 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_6 = self.inception_resnet_b(inception_resnet_b_5, '6', self.residual_scale)

        #inception-resnet-b-7 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_7 = self.inception_resnet_b(inception_resnet_b_6, '7', self.residual_scale)

        #inception-resnet-b-8 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_8 = self.inception_resnet_b(inception_resnet_b_7, '8', self.residual_scale)

        #inception-resnet-b-9 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_9 = self.inception_resnet_b(inception_resnet_b_8, '9', self.residual_scale)

        #inception-resnet-b-10 (17x17x1152) -> (17x17x1152)
        inception_resnet_b_10 = self.inception_resnet_b(inception_resnet_b_9, '10', self.residual_scale)

        #reduction-b (17x17x1152) -> (8x8x2144)
        reduction_b = self.reduction_b(inception_resnet_b_10)

        #inception-resnet-c-1 (8x8x2144) -> (8x8x2144)
        inception_resnet_c_1 = self.inception_resnet_c(reduction_b, '1', self.residual_scale)

        #inception-resnet-c-2 (8x8x2144) -> (8x8x2144)
        inception_resnet_c_2 = self.inception_resnet_c(inception_resnet_c_1, '2', self.residual_scale)

        #inception-resnet-c-3 (8x8x2144) -> (8x8x2144)
        inception_resnet_c_3 = self.inception_resnet_c(inception_resnet_c_2, '3', self.residual_scale)
            
        #inception-resnet-c-4 (8x8x2144) -> (8x8x2144)
        inception_resnet_c_4 = self.inception_resnet_c(inception_resnet_c_3, '4', self.residual_scale)
            
        #inception-resnet-c-5 (8x8x2144) -> (8x8x2144)
        inception_resnet_c_5 = self.inception_resnet_c(inception_resnet_c_4, '5', self.residual_scale)
            
        #avg-pool (8x8x2144) -> (1x1x2144)
        avg_pooling = tf.layers.average_pooling2d(inputs=inception_resnet_c_5, pool_size=[8, 8], strides=1, name="Avg_pooling")

        #avg-pool-flat (1x1x2144) -> (2144)
        avg_flat_flat = tf.reshape(avg_pooling, [-1, 2144])

        #dropout keep 80%
        dropout = tf.layers.dropout(inputs=avg_flat_flat, rate=0.2, training=self.is_training)

        #fully_connected
        fully = tf.contrib.layers.fully_connected(dropout, num_outputs=self.nclasses, activation_fn=None)    
        
        return fully

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
