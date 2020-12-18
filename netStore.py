#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
import TensorflowUtils as utils
import numpy as np


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def net_encoder1(image):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('fcrn_encoder_')]) > 0
    with tf.variable_scope('fcrn_encoder_', reuse=reuse):
        print(image.shape)
        
        x = slim.conv2d(image, 6, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 6, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        #  卷积代替池化
        x = slim.conv2d(x, 6, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        print('en_1',x.shape)
        
        
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        D1 = x
        print('D1',D1.shape)
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        print('en_2',x.shape)
        
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        D2 = x
        print('D2',D2.shape)
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        D3 = x
        print('D3',D3.shape)
        x = slim.conv2d(x, 96, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        print(x.shape)
    #  返回最后三层真特征层以及三个之路D1-D3
    return x,D1,D2,D3

# 噪声发生器网络
def net_G_encoder2(image):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('G_encoder_')]) > 0
    with tf.variable_scope('G_encoder_', reuse=reuse):
        print(image.shape)
        
        x = slim.conv2d(image, 6, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 6, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        #  卷积代替池化
        x = slim.conv2d(x, 6, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        print('g_en_1',x.shape)
        
        
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)

        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        print('g_en_2',x.shape)
        
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
 
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)

        x = slim.conv2d(x, 96, kernel_size=[3,3], stride=2, activation_fn = lrelu)
        
        z1 = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        z2 = slim.conv2d(z1, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        z3 = slim.conv2d(z2, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        #  返回最后三层添加噪声后的特征层
    return z1,z2,z3

#     解码器，
def net_decoder(x,D1,D2,D3,a,noise_feature=None):
    '''
    x1,x2,x3: encoder1 输出的三层真特征层
    D1,D2,D3:编码器的支路
    a：超参数
    z1,z2,z3:encoder2  输出的三层假特征层
    '''
    reuse = len([t for t in tf.global_variables() if t.name.startswith('fcrn_decoder_')]) > 0
    with tf.variable_scope('fcrn_decoder_', reuse=reuse):
        
#  ----------------------判断是否融合噪声 ，z输入为空则不融合，
        if noise_feature != None:
            #融合噪声   fake_feature  = (1.0-a)*real_feature + a*noise_feature
            
            z1,z2,z3 = noise_feature[0],noise_feature[1],noise_feature[2]
            
            print('z1 and x1:',z1.shape,x.shape)
            x = (1-a)*x + a*z1
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
            x = (1-a)*x + a*z2
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
            x = (1-a)*x + a*z3
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        else:
             #不融合噪声
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
            x = slim.conv2d(x, 96, kernel_size=[3,3], stride=1, activation_fn = lrelu)
    
# ------------decoder部分------------
        
        print('decoder x',x.shape)
        x = slim.conv2d_transpose(x, 48, kernel_size=[3, 3], stride=2, activation_fn=lrelu)
        print('decoder x',x.shape)
        print('decoder D3',D3.shape)
    

        x += D3
        print('x+D3 ',x.shape)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 48, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d_transpose(x, 24, kernel_size=[3, 3], stride=2, activation_fn=lrelu)
        
        x += D2
        print('x + D2 ',x.shape)
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 24, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d_transpose(x, 12, kernel_size=[3, 3], stride=2, activation_fn=lrelu)
        
        x += D1
        print('x + D1 ',x.shape)
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d(x, 12, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        x = slim.conv2d_transpose(x, 6, kernel_size=[3, 3], stride=2, activation_fn=lrelu)
        print(x.shape)
        
        x = slim.conv2d(x, 1, kernel_size=[3,3], stride=1, activation_fn = lrelu)
        print(x.shape)
        return x
    
