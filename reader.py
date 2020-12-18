#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:27:24 2019

@author: hxx
"""
import cv2 as cv
import numpy as np
import sys ,os

def read_list(data_dir,file_postfix):
    output = []
    for i in os.listdir(data_dir):
        if i[-3:]==file_postfix:
            output.append(i) 
    return output

def gasuss_noise(image, mean=0, var=0.003):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
#     if out.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.
#     out = np.clip(out, low_clip, 1.0)
    noise_image = np.uint8(out*255)
    return noise_image

def read_data(data_dir,image_list,output_size=(512, 384), resize_mode=cv.INTER_AREA):

    image_output = []
    label_output = []
    noise_image_output = []

    
    for i in range(len(image_list)):
        image_name = os.path.join(data_dir,image_list[i])
        label_name =image_name.replace('image','label')
        
#         print(image_name,label_name)

        image = cv.imread(image_name)
        label = cv.imread(label_name,0)#读取灰度图
        


        
        if image is None :
            print(image_name,label_name)
            print("image无数据")
            print(image_list)
            sys.exit()
        if label is None:
            print("label无数据")
            sys.exit()
    #------------生成添加噪声的图片---------------------------------    
    
        noise_image = gasuss_noise(image)  
        image = cv.resize(image, output_size, interpolation=resize_mode)
        label = cv.resize(label, output_size, interpolation=resize_mode)
        noise_image = cv.resize(noise_image, output_size, interpolation=resize_mode)

        
        image_output.append(image)
        label_output.append(label)
        noise_image_output.append(noise_image)
        

    image_output = np.reshape(image_output, [len(image_list), output_size[1], output_size[0], 3])#3通道
    label_output = np.reshape(label_output, [len(image_list), output_size[1], output_size[0], 1])#单通道，若label不是单通道需修改
    noise_image_output = np.reshape(noise_image_output, [len(image_list), output_size[1], output_size[0],3])
    
    return image_output, label_output,noise_image_output

def read_data_one(data_dir,name,output_size=(512, 384), resize_mode=cv.INTER_AREA):

#     image_output = []
#     label_output = []
#     noise_image_output = []

    
#     for i in range(len(image_list)):
    image_name = os.path.join(data_dir,image_list[i])
    label_name =image_name.replace('image','label')

#         print(image_name,label_name)

    image = cv.imread(image_name)
    label = cv.imread(label_name,0)#读取灰度图




    if image is None :
        print(image_name,label_name)
        print("image无数据")
        print(image_list)
        sys.exit()
    if label is None:
        print("label无数据")
        sys.exit()
#------------生成添加噪声的图片---------------------------------    

    noise_image = gasuss_noise(image)  
    image = cv.resize(image, output_size, interpolation=resize_mode)
    label = cv.resize(label, output_size, interpolation=resize_mode)
    noise_image = cv.resize(noise_image, output_size, interpolation=resize_mode)


#     image_output.append(image)
#     label_output.append(label)
#     noise_image_output.append(noise_image)


#     image_output = np.reshape(image_output, [len(image_list), output_size[1], output_size[0], 3])#3通道
#     label_output = np.reshape(label_output, [len(image_list), output_size[1], output_size[0], 1])#单通道，若label不是单通道需修改
#     noise_image_output = np.reshape(noise_image_output, [len(image_list), output_size[1], output_size[0],3])
    
    return image, label,noise_image


