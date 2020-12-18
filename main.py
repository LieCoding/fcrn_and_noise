#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用了异步读取数据的方式
"""

import tensorflow as tf
import reader, netStore, cv2, random,os,time
import numpy as np

tf.reset_default_graph()

epoch =700#迭代次数
batch_size = 64

# 超参数,噪声所占比例
a = 0.5

# 用于二次训练时候 保存的文件名能够接着上次训练的数字保存 
start_epoch = 800


#选择数据库
data_dir = 'dataset/train/image'

# 创建记录保存的图片文件夹
if not os.path.exists('./record'):
    os.makedirs('./record')

#读取文件夹图片的文件名
images = reader.read_list(data_dir,file_postfix='jpg')

# 图片的大小
h,w = 384, 512
# 定义占位符
image = tf.placeholder(tf.float32, [None, h, w, 3]) 
noise_image = tf.placeholder(tf.float32, [None,h, w, 3])
# 单通道label
label = tf.placeholder(tf.float32, [None, h, w, 1]) 

#获取编码器网络输出
real_feature,D1,D2,D3 = netStore.net_encoder1(image)
#获取噪声发生器网络输出
noise_feature = netStore.net_G_encoder2(noise_image)

# 真特征的检测输出
real_map  = netStore.net_decoder(real_feature,D1,D2,D3,a)
# 假特征的检测输出
fake_map  = netStore.net_decoder(real_feature,D1,D2,D3,a,noise_feature)


# 解码器的损失函数：
loss_decoder = tf.reduce_mean(((real_map-label)**2)+((fake_map-label)**2))
# 噪声发生器的损失：
loss_G = tf.reduce_mean(-1*(fake_map-label)**2)
# 编码器的损失函数：
loss_encoder = tf.reduce_mean(((real_map-label)**2)+((fake_map-label)**2))
# 总损失
total_loss = loss_decoder+loss_G+loss_encoder
#设置学习率
global_step = tf.Variable(0, trainable=False)  
# 学习率 指数衰减衰减
lr = tf.train.exponential_decay(0.001,global_step,1000,0.9,staircase=True)

#设置优化器
optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)


# 使用队列方式获取数据，由于单线程方式下，在使用cv2读取文件时候仅仅消耗CPU而且时间较长，就采用了异步方式。
def get_batch_data():
#     从本地一次性读取所有数据，
    image, label,noise_image = reader.read_data(data_dir,images)
    
#     设置输入队列  
    input_queue = tf.train.slice_input_producer([image, label,noise_image ], shuffle=True,num_epochs=None)
#     获取一个批次的数据，这里的类型为张量   32线程
    image_batch, label_batch,noise_image_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=32,
                                                                capacity=64,allow_smaller_final_batch=True)
    return image_batch,label_batch,noise_image_batch

# 定义保存模型操作
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# 定义一个批次的数据，类型为张量
image_data, label_data ,noise_image_data = get_batch_data()

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#这一行必须加，因为slice_input_producer的原因
    coord = tf.train.Coordinator()
    # 启动计算图中所有的队列线程
    threads = tf.train.start_queue_runners(sess,coord)
    
#     加载检查点，方便二次训练
    ckpt = tf.train.get_checkpoint_state('./model/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Import models successful!')
    else:
        sess.run(tf.global_variables_initializer())
        print('Initialize successful!')
        
     # 主线程，循环epoch
    for i in range(epoch):
#       计时用
        start = time.time()
#         由于批次类型为张量  这里先使用 run 获取到数据信息后再feed到网络中训练，
        feed_image,feed_label,feef_noise_image = sess.run([image_data, label_data ,noise_image_data])
#         喂入网络的数据
        feeds = {image: feed_image, label: feed_label,noise_image:feef_noise_image}
#         训练网络，获取相关信息
        _,input_image, input_noise_image,output_real_map,output_fake_map, input_label,\
        decoder_loss_output, encoder_loss_output,G_loss_output, total_loss_output,\
        learningrate = sess.run([optimizer,image,noise_image,real_map, fake_map,
                                 label,loss_decoder,loss_encoder,loss_G,total_loss,lr], feeds)
    
#           打印当前的损失
        print('epoch:{} total_loss:{} decoder_loss:{} encoder_loss:{} G_loss:{} lr:{}'.format(i+start_epoch,
                                                                                              total_loss_output,
                                                                                              decoder_loss_output, 
                                                                                           encoder_loss_output,
                                                                                              G_loss_output, 
                                                                                              learningrate))
        end = time.time()
        print("运行时间:%.2f秒"%(end-start))
        
#         50个epoch保存一次相关图片
        if i%50 == 0:
                cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  '_image' + '.png' , input_image[0])
                cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'input_noise_image' + '.png' , input_noise_image[0])
                cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  '_label' + '.png' , input_label[0])
                cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'output_real_map' + '.png' , output_real_map[0])
                cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'output_fake_map' + '.png' , output_fake_map[0])

#       训练10 epoch保存一次模型
        if i%100==0:
            saver.save(sess, './model/fcrn.ckpt', global_step=global_step)
        
#      最后保存一次相关图片       
    cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  '_image' + '.png' , input_image[0])
    cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'input_noise_image' + '.png' , input_noise_image[0])
    cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  '_label' + '.png' , input_label[0])
    cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'output_real_map' + '.png' , output_real_map[0])
    cv2.imwrite('./record/' + str(i+start_epoch) + "_" +  'output_fake_map' + '.png' , output_fake_map[0])
    print("Done!")
    
# 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(threads)
#     关闭会话
    sess.close()

    
