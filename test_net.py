# 测试验证集的网络，在训练的网络稍微修改了下
import tensorflow as tf
import reader, netStore, cv2, random,os,time,sys
import numpy as np

tf.reset_default_graph()

epoch =1#迭代次数
batch_size = 32
save_step = 50

a = 0.5
# 用于断点训练时候 保存文件名 
start_epoch = 130

#选择数据库
data_dir = 'dataset/val/image'

if not os.path.exists('./test_record'):
    os.makedirs('./test_record')

#读取图片列表
images = reader.read_list(data_dir,file_postfix='jpg')

h,w = 384, 512,
image = tf.placeholder(tf.float32, [None, h, w, 3]) 
noise_image = tf.placeholder(tf.float32, [None,h, w, 3])
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
lr = tf.train.exponential_decay(0.001,
                                global_step,
                                1000,
                                0.9,
                                staircase=True)#指数衰减
#设置优化器
optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=global_step)

# optimizer_G = tf.train.AdamOptimizer(lr).minimize(loss_G, global_step=global_step,var_list=G_vars)


saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Import models successful!')
    else:
        sess.run(tf.global_variables_initializer())
        print('模型加载失败')
        sys.exit()
    for i in range(epoch):
        start = time.time()
        index = 0
        for j in range(int(len(images)/batch_size)):
            if index + batch_size > len(images):
                list_temp = images[index:len(images)]
            else:
                list_temp = images[index:index+batch_size]
            
            #获取图片数据 label 加入了噪声的图片
            image_data, real_label_data ,noise_image_data = reader.read_data(data_dir,list_temp,output_size=(w, h))

            #------------训练网络-------------
            feeds = {image: image_data, label: real_label_data,noise_image:noise_image_data}
            
            input_image, input_noise_image,output_real_map,output_fake_map, input_label= sess.run([image,noise_image,real_map,
                                                                                                   fake_map,label], feeds)
            
#             _,input_image, input_noise_image,output_real_map,output_fake_map, input_label,\
#             decoder_loss_output, encoder_loss_output,G_loss_output, total_loss_output,\
#             learningrate = sess.run([optimizer,image,noise_image,real_map, fake_map,
#                                      label,loss_decoder,loss_encoder,loss_G,total_loss,lr], feeds)
            
            index += batch_size 
            
            
            
#         print('epoch:{} total_loss:{} decoder_loss:{} encoder_loss:{} G_loss:{} lr:{}'.format(i+start_epoch,
#                                                                                               total_loss_output,decoder_loss_output, 
#                                                                                            encoder_loss_output,G_loss_output, 
#                                                                                               learningrate))
        
        end = time.time()
        print("运行时间:%.2f秒"%(end-start))
        
        for image_id in range(len(input_image)):
                cv2.imwrite('./test_record/' + str(i+image_id) + str(j) + "_" +  '_image' + '.png' , input_image[image_id])
                cv2.imwrite('./test_record/' + str(i+image_id) + str(j) + "_" +  'input_noise_image' + '.png' , input_noise_image[image_id])
                cv2.imwrite('./test_record/' + str(i+image_id) + str(j) + "_" +  '_label' + '.png' , input_label[image_id])
                cv2.imwrite('./test_record/' + str(i+image_id) + str(j)+ "_" +  'output_real_map' + '.png' , output_real_map[image_id])
                cv2.imwrite('./test_record/' + str(i+image_id) + str(j)+ "_" +  'output_fake_map' + '.png' , output_fake_map[image_id])
                
    print("Done!")
    