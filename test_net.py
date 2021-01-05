# 测试验证集的网络，在训练的网络稍微修改了下
import tensorflow as tf
import reader, netStore, cv2, random,os,time,sys
import numpy as np

tf.reset_default_graph()

a = 0

#选择数据库
data_dir = './dataset/val/image'

if not os.path.exists('./test_record'):
    os.makedirs('./test_record')
    
def get_acc(label,im1):
    # print(im1.shape)
    # 像素准确率 PA(Pixel Accuracy)
    intersection ,all_pix= 0.,0.

    for i,j in zip(im1,label):
        for m,n in zip(i,j):
            if m!=0 and n!=0:
                intersection+=1
    for i in label:
        for m in i:
            if m!=0:
                all_pix += 1
    return intersection/all_pix

#读取图片列表
images = reader.read_list(data_dir,file_postfix='jpg')

h,w = 384, 512,
# 定义占位符
image = tf.placeholder(tf.float32, [None, h, w, 3]) 
noise_image = tf.placeholder(tf.float32, [None,h, w,4])
# 单通道label
label = tf.placeholder(tf.float32, [None, h, w, 1]) 

#获取编码器网络输出
real_feature = netStore.net_encoder1(image)
#获取噪声发生器网络输出
noise_feature = netStore.net_G_encoder2(noise_image)

# 真特征的检测输出
real_map  = netStore.net_decoder(real_feature,a)
# 假特征的检测输出
fake_map  = netStore.net_decoder(real_feature,a,noise_feature)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# 计算准确率
acc = 0.
im_num = 0.

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Import models successful!')
    else:
        print('模型加载失败')
        sys.exit()

    start = time.time()

    list_temp = images
    #获取图片数据 label 加入了噪声的图片
    image_data, real_label_data ,noise_image_data = reader.read_data(data_dir,list_temp,output_size=(w, h))

    #------------训练网络-------------
    feeds = {image: image_data, label: real_label_data,noise_image:noise_image_data}

    input_image,output_real_map,output_fake_map, input_label= sess.run([image,real_map,fake_map,label], feeds)

    end = time.time()
    print("运行时间:%.2f秒"%(end-start),'正在保存图片 计算准确率------')

    for image_id in range(len(input_image)):

        cv2.imwrite('./test_record/' + str(image_id)+  '_image' + '.png' , input_image[image_id])
        cv2.imwrite('./test_record/' + str(image_id) +  '_label' + '.png' , input_label[image_id])
        cv2.imwrite('./test_record/' + str(image_id)+ 'output_real_map' + '.png' , output_real_map[image_id])
        cv2.imwrite('./test_record/' + str(image_id) + 'output_fake_map' + '.png' , output_fake_map[image_id])

        #计算准确率cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im1 = cv2.imread('./test_record/' + str(image_id) +  '_label' + '.png')
        im2 = cv2.imread('./test_record/' + str(image_id)+ 'output_real_map' + '.png')
        im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
#         print(im1.shape)
        acc_id = get_acc(im1,im2)
        print(acc_id)
        acc+=acc_id
        im_num+=1
                
    acc /= im_num    
    print(acc*100,'%','im_num:',im_num)
    print("Done!")
    
