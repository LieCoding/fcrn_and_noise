# fcrn_and_noise

综合实训的代码，FCRN加上噪声后检测船只

去掉了网络中的支路

main.py使用了多线程队列的方式读取数据，加快了训练速度  

only_fcrn.py 为了做对比实验，单独使用了FCRN网络训练，

test_net增加了准确率计算方法，test_net_fcrn为FCRN的测试程序

