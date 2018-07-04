from read_wav_extract_feature import Data
import tensorflow as tf
import math
import numpy as np


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)



def construct_cnn(x_shape, class_num, batch_size, learing_rate):
    x = tf.placeholder(tf.float32, [None, x_shape[0] * x_shape[1]])
    y = tf.placeholder(tf.float32, [None, class_num])
    x_image = tf.reshape(x, [-1, x_shape[0], x_shape[1], 1])

    # 第一个卷积层
    filter1_shape = [5, 5, 1, 32]
    bias1_shape = [32]
    stddev1 = 0.1
    strides1 = [1, 2, 2, 1]
    # 第二个卷积层
    filter2_shape = [5, 5, 32, 32]
    bias2_shape = [32]
    stddev2 = 0.1
    strides2 = [1, 1, 1, 1]
    # 第一个池化层
    ksize1 = [1, 2, 2, 1]
    pool_strides1 = [1, 2, 2, 1]
    # keep_prob1 = 0.7
    keep_prob1 = tf.placeholder(tf.float32)

    # 第三个卷积层
    filter3_shape = [5, 5, 32, 64]
    bias3_shape = [64]
    stddev3 = 0.1
    strides3 = [1, 1, 1, 1]
    # 第四个卷积层
    filter4_shape = [5, 5, 64, 64]
    bias4_shape = [64]
    stddev4 = 0.1
    strides4 = [1, 1, 1, 1]
    # 第二个池化层
    ksize2 = [1, 2, 2, 1]
    pool_strides2 = [1, 2, 2 ,1]
    # keep_prob2 = 0.7
    keep_prob2 = tf.placeholder(tf.float32)

    # 第五个卷积层
    filter5_shape = [5, 5, 64, 128]
    bias5_shape = [128]
    stddev5 = 0.1
    strides5 = [1, 1, 1, 1]
    # 第六个卷积层
    filter6_shape = [5, 5, 128, 128]
    bias6_shape = [128]
    stddev6 = 0.1
    strides6 = [1, 1, 1, 1]
    # 第七个卷积层
    filter7_shape = [5, 5, 128, 128]
    bias7_shape = [128]
    stddev7 = 0.1
    strides7 = [1, 1, 1, 1]
    # 第八个卷积层
    filter8_shape = [5, 5, 128, 128]
    bias8_shape = [128]
    stddev8 = 0.1
    strides8 = [1, 1, 1, 1]
    # 第三个池化层
    ksize3 = [1, 2, 2, 1]
    pool_strides3 = [1, 2, 2, 1]
    # keep_prob3 = tf.placeholder(tf.)
    keep_prob3 = tf.placeholder(tf.float32)

    # 第九个卷积层
    filter9_shape = [3, 3, 128, 512]
    bias9_shape = [512]
    stddev9 = 0.1
    strides9 = [1, 1, 1, 1]
    # keep_prob4 = 0.5
    keep_prob4 = tf.placeholder(tf.float32)
    # 第十个卷积层
    filter10_shape = [1, 1, 512, 512]
    bias10_shape = [512]
    stddev10 = 0.1
    strides10 = [1, 1, 1, 1]
    # keep_prob5 = 0.5
    keep_prob5 = tf.placeholder(tf.float32)

    # 第11个卷积层
    filter11_shape = [1, 1, 512, 10]
    bias11_shape = [10]
    stddev11 = 0.1
    strides11 = [1, 1, 1, 1]

    # 第一个全连接层
    flat_shape = [-1, 10 * 8 * 8]
    fc1_shape = [10 * 8 * 8, 1024]
    b_fc1_shape = [1024]
    #第二个全连接层
    fc2_shape = [1024, class_num]
    b_fc2_shape = [class_num]

    """
    # 第一个全连接层
    flat_shape = [-1, 32 * math.ceil(x_shape[0] * x_shape[1] / 4 / 4)]
    fc1_shape = [32 * math.ceil(x_shape[0] * x_shape[1] / 4 / 4), 1024]
    b_fc1_shape = [1024]
    # 第二个全连接层
    fc2_shape = [1024, class_num]
    b_fc2_shape = [class_num]
    """
    W_conv1 = weight_variable(filter1_shape, stddev1)
    b_conv1 = bias_variable(bias1_shape)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides1, padding="SAME") + b_conv1)

    W_conv2 = weight_variable(filter2_shape, stddev2)
    b_conv2 = bias_variable(bias2_shape)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides2, padding="SAME") + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize = ksize1, strides=pool_strides1, padding="SAME")
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob=keep_prob1)

    W_conv3 = weight_variable(filter3_shape, stddev3)
    b_conv3 = bias_variable(bias3_shape)
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2_drop, W_conv3, strides=strides3, padding="SAME") + b_conv3)

    W_conv4 = weight_variable(filter4_shape, stddev4)
    b_conv4 = bias_variable(bias4_shape)
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=strides4, padding="SAME") + b_conv4)
    h_pool4 = tf.nn.max_pool(h_conv4, ksize=ksize2, strides=pool_strides2, padding="SAME")
    h_pool4_drop = tf.nn.dropout(h_pool4, keep_prob=keep_prob2)

    W_conv5 = weight_variable(filter5_shape, stddev5)
    b_conv5 = bias_variable(bias5_shape)
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4_drop, W_conv5, strides=strides5, padding="SAME") + b_conv5)

    W_conv6 = weight_variable(filter6_shape, stddev6)
    b_conv6 = bias_variable(bias6_shape)
    h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, W_conv6, strides=strides6, padding="SAME") + b_conv6)

    W_conv7 = weight_variable(filter7_shape, stddev7)
    b_conv7 = bias_variable(bias7_shape)
    h_conv7 = tf.nn.relu(tf.nn.conv2d(h_conv6, W_conv7, strides=strides7, padding="SAME") + b_conv7)


    W_conv8 = weight_variable(filter8_shape, stddev8)
    b_conv8 = bias_variable(bias8_shape)
    h_conv8 = tf.nn.relu(tf.nn.conv2d(h_conv7, W_conv8, strides=strides8, padding="SAME") + b_conv8)
    h_pool8 = tf.nn.max_pool(h_conv8, ksize=ksize3, strides=pool_strides3, padding="SAME")
    h_pool8_drop = tf.nn.dropout(h_pool8, keep_prob=keep_prob3)

    W_conv9 = weight_variable(filter9_shape, stddev9)
    b_conv9 = bias_variable(bias9_shape)
    h_conv9 = tf.nn.relu(tf.nn.conv2d(h_pool8_drop, W_conv9, strides=strides9, padding="SAME") + b_conv9)
    h_conv9_drop = tf.nn.dropout(h_conv9, keep_prob=keep_prob4)

    W_conv10 = weight_variable(filter10_shape, stddev10)
    b_conv10 = bias_variable(bias10_shape)
    h_conv10 = tf.nn.relu(tf.nn.conv2d(h_conv9_drop, W_conv10, strides=strides10, padding="SAME") + b_conv10)
    h_conv10_drop = tf.nn.dropout(h_conv10, keep_prob=keep_prob5)

    W_conv11 = weight_variable(filter11_shape, stddev11)
    b_conv11 = bias_variable(bias11_shape)
    h_conv11 = tf.nn.relu(tf.nn.conv2d(h_conv10_drop, W_conv11, strides=strides11, padding="SAME") + b_conv11)
    h_global_avg_pool = tf.reduce_mean(h_conv11, axis = [1, 2])
    y_conv = tf.nn.softmax(h_global_avg_pool)

    """
    h_conv11_flat = tf.reshape(h_conv11, flat_shape)
    W_fc1 = weight_variable(fc1_shape, stddev1)
    b_fc1 = bias_variable(b_fc1_shape)
    h_fc1 = tf.nn.relu(tf.matmul(h_conv11_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable(fc2_shape, stddev1)
    b_fc2 = bias_variable(b_fc2_shape)

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_pool2_flat = tf.reshape(h_pool2_drop, flat_shape)
    W_fc1 = weight_variable(fc1_shape, stddev1)
    b_fc1 = bias_variable(b_fc1_shape)

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable(fc2_shape, stddev1)
    b_fc2 = bias_variable(b_fc2_shape)

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    """
    y_conv = tf.clip_by_value(y_conv, 1e-10, 1)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), 1))
    # cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
    product_y_y_conv = y * tf.log(y_conv)
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(cross_entropy)

    correct_predition = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predition, 'float32'))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_data = Data("train_data.npy", "train_labels.npy")
    total_num = train_data.get_total_num()

    Test = Data("evaluation__data.npy", "evaluation_labels.npy")
    test_total_num = Test.get_total_num()
    # test_data, test_labels = Test.get_all_data()

    for epoch in range(1000):
        batch_num = math.floor(total_num / batch_size)
        for i in range(batch_num):
            batch = train_data.get_next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob1: 0.7, keep_prob2: 0.5, keep_prob3: 0.5, keep_prob4: 0.5, keep_prob5: 0.5})
            # p =  sess.run(product_y_y_conv, feed_dict = {x:batch[0], y:batch[1]})
            # print(type(p))
            # print(p)
            #if i % 100 == 0:
            train_accuracy, loss  = sess.run([accuracy, cross_entropy], feed_dict={x:batch[0], y:batch[1], keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, keep_prob4: 1.0, keep_prob5: 1.0})
            # y_conv, y= sess.run([y_conv, y], feed_dict = {x:batch[0], y:batch[1]})
            # print("cross entropy shape: ", y_conv, ", \n", y)
            print("step %d, training accuracy: %g, loss = %g" % (i, train_accuracy, loss))

        test_batch_num = math.floor(test_total_num / batch_size)
        test_accuracy = 0.0
        for i in range(test_batch_num):
            test_data, test_labels = Test.get_next_batch(batch_size)
            test_acc = sess.run(accuracy, feed_dict = {x:test_data, y:test_labels, keep_prob1:1.0, keep_prob2:1.0, keep_prob3:1.0, keep_prob4:1.0, keep_prob5:1.0})
            test_accuracy += test_acc

        test_accuracy /= float(test_batch_num)
        print("                                                         ")
        print("=========================================================")
        print("=========================================================")
        print("epoch %d: test accuracy: %g" % (epoch, test_accuracy))
        print("=========================================================")
        print("=========================================================")
        print("                                                         ")

    #print("test accuracy: %g" % sess.run(accuracy, feed_dict={x:test_data, y:test_labels}))

if __name__ == "__main__":
    x_shape = [128, 128]
    class_num = 10
    batch_size = 128 
    learing_rate = 1e-4
    construct_cnn(x_shape, class_num, batch_size, learing_rate)
