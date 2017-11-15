import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pool(h):
    return tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def network(dict, dict_test):
    x_image = tf.placeholder(tf.float32, [None, 48, 48, 3])
    mask = tf.placeholder(tf.float32, [None, 6, 6])
    reg_mask = tf.placeholder(tf.float32, [None, 6, 6, 3])

    tf.summary.image('images', x_image)

    with tf.name_scope('conv1'):
        W = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1), name='W')
        norm = tf.contrib.layers.batch_norm(conv2d(x_image, W), is_training=True,
                                            scale=True, fused=True)
        h = tf.nn.relu(norm)
        p1 = pool(h)

    with tf.name_scope('conv2'):
        W = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W')
        norm = tf.contrib.layers.batch_norm(conv2d(p1, W), is_training=True,
                                            scale=True, fused=True)
        h = tf.nn.relu(norm)
        p2 = pool(h)

    with tf.name_scope('conv3'):
        W = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1), name='W')
        norm = tf.contrib.layers.batch_norm(conv2d(p2, W), is_training=True,
                                            scale=True, fused=True)
        h = tf.nn.relu(norm)
        p3 = pool(h)

    with tf.name_scope('conv4'):
        W = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name='W')
        norm = tf.contrib.layers.batch_norm(conv2d(p3, W), is_training=True,
                                            scale=True, fused=True)
        p4 = tf.nn.relu(norm)

    with tf.name_scope('intermediate'):
        W = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='W')
        norm = tf.contrib.layers.batch_norm(conv2d(p4, W), is_training=True,
                                            scale=True, fused=True)
        p5 = tf.nn.relu(norm)

    with tf.name_scope('classify'):
        W = tf.Variable(tf.truncated_normal([1, 1, 256, 1], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')
        mask_hat = tf.nn.sigmoid(conv2d(p5, W) + b)
        mask_hat = tf.reshape(mask_hat, [-1, 6, 6])

    with tf.name_scope('regression'):
        W = tf.Variable(tf.truncated_normal([1, 1, 256, 3], stddev=0.1), name='W')
        b = tf.Variable(tf.constant([24., 24., 32.]), name='b')
        reg_hat = conv2d(p5, W) + b


    two = tf.constant(2, dtype=tf.float32)
    cond = tf.not_equal(mask, two)
    cross_entropy_classify = tf.reduce_sum(
        tf.where(cond, tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=mask_hat), tf.zeros_like(mask)))
    effect_area = tf.reduce_sum(tf.where(cond, tf.ones_like(mask), tf.zeros_like(mask)))
    cross_entropy_classify /= effect_area

    zero = tf.constant(0, dtype=tf.float32)
    cond_reg = tf.not_equal(reg_mask, zero)
    reg_hat_norm = tf.stack(
        [tf.divide(reg_hat[:, :, :, 0], 32), tf.divide(reg_hat[:, :, :, 1], 32), tf.log(reg_hat[:, :, :, 2])])
    reg_hat_norm = tf.transpose(reg_hat_norm, [1, 2, 3, 0])
    reg_mask_norm = tf.stack(
        [tf.divide(reg_mask[:, :, :, 0], 32), tf.divide(reg_mask[:, :, :, 1], 32), tf.log(reg_mask[:, :, :, 2])])
    reg_mask_norm = tf.transpose(reg_mask_norm, [1, 2, 3, 0])

    cross_entropy_regression = tf.reduce_sum(
        tf.where(cond_reg, tf.losses.huber_loss(reg_hat_norm, reg_mask_norm, reduction=tf.losses.Reduction.NONE),
                 tf.zeros_like(reg_mask_norm)))
    effect_area_reg = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_mask_norm), tf.zeros_like(reg_mask_norm)))
    cross_entropy_regression /= effect_area_reg

    cross_entropy = 100 * cross_entropy_regression + cross_entropy_classify
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct = tf.reduce_sum(tf.where(cond, tf.cast(abs(mask - mask_hat) < 0.5, tf.float32), tf.zeros_like(mask)))
    effect_area = tf.reduce_sum(tf.where(cond, tf.ones_like(mask), tf.zeros_like(mask)))
    accuracy = correct / effect_area

    correct_reg = tf.reduce_sum(
        tf.where(cond_reg, tf.cast(abs(reg_mask - reg_hat) < 1, tf.float32), tf.zeros_like(reg_mask)))
    effect_area = tf.reduce_sum(tf.where(cond_reg, tf.ones_like(reg_mask), tf.zeros_like(reg_mask)))
    accuracy_reg = correct_reg / effect_area

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('accuracy_reg', accuracy_reg)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('log')
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            arr = np.arange(dict['data'].shape[0])
            np.random.shuffle(arr)
            batch_data = dict['data'][arr[:100], :, :, :]
            batch_mask = dict['mask'][arr[:100], :, :]
            batch_reg_mask = dict['x_y_w'][arr[:100], :, :, :]
            if i % 100 == 0:
                l, lc, lr, ac, ar = sess.run(
                    [cross_entropy, cross_entropy_classify, cross_entropy_regression, accuracy, accuracy_reg],
                    {x_image: batch_data, mask: batch_mask, reg_mask: batch_reg_mask})
                s = sess.run(merged, {x_image: batch_data, mask: batch_mask, reg_mask: batch_reg_mask})
                writer.add_summary(s, i)

                # train_accuracy = accuracy.eval(feed_dict={
                # 					x_image: batch_data, mask: batch_mask})
                print(
                'step %d, training loss %g, training classify loss %g, training reg loss %g, training classify accuracy %g, training reg accuracy %g' % (
                i, l, lc, lr, ac, ar))
            train_step.run(feed_dict={x_image: batch_data, mask: batch_mask, reg_mask: batch_reg_mask})

        batch_size = 100
        test_accuracy = 0
        test_accuracy_reg = 0
        test_loss = 0
        test_loss_classify = 0
        test_loss_reg = 0
        for i in range(10000 / batch_size):
            l, lc, lr, ac, ar = sess.run(
                [cross_entropy, cross_entropy_classify, cross_entropy_regression, accuracy, accuracy_reg],
                feed_dict={x_image: dict_test['data'][i * batch_size:(i + 1) * batch_size, :, :, :],
                           mask: dict_test['mask'][i * batch_size:(i + 1) * batch_size, :, :],
                           reg_mask: dict_test['x_y_w'][i * batch_size:(i + 1) * batch_size, :, :, :]})
            test_loss += l
            test_loss_classify += lc
            test_loss_reg += lr
            test_accuracy += ac
            test_accuracy_reg += ar
            # test_accuracy += accuracy.eval(
            #     feed_dict={x_image: dict_test['data'][i * batch_size:(i + 1) * batch_size, :, :, :],
            #                mask: dict_test['mask'][i * batch_size:(i + 1) * batch_size, :, :],
            #                reg_mask: dict_test['x_y_w'][i * batch_size:(i + 1) * batch_size, :, :, :]})
            # test_accuracy_reg += accuracy_reg.eval(
            #     feed_dict={x_image: dict_test['data'][i * batch_size:(i + 1) * batch_size, :, :, :],
            #                mask: dict_test['mask'][i * batch_size:(i + 1) * batch_size, :, :],
            #                reg_mask: dict_test['x_y_w'][i * batch_size:(i + 1) * batch_size, :, :, :]})
        print('test loss %g, test classify loss %g, test reg loss %g, test classify accuracy %g, test reg accuracy %g' %(
               test_loss * batch_size / 10000, test_loss_classify * batch_size / 10000,
               test_loss_reg * batch_size / 10000, test_accuracy * batch_size / 10000, test_accuracy_reg * batch_size / 10000))





if __name__ == '__main__':
    dict = {'data': [], 'mask': [], 'x_y_w': []}
    with open('../data/cifar10_transformed/devkit/train.txt') as f:
        for line in f:
            file_info = line.split()
            file_name = file_info[0]
            # dict['label_center_width'].append(file_info[1:])
            image = '../data/cifar10_transformed/imgs/' + file_name
            mask_image = '../data/cifar10_transformed/masks/' + file_name
            # plt.imshow(mpimg.imread(mask))
            # plt.show()
            dict['data'].append(mpimg.imread(image))
            float_mask = mpimg.imread(mask_image)
            for i in range(float_mask.shape[0]):
                for j in range(float_mask.shape[0]):
                    if float_mask[i, j] > 0.007:
                        float_mask[i, j] = 2
                    elif float_mask[i, j] > 0.003 and float_mask[i, j] < 0.004:
                        float_mask[i, j] = 1
            dict['mask'].append(float_mask)
            x_y_w_map = np.zeros((float_mask.shape[0], float_mask.shape[1], 3), dtype=np.float32)
            x_y_w_map[float_mask == 1, 0] = file_info[2]
            x_y_w_map[float_mask == 1, 1] = file_info[3]
            x_y_w_map[float_mask == 1, 2] = file_info[4]
            dict['x_y_w'].append(x_y_w_map)
    dict['data'] = np.array(dict['data'])
    # dict['label_center_width'] = np.array(dict['label_center_width'])
    dict['mask'] = np.array(dict['mask'])
    dict['x_y_w'] = np.array(dict['x_y_w'])

    dict_test = {'data': [], 'mask': [], 'x_y_w': []}
    with open('../data/cifar10_transformed/devkit/test.txt') as f:
        for line in f:
            file_info = line.split()
            file_name = file_info[0]
            image = '../data/cifar10_transformed/imgs/' + file_name
            mask_image = '../data/cifar10_transformed/masks/' + file_name
            dict_test['data'].append(mpimg.imread(image))
            float_mask = mpimg.imread(mask_image)
            for i in range(float_mask.shape[0]):
                for j in range(float_mask.shape[0]):
                    if float_mask[i, j] > 0.007:
                        float_mask[i, j] = 2
                    elif float_mask[i, j] > 0.003 and float_mask[i, j] < 0.004:
                        float_mask[i, j] = 1
            dict_test['mask'].append(float_mask)
            x_y_w_map = np.zeros((float_mask.shape[0], float_mask.shape[1], 3), dtype=np.float32)
            x_y_w_map[float_mask == 1, 0] = file_info[2]
            x_y_w_map[float_mask == 1, 1] = file_info[3]
            x_y_w_map[float_mask == 1, 2] = file_info[4]
            dict_test['x_y_w'].append(x_y_w_map)
    dict_test['data'] = np.array(dict_test['data'])
    dict_test['x_y_w'] = np.array(dict_test['x_y_w'])
    dict_test['mask'] = np.array(dict_test['mask'])

    network(dict, dict_test)
