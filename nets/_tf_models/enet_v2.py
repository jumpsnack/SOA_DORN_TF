import tensorflow as tf
import cv2
import numpy as np
import os


def PReLU(x, scope):
    # PReLU(x) = x if x > 0, alpha*x otherwise

    alpha = tf.get_variable(scope + "/alpha", shape=[1],
                            initializer=tf.constant_initializer(0), dtype=tf.float32)

    output = tf.nn.relu(x) + alpha * (x - abs(x)) * 0.5

    return output


# function for 2D spatial dropout:
def spatial_dropout(x, drop_prob,is_training):
    # x is a tensor of shape [batch_size, height, width, channels]

    keep_prob = 1.0 - drop_prob
    keep_prob = tf.cond(is_training, lambda :keep_prob, lambda :1.0)
    input_shape = x.get_shape().as_list()

    batch_size = 3
    channels = input_shape[3]

    # drop each channel with probability drop_prob:
    noise_shape = tf.constant(value=[batch_size, 1, 1, channels])
    x_drop = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)

    output = x_drop

    return output


# function for unpooling max_pool:
def max_unpool(inputs, pooling_indices, output_shape=None, k_size=[1, 2, 2, 1]):
    # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet

    # inputs has shape [batch_size, height, width, channels]

    # pooling_indices: pooling indices of the previously max_pooled layer

    # output_shape: what shape the returned tensor should have

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = tf.shape(inputs, out_type=tf.int32)

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices * batch_range
    y = pooling_indices // (output_shape[2] * output_shape[3])
    x = (pooling_indices // output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices * feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])
    output_shape[0] = 3
    ret = tf.scatter_nd(indices, values, tf.constant(output_shape))

    return ret


# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        19: [81, 0, 81]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color


class ENet_model(object):

    def __init__(self, model_id, img_height=512, img_width=1024, batch_size=4):
        self.model_id = model_id

        # self.project_dir = "/root/segmentation/"
        #
        # self.logs_dir = self.project_dir + "training_logs/"
        # if not os.path.exists(self.logs_dir):
        #     os.makedirs(self.logs_dir)

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.no_of_classes = 142

        self.wd = 2e-4  # (weight decay)
        self.lr = 5e-4  # (learning rate)

        # create all dirs for storing checkpoints and other log data:
        # self.create_model_dirs()

        # add placeholders to the comp. graph:
        self.add_placeholders()

        # define the forward pass, compute logits and add to the comp. graph:

    # def create_model_dirs(self):
    # self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
    # self.checkpoints_dir = self.model_dir + "checkpoints/"
    # self.debug_imgs_dir = self.model_dir + "imgs/"
    # if not os.path.exists(self.model_dir):
    #     os.makedirs(self.model_dir)
    #     os.makedirs(self.checkpoints_dir)
    #     os.makedirs(self.debug_imgs_dir)

    def add_placeholders(self):
        self.imgs_ph = tf.placeholder(tf.float32,
                                      shape=[self.batch_size, self.img_height, self.img_width, 3],
                                      name="imgs_ph")

        self.onehot_labels_ph = tf.placeholder(tf.float32,
                                               shape=[self.batch_size, self.img_height, self.img_width,
                                                      self.no_of_classes],
                                               name="onehot_labels_ph")

        # dropout probability in the early layers of the network:
        # self.early_drop_prob_ph = tf.placeholder(tf.float32, name="early_drop_prob_ph")
        self.early_drop_prob_ph = 0.01

        # dropout probability in the later layers of the network:
        # self.late_drop_prob_ph = tf.placeholder(tf.float32, name="late_drop_prob_ph")
        self.late_drop_prob_ph = 0.1

    def create_feed_dict(self, imgs_batch, early_drop_prob, late_drop_prob, onehot_labels_batch=None):
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.early_drop_prob_ph] = early_drop_prob
        feed_dict[self.late_drop_prob_ph] = late_drop_prob
        if onehot_labels_batch is not None:
            # only add the labels data if it's specified (during inference, we
            # won't have any labels):
            feed_dict[self.onehot_labels_ph] = onehot_labels_batch

        return feed_dict

    def get_logits(self, inputs, is_training):
        with tf.variable_scope('ENet'):
            # encoder:
            # # initial block:
            network = self.initial_block(x=inputs, is_training=is_training,scope="inital")
            print(network.get_shape().as_list())

            # # layer 1:
            # # # save the input shape to use in max_unpool in the decoder:
            inputs_shape_1 = network.get_shape().as_list()
            network, pooling_indices_1 = self.encoder_bottleneck_regular(x=network,
                                                                         is_training=is_training,
                                                                         output_depth=64,
                                                                         drop_prob=self.early_drop_prob_ph,
                                                                         scope="bottleneck_1_0", downsampling=True)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training,output_depth=64,
                                                      drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_1")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training,output_depth=64,
                                                      drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_2")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network,is_training=is_training, output_depth=64,
                                                      drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_3")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network,is_training=is_training, output_depth=64,
                                                      drop_prob=self.early_drop_prob_ph, scope="bottleneck_1_4")
            print(network.get_shape().as_list())

            # # layer 2:
            # # # save the input shape to use in max_unpool in the decoder:
            inputs_shape_2 = network.get_shape().as_list()
            network, pooling_indices_2 = self.encoder_bottleneck_regular(x=network,is_training=is_training,
                                                                         output_depth=128,
                                                                         drop_prob=self.late_drop_prob_ph,
                                                                         scope="bottleneck_2_0", downsampling=True)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_1")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_2",
                                                      dilation_rate=2)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training,output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_3")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training,output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_4",
                                                      dilation_rate=4)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training,output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_5")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_6",
                                                      dilation_rate=8)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network,is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_7")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_2_8",
                                                      dilation_rate=16)
            print(network.get_shape().as_list())

            # layer 3:
            network = self.encoder_bottleneck_regular(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2",
                                                      dilation_rate=2)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network,is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_3")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_4",
                                                      dilation_rate=4)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training,output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_5")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network,is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_6",
                                                      dilation_rate=8)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training,output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_7")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training,output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_8",
                                                      dilation_rate=16)
            print(network.get_shape().as_list())

            # layer 3-1
            network = self.encoder_bottleneck_regular(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_1")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_2",
                                                      dilation_rate=2)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_3")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_4",
                                                      dilation_rate=4)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_5")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_6",
                                                      dilation_rate=8)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_7")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_1_8",
                                                      dilation_rate=16)
            print(network.get_shape().as_list())

            # layer 3-2
            network = self.encoder_bottleneck_regular(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_1")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_2",
                                                      dilation_rate=2)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_3")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_4",
                                                      dilation_rate=4)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_regular(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_5")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_6",
                                                      dilation_rate=8)
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_asymmetric(x=network, is_training=is_training, output_depth=128,
                                                         drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_7")
            print(network.get_shape().as_list())

            network = self.encoder_bottleneck_dilated(x=network, is_training=is_training, output_depth=128,
                                                      drop_prob=self.late_drop_prob_ph, scope="bottleneck_3_2_8",
                                                      dilation_rate=16)
            print(network.get_shape().as_list())

            # decoder:
            # layer 4:
            network = self.decoder_bottleneck(x=network,is_training=is_training, output_depth=64,
                        scope="bottleneck_4_0", upsampling=True,
                        pooling_indices=pooling_indices_2, output_shape=inputs_shape_2)
            print(network.get_shape().as_list())

            network = self.decoder_bottleneck(x=network,is_training=is_training, output_depth=64,
                        scope="bottleneck_4_1")
            print(network.get_shape().as_list())

            network = self.decoder_bottleneck(x=network,is_training=is_training, output_depth=64,
                        scope="bottleneck_4_2")
            print(network.get_shape().as_list())


            # # layer 5:
            # network = self.decoder_bottleneck(x=network, output_depth=16,
            #             scope="bottleneck_5_0", upsampling=True,
            #             pooling_indices=pooling_indices_1, output_shape=inputs_shape_1)
            # print(network.get_shape().as_list())
            #
            # network = self.decoder_bottleneck(x=network, output_depth=16,
            #             scope="bottleneck_5_1")
            # print(network.get_shape().as_list())
            #
            #
            #
            # # fullconv:
            # network = tf.contrib.slim.conv2d_transpose(network, self.no_of_classes,
            #             [2, 2], stride=2, scope="fullconv", padding="SAME")
            # print(network.get_shape().as_list())

            self.logits = network
            return self.logits

    def initial_block(self, x, is_training,scope):
        # convolution branch:
        W_conv = self.get_variable_weight_decay(scope + "/W",
                                                shape=[3, 3, 3, 13],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b", shape=[13],  # ([out_depth])
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(x, W_conv, strides=[1, 2, 2, 1],
                                   padding="VALID") + b_conv

        # max pooling branch:
        pool_branch = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding="VALID")

        # concatenate the branches:
        concat = tf.concat([conv_branch, pool_branch], axis=3)  # (3: the depth axis)

        # apply batch normalization and PReLU:
        output = tf.contrib.slim.batch_norm(concat,is_training=is_training)
        output = PReLU(output, scope=scope)

        return output

    def encoder_bottleneck_regular(self, x, is_training,output_depth, drop_prob, scope,
                                   proj_ratio=4, downsampling=False):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth / proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        if downsampling:
            W_conv = self.get_variable_weight_decay(scope + "/W_proj",
                                                    shape=[2, 2, input_depth, internal_depth],
                                                    # ([filter_height, filter_width, in_depth, out_depth])
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 2, 2, 1],
                                       padding="VALID")  # NOTE! no bias terms
        else:
            W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                    shape=[1, 1, input_depth, internal_depth],
                                                    # ([filter_height, filter_width, in_depth, out_depth])
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="encoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                                       padding="VALID")  # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                shape=[3, 3, internal_depth, internal_depth],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth],  # ([out_depth])
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                                   padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               # ([filter_height, filter_width, in_depth, out_depth])
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob,is_training)

        # main branch:
        main_branch = x

        if downsampling:
            # max pooling with argmax (for use in max_unpool in the decoder):
            main_branch, pooling_indices = tf.nn.max_pool_with_argmax(main_branch,
                                                                      ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                                      padding="SAME")
            # (everytime we downsample, we also increase the feature block depth)

            # pad with zeros so that the feature block depth matches:
            depth_to_pad = output_depth - input_depth
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            # (paddings is an integer tensor of shape [4, 2] where 4 is the rank
            # of main_branch. For each dimension D (D = 0, 1, 2, 3) of main_branch,
            # paddings[D, 0] is the no of values to add before the contents of
            # main_branch in that dimension, and paddings[D, 0] is the no of
            # values to add after the contents of main_branch in that dimension)
            main_branch = tf.pad(main_branch, paddings=paddings, mode="CONSTANT")

        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        if downsampling:
            return output, pooling_indices
        else:
            return output

    def encoder_bottleneck_dilated(self, x, is_training,output_depth, drop_prob, scope,
                                   dilation_rate, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth / proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # dilated conv:
        W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                shape=[3, 3, internal_depth, internal_depth],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_losses")
        b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth],  # ([out_depth])
                                                initializer=tf.constant_initializer(0),
                                                loss_category="encoder_wd_losses")
        conv_branch = tf.nn.atrous_conv2d(conv_branch, W_conv, rate=dilation_rate,
                                          padding="SAME") + b_conv
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               # ([filter_height, filter_width, in_depth, out_depth])
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob,is_training)

        # main branch:
        main_branch = x

        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def encoder_bottleneck_asymmetric(self, x,is_training, output_depth, drop_prob, scope, proj_ratio=4):
        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth / proj_ratio)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/proj")

        # # asymmetric conv:
        # # # asymmetric conv 1:
        W_conv1 = self.get_variable_weight_decay(scope + "/W_conv1",
                                                 shape=[5, 1, internal_depth, internal_depth],
                                                 # ([filter_height, filter_width, in_depth, out_depth])
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv1, strides=[1, 1, 1, 1],
                                   padding="SAME")  # NOTE! no bias terms
        # # # asymmetric conv 2:
        W_conv2 = self.get_variable_weight_decay(scope + "/W_conv2",
                                                 shape=[1, 5, internal_depth, internal_depth],
                                                 # ([filter_height, filter_width, in_depth, out_depth])
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 loss_category="encoder_wd_losses")
        b_conv2 = self.get_variable_weight_decay(scope + "/b_conv2", shape=[internal_depth],  # ([out_depth])
                                                 initializer=tf.constant_initializer(0),
                                                 loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_conv2, strides=[1, 1, 1, 1],
                                   padding="SAME") + b_conv2
        # # # batch norm and PReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = PReLU(conv_branch, scope=scope + "/conv")

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               # ([filter_height, filter_width, in_depth, out_depth])
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="encoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        # NOTE! no PReLU here

        # # regularizer:
        conv_branch = spatial_dropout(conv_branch, drop_prob,is_training=is_training)

        # main branch:
        main_branch = x

        # add the branches:
        merged = conv_branch + main_branch

        # apply PReLU:
        output = PReLU(merged, scope=scope + "/output")

        return output

    def decoder_bottleneck(self, x, is_training,output_depth, scope, proj_ratio=4,
                           upsampling=False, pooling_indices=None, output_shape=None):
        # NOTE! decoder uses ReLU instead of PReLU

        input_shape = x.get_shape().as_list()
        input_depth = input_shape[3]

        internal_depth = int(output_depth / proj_ratio)

        # main branch:
        main_branch = x

        if upsampling:
            # # 1x1 projection (to decrease depth to the same value as before downsampling):
            W_upsample = self.get_variable_weight_decay(scope + "/W_upsample",
                                                        shape=[1, 1, input_depth, output_depth],
                                                        # ([filter_height, filter_width, in_depth, out_depth])
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        loss_category="decoder_wd_losses")
            main_branch = tf.nn.conv2d(main_branch, W_upsample, strides=[1, 1, 1, 1],
                                       padding="VALID")  # NOTE! no bias terms
            # # # batch norm:
            main_branch = tf.contrib.slim.batch_norm(main_branch,is_training=is_training)
            # NOTE! no ReLU here

            # # max unpooling:
            main_branch = max_unpool(main_branch, pooling_indices, output_shape)

        main_branch = tf.cast(main_branch, tf.float32)

        # convolution branch:
        conv_branch = x

        # # 1x1 projection:
        W_proj = self.get_variable_weight_decay(scope + "/W_proj",
                                                shape=[1, 1, input_depth, internal_depth],
                                                # ([filter_height, filter_width, in_depth, out_depth])
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_proj, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = tf.nn.relu(conv_branch)

        # # conv:
        if upsampling:
            # deconvolution:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                    shape=[3, 3, internal_depth, internal_depth],
                                                    # ([filter_height, filter_width, in_depth, out_depth])
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth],
                                                    # ([out_depth]], one bias weight per out depth layer),
                                                    initializer=tf.constant_initializer(0),
                                                    loss_category="decoder_wd_losses")
            main_branch_shape = main_branch.get_shape().as_list()
            output_shape = tf.convert_to_tensor([3,
                                                 main_branch_shape[1], main_branch_shape[2], internal_depth])
            conv_branch = tf.nn.conv2d_transpose(conv_branch, W_conv, output_shape=output_shape,
                                                 strides=[1, 2, 2, 1], padding="SAME") + b_conv
        else:
            W_conv = self.get_variable_weight_decay(scope + "/W_conv",
                                                    shape=[3, 3, internal_depth, internal_depth],
                                                    # ([filter_height, filter_width, in_depth, out_depth])
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    loss_category="decoder_wd_losses")
            b_conv = self.get_variable_weight_decay(scope + "/b_conv", shape=[internal_depth],  # ([out_depth])
                                                    initializer=tf.constant_initializer(0),
                                                    loss_category="decoder_wd_losses")
            conv_branch = tf.nn.conv2d(conv_branch, W_conv, strides=[1, 1, 1, 1],
                                       padding="SAME") + b_conv
        # # # batch norm and ReLU:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        conv_branch = tf.nn.relu(conv_branch)

        # # 1x1 expansion:
        W_exp = self.get_variable_weight_decay(scope + "/W_exp",
                                               shape=[1, 1, internal_depth, output_depth],
                                               # ([filter_height, filter_width, in_depth, out_depth])
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               loss_category="decoder_wd_losses")
        conv_branch = tf.nn.conv2d(conv_branch, W_exp, strides=[1, 1, 1, 1],
                                   padding="VALID")  # NOTE! no bias terms
        # # # batch norm:
        conv_branch = tf.contrib.slim.batch_norm(conv_branch,is_training=is_training)
        # NOTE! no ReLU here

        # NOTE! no regularizer

        # add the branches:
        merged = conv_branch + main_branch

        # apply ReLU:
        output = tf.nn.relu(merged)

        return output

    def get_variable_weight_decay(self, name, shape, initializer, loss_category,
                                  dtype=tf.float32):
        variable = tf.get_variable(name, shape=shape, dtype=dtype,
                                   initializer=initializer)

        # add a variable weight decay loss:
        weight_decay = self.wd * tf.nn.l2_loss(variable)
        tf.add_to_collection(loss_category, weight_decay)

        return variable
