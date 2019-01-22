#!/usr/bin/env python
# -*- coding:utf-8 -*-
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : fa.py
#   Author      : Mofan
#   Created date: 2019-01-21 23:56:12
#   Description :
#
#================================================================
import tensorflow as tf
import numpy as np
import argparse
import datetime
import time
import os
import yolo.config as cfg

from pascal_voc import Pascal_voc
from linemod import Linemod
from six.moves import xrange
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19

class Train(object):
    def __init__(self, yolo, data):
        self.yolo = yolo
        self.data = data
        self.num_class = len(cfg.CLASSES)
        # self.max_step = cfg.MAX_ITER
        self.max_step = int(data.total_train_num / yolo.batch_size)
        self.saver_iter = cfg.SAVER_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.initial_learn_rate = cfg.LEARN_RATE
        self.output_dir = os.path.join(cfg.DATA_DIR, 'output')
        self.backup_dir = os.path.join(cfg.DATA_DIR, 'backup')
        # weight_file = os.path.join(self.output_dir, cfg.WEIGHTS_FILE)
        # weight_file = os.path.join(self.output_dir, 'yolo_v2.ckpt-3425')
        self.labels_test = None
        self.labels_train = None

        # self.variable_to_restore = tf.global_variables()
        self.variable_to_save = tf.global_variables()
        # self.restorer = tf.train.Saver(self.variable_to_restore)
        self.saver = tf.train.Saver(self.variable_to_save)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 1250, 0.1, name='learn_rate')
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 190, 10000, 15500], [1e-3, 5e-3, 1e-2, 1e-3, 1e-4])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step)

        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.average_op)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions())
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # print('Restore weights from:', weight_file)
        # self.restorer.restore(self.sess, weight_file)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        self.labels_train = self.data.load_labels('train')
        self.labels_test = self.data.load_labels('test')

        num = 1
        initial_time = time.time()
        lowest_loss = 1e4

        for step in xrange(1, self.max_step + 1):
            images, labels = self.data.next_batches(self.labels_train)
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels}

            if step % self.summary_iter == 0:
                if step % 30 == 0:
                    print("   Testing... ")
                    self.test()

                    summary_, loss, logit, _ = self.sess.run([self.summary_op, self.yolo.total_loss, self.yolo.logits, self.train_op], feed_dict = feed_dict)
                    sum_loss = 0

                    log_str = ('{} Epoch: {}, Step: {},\n train_Loss: {:.4f}, test_Loss: {:.4f},\n Remain: {}, learning rate: {}').format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step),
                        loss, sum_loss/num, self.remain(step, initial_time), self.learn_rate.eval(session=self.sess))
                    print(log_str)

                    if loss < lowest_loss and step > 100:
                        lowest_loss = loss
                        print('\n Best model! Save ckpt file to {} '.format(self.output_dir))
                        self.saver.save(self.sess, self.backup_dir + '/yolo_v2.ckpt', global_step = step)

                    if loss < 1e4:
                        pass
                    else:
                        print('loss > 1e04')
                        break

                else:
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)

                self.writer.add_summary(summary_, step)

            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)

            if step % self.saver_iter == 0:
                self.saver.save(self.sess, self.output_dir + '/yolo_v2.ckpt', global_step = step)

        self.saver.save(self.sess, self.output_dir + '/yolo_v2.ckpt', global_step=self.global_step)

    def test(self):
        sum_loss = 0
        images, labels = self.data.next_batches_test(self.labels_train)
        feed_dict = {self.yolo.images: images, self.yolo.labels: labels}
        logit, loss_t = self.sess.run([self.yolo.logits, self.yolo.total_loss], feed_dict = feed_dict)
        sum_loss += loss_t
        output = self.softmax(logit, axis=1)
        correct = 0
        for i in range(self.yolo.batch_size):
            out_max_idx = np.where(output[i]==np.max(output[i]))
            lab_max_idx = np.where(labels[i]==np.max(labels[i]))
            if_true = out_max_idx[0] == lab_max_idx[0]
            if if_true:
                correct += 1
        acc = correct / self.yolo.batch_size

        print('Epoch: {} in testing, loss: {}, accuracy: {}\n'.format(self.data.epoch, sum_loss, acc))

    def softmax(self, X, theta = 1.0, axis = None):
        """
        Compute the softmax of each element along an axis of X.
        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.
        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        # make X at least 2d
        y = np.atleast_2d(X)
        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        # multiply y against the theta parameter,
        y = y * float(theta)
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis = axis), axis)
        # exponentiate y
        y = np.exp(y)
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
        # finally: divide elementwise
        p = y / ax_sum
        # flatten if X was 1D
        if len(X.shape) == 1: p = p.flatten()

        return p

    def remain(self, i, start):
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))

    def __del__(self):
        self.sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)  # darknet-19.ckpt
    parser.add_argument('--gpu', default = '', type = str)  # which gpu to be selected
    parser.add_argument('--name', default = 'ape', type = str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS_FILE = args.weights

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo = yolo_v2()
    # yolo = Darknet19()
    #pre_data = Pascal_voc()
    pre_data = Linemod()

    train = Train(yolo, pre_data)

    print('start training ...')
    train.train()
    print('successful training.')


if __name__ == '__main__':
    main()
