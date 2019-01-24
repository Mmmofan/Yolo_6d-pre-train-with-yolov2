#!/usr/bin/env python3
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
import cv2
import os

import yolo.config as cfg
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19

class Detector(object):
    def __init__(self, yolo, weights_file):
        self.yolo = yolo
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE * 2
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.threshold = cfg.THRESHOLD
        self.anchor = cfg.ANCHOR

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file)

    def test(self, image_path):
        with open(image_path, 'r') as f:
            image_list = [x.split() for x in f.readlines()]

        total_num = len(image_list)
        image_files = [image_list[i][0] for i in range(total_num)]
        label_number = [image_list[i][1] for i in range(total_num)]
        batch_id = 0
        accuracy_num = []

        images = np.zeros([self.batch_size, 416, 416, 3], np.float32)
        labels = np.zeros([self.batch_size, 15], np.int32)

        while batch_id <= int(total_num / self.batch_size):
            for idx in range(self.batch_size):
                images[idx] = self.image_read(image_files[idx + batch_id*self.batch_size])
                labels[idx] = self.label_read(int(label_number[idx + batch_id*self.batch_size]))

            #batch_images = images[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
            #batch_labels = labels[batch_id*self.batch_size:(batch_id+1)*self.batch_size]
            feed_dict = {self.yolo.images: images}
            output = self.sess.run(self.yolo.logits, feed_dict=feed_dict)
            output = self.softmax(output, axis=1)
            correct = 0
            for i in range(self.batch_size):
                out_max_idx = np.where(output[i]==np.max(output[i]))
                lab_max_idx = np.where(labels[i]==np.max(labels[i]))
                if_true = out_max_idx[0] == lab_max_idx[0]
                if if_true:
                    correct += 1
            acc = correct / self.batch_size
            batch_id += 1
            accuracy_num.append(correct)
            print("  In batch {}, batch accuracy: {}".format(batch_id, acc))
        accurate_number = sum(accuracy_num)
        accuracy = accurate_number / batch_id*self.batch_size

        print("  \n Total accuracy: {}".format(accuracy))

    def label_read(self, label_id):
        output = np.zeros([1, 15], np.int32)
        output[:, label_id] = 1.0
        return output

    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0 * 2.0 - 1.0
        return image

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)    # darknet-19.ckpt
    parser.add_argument('--weight_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '', type = str)    # which gpu to be selected
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    # configure gpu
    weights_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    yolo = yolo_v2(False)    # 'False' mean 'test'
    # yolo = Darknet19(False)

    detector = Detector(yolo, weights_file)

    #detect the image

    image_files_path = './linemod/cfg/test_shuf_labels.txt'
    #imagename = './test/02.jpg'
    #detector.image_detect(imagename)
    detector.test(image_files_path)


if __name__ == '__main__':
    main()
