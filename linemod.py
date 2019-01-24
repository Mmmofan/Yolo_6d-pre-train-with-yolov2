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
import os
import yolo.config as cfg
import numpy as np
import cv2

class Linemod(object):
    def __init__(self):
        self.batch_size = cfg.BATCH_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.image_size = cfg.IMAGE_SIZE

        self.data_path = 'linemod/'
        self.config_path = os.path.join(self.data_path, 'cfg/')

        self.train_file = None
        self.test_file = None

        self.train_list_name = []
        self.test_list_name = []

        self.count = 0
        self.epoch = 1
        self.prepare()
        self.total_train_num = len(self.train_list_name)
        self.total_test_num = len(self.test_list_name)

    def prepare(self):
        self.train_file = os.path.join(self.config_path, 'train_shuf_labels.txt')
        self.test_file = os.path.join(self.config_path, 'test_shuf_labels.txt')

        with open(self.train_file, 'r') as train:
            train_list = [x.split() for x in train.readlines()]
        self.train_list_name = train_list

        with open(self.test_file, 'r') as test:
            test_list = [x.split() for x in test.readlines()]
        self.test_list_name = test_list

    def load_labels(self, phase):
        if phase == 'train':
            return self.train_list_name
        elif phase == 'test':
            return self.test_list_name

    def next_batches(self, label):
        images = np.zeros([self.batch_size, 416, 416, 3], dtype=np.float32)
        labels = np.zeros([self.batch_size, 15], dtype=np.float32)
        batch_list = label[self.count * self.batch_size : (self.count+1) * self.batch_size]
        num = 0
        while num < self.batch_size:
            images[num, :, :, :] = self.image_read(batch_list[num][0])
            labels[num] = self.label_read(batch_list[num][1])
            num += 1
        self.count += 1
        return images, labels

    def next_batches_test(self, label):
        images = np.zeros([self.batch_size, 416, 416, 3], dtype=np.float32)
        labels = np.zeros([self.batch_size, 15], dtype=np.float32)
        batch_list = label[self.count * self.batch_size : (self.count+1) * self.batch_size]
        num = 0
        while num < self.batch_size:
            images[num, :, :, :] = self.image_read(batch_list[num][0])
            labels[num] = self.label_read(batch_list[num][1])
            num += 1
        return images, labels

    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0 * 2.0 - 1.0
        return image

    def label_read(self, label_num):
        label = np.zeros([1, 15], dtype=np.float32)
        label_id = int(label_num)
        label[0, label_id] = 1
        return label
