#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#-----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import argparse
import colorsys
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

    def detect(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])

        output = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: image})

        results = self.calc_output(output)

        for i in range(len(results)):
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)

        return results

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


    def label_read(self, label_id):
        output = np.zeros([1, 15], np.int32)
        output[:, label_id] = 1.0
        return output

    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.0 * 2.0 - 1.0
        return image

    def calc_output(self, output):
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])    #boxes coordinate
        boxes = self.get_boxes(boxes) * self.image_size

        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])    #the confidence of the each anchor boxes
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))

        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])    #classes
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))

        probs = classes * confidence

        filter_probs = np.array(probs >= self.threshold, dtype = 'bool')
        filter_index = np.nonzero(filter_probs)
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
        probs_filter = probs[filter_probs]
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]

        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0

        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append([self.classes[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i]])

        return results

    def get_boxes(self, boxes):
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 5]) / self.cell_size,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 5]) / self.cell_size])

        return np.transpose(boxes1, (1, 2, 3, 0))


    def calc_iou(self, box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors


    def draw(self, image, result):
        image_h, image_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))


    def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def video_detect(self, cap):
        while(1):
            ret, image = cap.read()
            if not ret:
                print('Cannot capture images from device')
                break

            result = self.detect(image)
            self.draw(image, result)
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


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

    #detect the video

    #cap = cv2.VideoCapture('asd.mp4')
    #cap = cv2.VideoCapture(0)
    #detector.video_detect(cap)

    #detect the image

    image_files_path = './linemod/cfg/test_shuf_labels.txt'
    #imagename = './test/02.jpg'
    #detector.image_detect(imagename)
    detector.test(image_files_path)


if __name__ == '__main__':
    main()
