#!/usr/bin/python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# This code is derived from the MTCNN implementation of David Sandberg for Facenet
# (https://github.com/davidsandberg/facenet/)
# It has been rebuilt from scratch, taking the David Sandberg's implementation as a reference.
#
import copy

import cv2
import numpy as np
import pkg_resources
import tensorflow as tf

import math

from mtcnn.exceptions import InvalidImage
from mtcnn.network.factory import NetworkFactory

import memory_profiler
from memory_profiler import profile

__author__ = "Iván de Paz Centeno"


def IoU(boxA,
        boxB):  # Code taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    # print("BoxA: ")
    # print(boxA)
    # print("BoxB: ")
    # print(boxB)

    lambda_IoU = 0.6  # To match paper's concrete parameters

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # MTCNN doesn't give the end points in [2] and [3] but the distance from the start point to the end point
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # print("xA = %d -- yA = %d -- xB = %d -- yB = %d" % (xA,yA,xB,yB))
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # print("interArea = %d -- boxAArea = %d -- boxBArea = %d" % (interArea, boxAArea, boxBArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou > lambda_IoU


def not_negative(x):
    if x < 0:
        return 0
    else:
        return x


class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """

    def __init__(self, pad_result: tuple = None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class MTCNN(object):
    """
    Allows to perform MTCNN Detection ->
        a) Detection of faces (with the confidence probability)
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    """

    def __init__(self, weights_file: str = None, min_face_size: int = 20, steps_threshold: list = None,
                 scale_factor: float = 0.709):
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        if weights_file is None:
            weights_file = pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy')

        self.tape = tf.GradientTape(persistent=True)

        self._min_face_size = min_face_size
        self._steps_threshold = steps_threshold
        self._scale_factor = scale_factor

        # self.i = 0 # TO REMOVE

        self._pnet, self._rnet, self._onet = NetworkFactory().build_P_R_O_nets_from_file(weights_file)

    @property
    def min_face_size(self):
        return self._min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=20):
        try:
            self._min_face_size = int(mfc)
        except ValueError:
            self._min_face_size = 20

    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self._scale_factor, factor_count)]
            min_layer = min_layer * self._scale_factor
            factor_count += 1

        return scales

    @staticmethod
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        """
        if 'tensor' in str(type(image)):
          image_real = image
          image = image.numpy()
        """

        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = tf.image.resize(image, (height_scaled, width_scaled))

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(imap, reg, scale, t):

        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])

        y, x = np.where(imap >= t)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        score = imap[(y, x)]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

        if reg.size == 0:
            reg = np.empty(shape=(0, 3))

        bb = np.transpose(np.vstack([y, x]))

        q1 = np.fix((stride * bb + 1) / scale)
        q2 = np.fix((stride * bb + cellsize) / scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

        return boundingbox, reg

    @staticmethod
    def __tf_nms(boxes, threshold, method):
        """
        Non Maximum Suppression.

        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """

        if tf.size(boxes) == 0:  # start here
            print("tf_nms encountered an error, outputs tensor")
            quit()
            return tf.reshape(tf.convert_to_tensor(()), (0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            # print("TIFANYHI",x1, i, idx)
            x1idx = []
            for j in idx:
                x1idx.append(x1[j])
            # print("TIFFONIBOBI",x1idx, 'vs',x1[i])
            xx1 = tf.math.maximum(x1[i], x1idx)
            # print('fake xx1',xx1)
            y1idx = []
            for j in idx:
                y1idx.append(y1[j])
            yy1 = tf.math.maximum(y1[i], y1idx)
            x2idx = []
            for j in idx:
                x2idx.append(x2[j])
            xx2 = tf.math.minimum(x2[i], x2idx)
            y2idx = []
            for j in idx:
                y2idx.append(y2[j])
            yy2 = tf.math.minimum(y2[i], y2idx)

            # print('all fakes',xx1,xx2,yy1,yy2)

            w = tf.math.maximum(0.0, xx2 - xx1 + 1)
            h = tf.math.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            aidx = []
            for j in range(tf.size(area)):
                if np.isin(j, idx):
                    aidx.append(area[j])

            if method == 'Min':
                o = inter / tf.math.minimum(area[i], aidx)
            else:
                o = inter / (area[i] + aidx - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]
        return pick

    @staticmethod
    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.

        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """

        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]
            # print("NAOWRNMAOSD",x1, i, idx, 'ALSO',x1[idx], 'vs', x1[i])

            xx1 = np.maximum(x1[i], x1[idx])
            # print('rela xx1',xx1)
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            # print('all reals',xx1,xx2,yy1,yy2)
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    @staticmethod
    def __tf_pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = tf.cast((total_boxes[:, 2] - total_boxes[:, 0] + 1), dtype=tf.int32)
        tmph = tf.cast((total_boxes[:, 3] - total_boxes[:, 1] + 1), dtype=tf.int32)
        numbox = total_boxes.shape[0]

        dx = tf.ones(numbox, dtype=tf.int32)
        dy = tf.ones(numbox, dtype=tf.int32)  # potential error/mistake
        edx = tf.cast(tmpw, dtype=tf.int32)
        edy = tf.cast(tmph, dtype=tf.int32)

        x = tf.cast(total_boxes[:, 0], dtype=tf.int32)
        y = tf.cast(total_boxes[:, 1], dtype=tf.int32)
        ex = tf.cast(total_boxes[:, 2], dtype=tf.int32)
        ey = tf.cast(total_boxes[:, 3], dtype=tf.int32)

        tmp = tf.where(ex > w)
        tf_edx_flat = tf.expand_dims(tf.reshape(edx, [-1]), 1)  # took out gather
        tf_edx_mod = tf.gather(-ex, tmp) + w + tf.gather(tmpw, tmp)
        tf_edx = tf.tensor_scatter_nd_update(tf_edx_flat, tmp, tf_edx_mod)
        ex = tf.clip_by_value(ex, 0, w)

        tmp = tf.where(ey > h)
        tf_edy_flat = tf.expand_dims(tf.reshape(edy, [-1]), 1)  # took out gather
        tf_edy_mod = tf.gather(-ey, tmp) + h + tf.gather(tmph, tmp)
        tf_edy = tf.tensor_scatter_nd_update(tf_edy_flat, tmp, tf_edy_mod)
        ey = tf.clip_by_value(ey, 0, h)

        tmp = tf.where(x < 1)
        tf_dx_flat = tf.expand_dims(tf.reshape(dx, [-1]), 1)  # took out gather
        tf_dx_mod = 2 - tf.gather(x, tmp)
        tf_dx = tf.tensor_scatter_nd_update(tf_dx_flat, tmp, tf_dx_mod)
        x = tf.clip_by_value(x, 1, tf.int32.max)

        tmp = tf.where(y < 1)
        tf_dy_flat = tf.expand_dims(tf.reshape(dy, [-1]), 1)  # took out gather
        tf_dy_mod = 2 - tf.gather(y, tmp)
        tf_dy = tf.tensor_scatter_nd_update(tf_dy_flat, tmp, tf_dy_mod)
        y = tf.clip_by_value(y, 1, tf.int32.max)

        return tf_dy, tf_edy, tf_dx, tf_edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w
        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __rerec(bbox):
        # convert bbox to square
        height = bbox[:, 3] - bbox[:, 1]
        width = bbox[:, 2] - bbox[:, 0]
        max_side_length = np.maximum(width, height)
        bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
        bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(boundingbox, reg):
        # calibrate bounding boxes
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox

    @staticmethod
    def __tf_bbreg(boundingbox, reg):
        # calibrate bounding boxes
        if reg.shape[1] == 1:
            reg = tf.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        # print(boundingbox.shape)

        indices = []

        for j in range(boundingbox.shape[1] - 1):
            for i in range(boundingbox.shape[0]):
                indices.append([i, j])

        t = tf.tensor_scatter_nd_update(boundingbox, indices, tf.transpose(tf.concat([b1, b2, b3, b4], axis=0)))

        return t

    @staticmethod
    def __apply_patch(img, patch, ground_truths_of_image):
        """
        Applies the patch to the image. Please use this Function only in new_detect_faces().

        :param img: The original image
        :param patch: The adversial patch
        :param ground_truths_of_image: Ground truth bounding boxes of the image
        :return: The original image but with the patch placed, dependet on where the ground truths are given
        """
        adv_img = copy.deepcopy(img)

        alpha = 0.5
        # i = 0  # used for pairing items in AS

        # draw detected face + plaster patch over source
        for bounding_box in ground_truths_of_image:  # ground truth loop

            '''
            if i >= len(ground_truths[image_names[image_nr]]):
                B = ground_truths[image_names[image_nr]][0]
            else:
                B = ground_truths[image_names[image_nr]][i]
            '''

            resize_value = alpha * math.sqrt(bounding_box[2] * bounding_box[3])
            resized_P = cv2.resize(patch, (round(resize_value), round(resize_value)))

            x_P = round(bounding_box[2] / 2)
            y_P = round(resize_value / 2)

            # draw patch over source image
            adv_img[
            y_P + bounding_box[1] - round(resized_P.shape[1] / 2):y_P + bounding_box[1] - round(
                resized_P.shape[1] / 2) + resized_P.shape[1],
            x_P + bounding_box[0] - round(resized_P.shape[0] / 2):x_P + bounding_box[0] - round(
                resized_P.shape[0] / 2) + resized_P.shape[0]] = resized_P

        print(adv_img.shape)
        return adv_img

    @staticmethod
    def __tf_apply_patch(img, patch, ground_truths_of_image):
        """
        Applies the patch to the image. Please use this Function only in new_detect_faces().

        :param img: The original image
        :param patch: The adversial patch
        :param ground_truths_of_image: Ground truth bounding boxes of the image
        :return: The original image but with the patch placed, dependet on where the ground truths are given
        """
        alpha = 0.5
        # i = 0  # used for pairing items in AS

        # draw detected face + plaster patch over source
        for bounding_box in ground_truths_of_image:  # ground truth loop

            '''
            if i >= len(ground_truths[image_names[image_nr]]):
                B = ground_truths[image_names[image_nr]][0]
            else:
                B = ground_truths[image_names[image_nr]][i]
            '''

            # print("CV2")
            # print(resized_P)
            # print("TENSORFLOW")
            # print(tf_resized_P)

            """
            # draw patch over source image
            adv_img[
            y_P + bounding_box[1] - round(resized_P.shape[1] / 2):y_P + bounding_box[1] - round(
                resized_P.shape[1] / 2) + resized_P.shape[1],
            x_P + bounding_box[0] - round(resized_P.shape[0] / 2):x_P + bounding_box[0] - round(
                resized_P.shape[0] / 2) + resized_P.shape[0]] = resized_P

            print(adv_img[y_P + bounding_box[1] - round(resized_P.shape[1] / 2):y_P + bounding_box[1] - round(resized_P.shape[1] / 2) + resized_P.shape[1], x_P + bounding_box[0] - round(resized_P.shape[0] / 2):x_P + bounding_box[0] - round(resized_P.shape[0] / 2) + resized_P.shape[0]].shape)
            print(resized_P.shape)
            print(tf_resized_P.shape)
            """
            resize_value = alpha * math.sqrt(bounding_box[2] * bounding_box[3])
            # resized_P = cv2.resize(patch, (round(resize_value), round(resize_value)))
            tf_resized_P = tf.image.resize(patch, (round(resize_value), round(resize_value)), method='lanczos5',
                                           antialias=True)  # tf image resize? LANCZOSinstead of AREA
            # tf_resized_P = tf.cast(resized_P, dtype=tf.float32)

            x_P = round(bounding_box[2] / 2)
            y_P = round(resize_value / 2)

            adv_img_rows = img.shape[0]
            adv_img_cols = img.shape[1]

            y_start = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2)  # bounding_box[0]

            x_start = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2)  # bounding_box[1]
            y_end = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2) + tf_resized_P.shape[
                1]  # x_start + tf_resized_p.shape[0]
            x_end = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2) + tf_resized_P.shape[
                0]  # y_start + tf_resized_p.shape[1]
            p_shape = tf.shape(tf_resized_P)

            # print("IMGSHAPE", adv_img.shape)
            # print("PSHAPE: ", p_shape)
            # print("BBX: ", x_start, " + PX: ", tf_resized_P.shape[0] , " = ", x_end)
            # print("BBY: ", y_start, " + PY: ", tf_resized_P.shape[1], " = ", y_end)

            true_y_start = y_start
            y_start = not_negative(y_start)
            true_x_start = x_start
            x_start = not_negative(x_start)
            true_y_end = y_end
            y_end = not_negative(y_end)
            true_x_end = x_end
            x_end = not_negative(x_end)

            if y_start != true_y_start:
                y_end += y_start - true_y_start
            elif y_end != true_y_end:
                y_start -= y_end - true_y_end
            elif x_start != true_x_start:
                x_end += x_start - true_x_start
            elif x_end != true_x_end:
                x_start -= x_end - true_x_end

            # print(y_start,y_end,x_start,x_end,true_y_start, true_y_end, true_x_start, true_x_end)
            # print(tf_resized_P.shape, img[true_y_start:y_end, x_start:x_end].shape)

            overlay = tf_resized_P - img[y_start:y_end, x_start:x_end]
            overlay_pad = tf.pad(overlay, [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]])
            # print(overlay_pad)
            img = img + overlay_pad

        # print(adv_img.shape)
        return img

    @staticmethod
    def __my_tf_apply_patch(img, patch, ground_truths_of_image):
        """
        Applies the patch to the image. Please use this Function only in new_detect_faces().

        :param img: The original image
        :param patch: The adversial patch
        :param ground_truths_of_image: Ground truth bounding boxes of the image, i.e. the marked faces
        :return: The original image but with the patch placed, dependet on where the ground truths are given
        """
        # adv_img = copy.deepcopy(img)
        # tf_adv_img = tf.cast(img, dtype=tf.float32)

        alpha = 0.5
        # TODO
        tf_adv_img = None

        # draw detected face + plaster patch over source

        for bounding_box in ground_truths_of_image:  # ground truth loop
            if tf_adv_img is None:
                tf_adv_img = img

            # TODO
            resize_value = round(alpha * math.sqrt(bounding_box[2] * bounding_box[3]))
            resize_value = tf.cast(resize_value, dtype=tf.float32)
            tf_resized_P = tf.image.resize(patch, (resize_value, resize_value), method='lanczos5', antialias=True)

            x_P = tf.math.round(bounding_box[2] / 2.0)
            y_P = tf.math.round(resize_value / 2.0)

            adv_img_rows = img.shape[0]
            adv_img_cols = img.shape[1]

            # Finding the indices where to put the patch
            y_start = tf.cast(y_P + bounding_box[1] - round(tf_resized_P.shape[0] / 2.0),
                              dtype=tf.int32)  # bounding_box[0]
            x_start = tf.cast(x_P + bounding_box[0] - round(tf_resized_P.shape[1] / 2.0),
                              dtype=tf.int32)  # bounding_box[1]

            y_end = tf.cast(y_P + bounding_box[1] - round(tf_resized_P.shape[0] / 2.0) + tf_resized_P.shape[
                0], dtype=tf.int32)
            x_end = tf.cast(x_P + bounding_box[0] - round(tf_resized_P.shape[1] / 2.0) + tf_resized_P.shape[
                1], dtype=tf.int32)

            '''If the bounding box is outside the image'''
            if tf.math.less(x_start, 0):
                x_end -= x_start
                x_start = tf.cast(0, dtype=tf.int32)
            if tf.math.less(y_start, 0):
                y_end -= y_start
                y_start = tf.cast(0, dtype=tf.int32)

            if tf.math.greater(x_end, img.shape[1]):
                x_start -= x_end - img.shape[1]
                x_end = tf.cast(img.shape[1], dtype=tf.int32)
            if tf.math.greater(y_end, img.shape[0]):
                y_start -= y_end - img.shape[0]
                y_end = tf.cast(img.shape[0], dtype=tf.int32)

            tf_overlay = tf_resized_P - img[y_start:y_end, x_start:x_end]
            tf_overlay_pad = tf.pad(tf_overlay,
                                    [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]])

            tf_adv_img = tf_adv_img + tf_overlay_pad

        if tf_adv_img == None:
            print("ATTENTION no GroundTruth *************************************************")
            print(ground_truths_of_image)
            tf_adv_img = img

        return tf_adv_img

    def detect_faces(self, img) -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        height, width, _ = img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self._min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__stage1, self.__stage2, self.__stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in range(len(stages)):
            result = stages[stage](img, result[0], result[1])
            """
            if stage is 1: #if it's self.__stage2 -> dependency on immutability of stages list
              old_result = result
              result = [None, None] #for re-entry into stage3
              result[0] = tf.cast(old_result[0], dtype=tf.float32)
              result[1] = old_result[1]
            """
        [total_boxes, points] = result

        bounding_boxes = []

        for bounding_box, keypoints in zip(total_boxes, points.T):
            x = max(0, int(bounding_box[0]))
            y = max(0, int(bounding_box[1]))
            width = int(bounding_box[2] - x)
            height = int(bounding_box[3] - y)
            bounding_boxes.append({
                'box': [x, y, width, height],
                'confidence': bounding_box[-1],
                'keypoints': {
                    'left_eye': (int(keypoints[0]), int(keypoints[5])),
                    'right_eye': (int(keypoints[1]), int(keypoints[6])),
                    'nose': (int(keypoints[2]), int(keypoints[7])),
                    'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                    'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                }
            })

        return bounding_boxes

    def new_detect_faces(self, img, patch, ground_truths_of_image, amplification_factor: int) -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """

        # self.patch = patch
        self.patch = tf.Variable(patch, dtype=tf.float32)
        # oldie = tf.Variable(self.patch)
        self.ground_truths_of_image = ground_truths_of_image

        if ground_truths_of_image[0].count(0) == len(ground_truths_of_image[
                                                         0]):  # TODO checks if the first ground truth is a list containing only 0 [REFERING TO 0--Parade/0_Parade_Parade_0_452.jpg]
            tf_adv_img = tf.cast(img, dtype=tf.float32)
        else:
            tf_adv_img = self.__my_tf_apply_patch(img, self.patch, self.ground_truths_of_image)
        # print(type(tf_adv_img), tf_adv_img.shape, tf_adv_img.dtype)

        self.adv_img = tf_adv_img.numpy()
        if self.adv_img is None or not hasattr(self.adv_img, "shape"):
            raise InvalidImage("Image not valid.")

        height, width, _ = self.adv_img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self._min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__new_stage1, self.__new_stage2, self.__new_stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in range(len(stages)):
            result = stages[stage](self.adv_img, result[0], result[1], amplification_factor)
            if stage is 1:  # if it's self.__stage2 -> dependency on immutability of stages list
                old_result = result
                result = [None, None]  # for re-entry into stage3
                result[0] = tf.cast(old_result[0], dtype=tf.float32)
                result[1] = old_result[1]

        [total_boxes, points] = result

        bounding_boxes = []

        for bounding_box, keypoints in zip(total_boxes, points.T):
            x = max(0, int(bounding_box[0]))
            y = max(0, int(bounding_box[1]))
            width = int(bounding_box[2] - x)
            height = int(bounding_box[3] - y)
            bounding_boxes.append({
                'box': [x, y, width, height],
                'confidence': bounding_box[-1],
                'keypoints': {
                    'left_eye': (int(keypoints[0]), int(keypoints[5])),
                    'right_eye': (int(keypoints[1]), int(keypoints[6])),
                    'nose': (int(keypoints[2]), int(keypoints[7])),
                    'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                    'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                }
            })
        # newie = tf.Variable(self.patch)
        # self.adv_img = self.__my_tf_apply_patch(img, self.patch, ground_truths_of_image).numpy()
        # print(oldie - newie)
        # print(self.patch.dtype)
        return (bounding_boxes, self.adv_img, self.patch)

    def __stage1(self, image, scales: list, stage_status: StageStatus):
        """
        First stage of the MTCNN.
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        total_boxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image, scale)

            img_x = np.expand_dims(scaled_image, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            out = self._pnet(img_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
                                                    out0[0, :, :, :].copy(), scale, self._steps_threshold[0])

            # inter-scale nms
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numboxes = total_boxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = self.__rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return total_boxes, status

    def __stage2(self, img, total_boxes, stage_status: StageStatus):
        """
        Second stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """

        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, stage_status

        # second stage
        tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

        for k in range(0, num_boxes):
            tmp = np.zeros((int(stage_status.tmph[k]), int(stage_status.tmpw[k]), 3))

            tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = \
                img[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

            else:
                return np.empty(shape=(0,)), stage_status

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self._rnet(tempimg1)

        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self._steps_threshold[1])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = self.__nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.__rerec(total_boxes.copy())

        return total_boxes, stage_status


    def __stage3(self, img, total_boxes, stage_status: StageStatus):

        with self.tape as tape:

            img = self.__my_tf_apply_patch(img, self.patch, self.ground_truths_of_image)
            """
            Third stage of the MTCNN.
  
            :param img:
            :param total_boxes:
            :param stage_status:
            :return:
            """
            """ #backup
            num_boxes = total_boxes.shape[0]
            if num_boxes == 0:
                return total_boxes, np.empty(shape=(0,))
  
            total_boxes = np.fix(total_boxes).astype(np.int32)
  
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)
  
            tempimg = np.zeros((48, 48, 3, num_boxes))
  
            for k in range(0, num_boxes):
  
                tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))
  
                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]
  
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty(shape=(0,)), np.empty(shape=(0,))
  
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            tempimg2 = tf.Variable(tempimg1, dtype=tf.float32)
            """

            num_boxes = total_boxes.shape[0]
            if num_boxes == 0:
                return total_boxes, np.empty(shape=(0,))  # it's just face points - can be left on np

            total_boxes = np.fix(total_boxes).astype(np.int32)

            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

            tempimg = np.zeros((48, 48, 3, num_boxes))

            """
            tf_total_boxes = tf.experimental.numpy.fix(total_boxes)
            tf_total_boxes = tf.cast(tf_total_boxes, dtype=tf.int32)
  
            status = StageStatus(self.__tf_pad(tf_total_boxes, stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)
  
            #        return tf_dy, tf_edy, tf_dx, tf_edx, y, ey, x, ex, tmpw, tmph
  
            tempimg = tf.zeros((48, 48, 3, num_boxes))
            tf_tempimg = tf.zeros((48, 48, 3, num_boxes))
  
            #print(tempimg, tf_tempimg)
  
            for k in range(0, num_boxes):
  
                tf_tmp = tf.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))
                print(tf_tmp)
                #tmp = tf.where(y < 1) # used to be a condition
                tf_tmp_flat = tf.expand_dims(tf.reshape(tf_tmp,[-1]), 1) #took out gather
                tf_i_1 = tf.range(status.dy[k] - 1, status.edy[k])
                tf_i_2 = tf.range(status.dx[k] - 1, status.edx[k])
                tf_i_3 = tf.range(0, tf_tmp.shape[2])
                tf_index = [tf_i_1, tf_i_2, tf_i_3]
  
                tf_j_1 = tf.range(status.y[k] - 1, status.ey[k])
                tf_j_2 = tf.range(status.x[k] - 1, status.ex[k])
                tf_j_3 = tf.range(0, img.shape[2])
                tf_mod_index = [tf_j_1, tf_j_2, tf_j_3]
                print(img.shape)
                #, tf_mod_index.shape)
                tf_tmp_mod = tf.gather(img, tf_mod_index)
  
                tf_tmp = tf.tensor_scatter_nd_update(tf_tmp_flat, tf_index, tf_tmp_mod)
                print(tf_tmp)
                #y = tf.clip_by_value(y, 1, tf.int32.max)
                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]
  
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty(shape=(0,)), np.empty(shape=(0,))
            """

            for k in range(0, num_boxes):

                tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    print('oh no')
                    return tf.empty(shape=(0,)), np.empty(shape=(0,))

            tf_tempimg = tf.cast(tempimg,
                                 dtype=tf.float32)  # similar to attack - first resize in np then finish operations in tf
            tf_tempimg = (tf_tempimg - 127.5) * 0.0078125
            tf_tempimg1 = tf.transpose(tf_tempimg, (3, 1, 0, 2))
            tf_tempimg2 = tf.Variable(tf_tempimg1, dtype=tf.float32)

            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            tempimg2 = tf.Variable(tempimg1, dtype=tf.float32)

            tf_total_boxes = tf.cast(total_boxes, dtype=tf.float32)

            # print(tape)
            # maybe also watch the other variables?
            tape.watch(tf_tempimg2)

            tf_outie = tf.transpose(self._onet(tf_tempimg2)[2])[1, :]
            # print([var.name for var in self._onet.trainable_variables])
            tf_out = self._onet(tf_tempimg2)  # changed from 1 to 2
            # print([var.name for var in tape.watched_variables()])

            """
            loss = 0
            for box in total_boxes:
              loss += tf.math.log(tf.Variable(out[2][1,:]))
            """
            tf_out2 = tf.transpose(tf_out[2])
            tf_out0 = tf.transpose(tf_out[0])
            tf_score = tf_out2[1, :]

            tf_ipass = tf.transpose(tf.where(tf_score > self._steps_threshold[2]))
            tf_ipass = tf.cast(tf_ipass, dtype=tf.float32)
            a = []
            # tf_total_boxes[tf_ipass[0], 0:4]
            for e in tf_ipass[0]:
                a.append(tf_total_boxes[tf.cast(e, dtype=tf.int64), 0:4])
            b = []
            for e in tf_ipass[0]:
                b.append(tf_score[tf.cast(e, dtype=tf.int64)])
            c = tf.expand_dims(b, 1)

            tf_total_boxes = tf.concat([a, c], axis=1)

            tf_pre_mv = []
            for i in range(len(tf_out0)):
                m = []
                for j in range(len(tf_out0[i])):
                    if np.isin(j, tf_ipass[0]):
                        m.append(tf_out0[i][j])
                tf_pre_mv.append(m)

            tf_mv = tf.zeros([0])
            for m in tf_pre_mv:
                tf_mv = tf.concat([tf_mv, m], 0)
            tf_mv = tf.reshape(tf_mv, [4, -1])

            """ custom TensorFlow code adaptation end """
            out = self._onet(tempimg1)
            out0 = np.transpose(out[0])

            out2 = np.transpose(out[2])
            score = out2[1, :]  # relevant

            out1 = np.transpose(out[1])

            points = out1  # points influences only points

            ipass = np.where(score > self._steps_threshold[2])  # relevant

            points = points[:, ipass[0]]  # not relevant

            total_boxes = np.hstack(
                [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])  # relevant
            mv = out0[:, ipass[0]]  # relevant

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1  # not relevant
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1  # not relevant

            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0],
                                                                           (5, 1)) - 1  # not relevant
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1],
                                                                             (5, 1)) - 1  # not relevant

            tf_points = points
            """
            # Comment for Tensorflow, uncomment for Regular Numpy
            if total_boxes.shape[0] > 0:
                total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
                pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]
            """

            """ tensorflow adaptation - original code commented out to save resources"""
            if tf_total_boxes.shape[0] > 0:
                tf_total_boxes = self.__tf_bbreg(tf_total_boxes, tf.transpose(
                    tf_mv))  # didn't copy tf_total_boxes, let's see how it goes
                tf_pick = self.__tf_nms(tf_total_boxes, 0.7, 'Min')  # also didn't copy
                tf_pick = tf_pick.astype(
                    'int32')  # initially is type int16 which tensorflow doesn't support ... for some reason
                tf_total_boxes = tf.gather(tf_total_boxes, tf_pick)
                tf_points = tf_points[:,
                            tf_pick]  # have to comment out because during comparison to above it breaks stuff

            indices = []
            for score in tf_total_boxes[:, 4]:
                for i in range(len(tf_outie)):
                    if score == tf_outie[i]:
                        indices.append(i)

            tf_outie = tf.gather(tf.transpose(self._onet(tf_tempimg2)[2])[1, :], indices)

            """
            # Comment for Tensorflow, uncomment for Regular Numpy
            return total_boxes, points
            """
            # print([var.name for var in tape.watched_variables()])
        # print(tf_score)
        # print(tf_total_boxes[:,4], tf.math.reduce_max(tf_score))

        # print(tf_tempimg2.shape)

        # print(indices, tf.gather(tf_score, indices))

        # print(a.shape, b.shape, type(a), type(b), a, b, a is b)

        gradient = tape.gradient(tf_outie, self.patch)

        # print(gradient, self.patch.shape)
        # gradient = tape.gradient(tf_total_boxes[:,4], tempimg2)
        # print(gradient.shape, tf_tempimg2.shape)
        # print(gradient)
        return tf_total_boxes, tf_points

    def __new_stage1(self, image, scales: list, stage_status: StageStatus, amplification_factor: int):
        """
        First stage of the MTCNN.
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        total_boxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image, scale)

            img_x = np.expand_dims(scaled_image, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            out = self._pnet(img_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
                                                    out0[0, :, :, :].copy(), scale, self._steps_threshold[0])

            # inter-scale nms
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numboxes = total_boxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = self.__rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return total_boxes, status

    def __new_stage2(self, img, total_boxes, stage_status: StageStatus, amplification_factor: int):
        """
        Second stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """

        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, stage_status

        # second stage
        tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

        for k in range(0, num_boxes):
            tmp = np.zeros((int(stage_status.tmph[k]), int(stage_status.tmpw[k]), 3))

            tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = \
                img[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

            else:
                return np.empty(shape=(0,)), stage_status

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self._rnet(tempimg1)

        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self._steps_threshold[1])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = self.__nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.__rerec(total_boxes.copy())

        return total_boxes, stage_status

    def __new_stage3(self, img, total_boxes, stage_status: StageStatus, amplification_factor: int):
        # tf_img = tf.cast(img, dtype=tf.float32)
        # self.patch = tf.Variable(self.patch, dtype=tf.float32)

        with self.tape as tape:
            tape.watch(self.patch)

            if self.ground_truths_of_image[0].count(0) == len(self.ground_truths_of_image[
                                                                  0]):  # TODO checks if the first ground truth is a list containing only 0 [REFERING TO 0--Parade/0_Parade_Parade_0_452.jpg]
                tf_img = tf.cast(img, dtype=tf.float32)
            else:
                tf_img = self.__my_tf_apply_patch(img, self.patch, self.ground_truths_of_image)

            """
            Third stage of the MTCNN.
    
            :param img:
            :param total_boxes:
            :param stage_status:
            :return:
            """
            """ #backup
            num_boxes = total_boxes.shape[0]
            if num_boxes == 0:
                return total_boxes, np.empty(shape=(0,))
    
            total_boxes = np.fix(total_boxes).astype(np.int32)
    
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)
    
            tempimg = np.zeros((48, 48, 3, num_boxes))
    
            for k in range(0, num_boxes):
    
                tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))
    
                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]
    
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty(shape=(0,)), np.empty(shape=(0,))
    
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            tempimg2 = tf.Variable(tempimg1, dtype=tf.float32)
            """
            # print("IT WORKS IN STAGE 3")
            # print(self.patch)
            # print("______________________________________")

            new_total_boxes = []

            """ IoU selection 
            for box in total_boxes:
              tmp_box = box.numpy()
              #print('ORGI', box)
              tmp_box = np.delete(tmp_box, 4)
              tmp_box[2] = tmp_box[2] - tmp_box[0]
              tmp_box[3] = tmp_box[3] - tmp_box[1]
              for truth_box in self.ground_truths_of_image:
                #print(tmp_box,truth_box)
                if IoU(box, truth_box):
                  new_total_boxes.append(box)
            #print(total_boxes, new_total_boxes)
            total_boxes = new_total_boxes
            """

            num_boxes = len(total_boxes)
            tf_total_boxes = tf.cast(total_boxes, dtype=tf.float32)  # maybe requires pre-setting

            if num_boxes == 0:
                print('no faces detected')
                return tf_total_boxes, np.empty(shape=(0,))  # it's just face points - can be left on np

            total_boxes = np.fix(total_boxes).astype(np.int32)
            # print(total_boxes)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

            tempimg = np.zeros((48, 48, 3, num_boxes))

            tf_tempimg = tf.zeros((48, 48, 3, num_boxes))

            tf_total_boxes = tf.experimental.numpy.fix(tf_total_boxes)
            tf_total_boxes = tf.cast(tf_total_boxes, dtype=tf.int32)

            tf_status = StageStatus(self.__tf_pad(tf_total_boxes, stage_status.width, stage_status.height),
                                    width=stage_status.width, height=stage_status.height)
            #        return tf_dy, tf_edy, tf_dx, tf_edx, y, ey, x, ex, tmpw, tmph

            tf_tempimgs = []

            for k in range(0, num_boxes):
                tf_tmp = tf_img[tf_status.y[k] - 1:tf_status.ey[k], tf_status.x[k] - 1:tf_status.ex[k], :]
                if tf_tmp.shape[0] > 0 and tf_tmp.shape[1] > 0 or tf_tmp.shape[0] == 0 and tf_tmp.shape[1] == 0:
                    # tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48), method = 'area')) # - The correct one
                    # tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48))) #method = 'bilinear' is ehhhh sometimes very close, 'lanczos5' is one significant digit off,
                    tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48), method='lanczos5', antialias=True))

                else:
                    print('tf_tmp is misshapen')
                    return tf.empty(shape=(0,))

            tf_tempimg = tf.stack(tf_tempimgs, 3)

            for k in range(0, num_boxes):

                tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

                tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                    img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    self.tf_tmp = tmp
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)

                else:
                    print('oh no')
                    return tf.empty(shape=(0,)), np.empty(shape=(0,))

            # print('real',tempimg)
            # print('fake',tf_tempimg)
            tf_tempimg = (tf_tempimg - 127.5) * 0.0078125
            tf_tempimg2 = tf.transpose(tf_tempimg, (3, 1, 0, 2))
            # tf_tempimg2 = tf.Variable(tf_tempimg1, dtype=tf.float32)

            # tf_tempimg2 = tf.transpose((tf_tempimg - 127.5) * 0.0078125, (3, 1, 0, 2))

            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            tempimg2 = tf.Variable(tempimg1, dtype=tf.float32)

            tf_total_boxes = tf.cast(total_boxes, dtype=tf.float32)

            # print(tape)
            # maybe also watch the other variables?
            # tape.watch(tf_tempimg2)

            # tf_outie = tf.transpose(self._onet(tf_tempimg2)[2])[1, :]
            # print([var.name for var in self._onet.trainable_variables])
            out1 = np.transpose(self._onet(tf_tempimg2)[1])
            tf_out = self._onet(tf_tempimg2)  # changed from 1 to 2
            # print([var.name for var in tape.watched_variables()])

            """
            loss = 0
            for box in total_boxes:
              loss += tf.math.log(tf.Variable(out[2][1,:]))
            """
            tf_out2 = tf.transpose(tf_out[2])
            tf_out0 = tf.transpose(tf_out[0])
            tf_score = tf_out2[1, :]

            tf_ipass = tf.transpose(tf.where(tf_score > self._steps_threshold[2]))
            tf_ipass = tf.cast(tf_ipass, dtype=tf.float32)

            a = []
            # tf_total_boxes[tf_ipass[0], 0:4]
            for e in tf_ipass[0]:
                a.append(tf_total_boxes[tf.cast(e, dtype=tf.int64), 0:4])
            b = []
            for e in tf_ipass[0]:
                b.append(tf_score[tf.cast(e, dtype=tf.int64)])
            c = tf.expand_dims(b, -1)
            try:
                tf_total_boxes = tf.concat([a, c], axis=1)
            except:
                print('no confidence!')
                tf_total_boxes = tf.experimental.numpy.empty(0)
                return [], out1

                # check this out
                # print(tf_total_boxes)
                # return [], out1 #avoids ConcatOp : Expected concatenating dimensions in the range [-1, 1), but got 1 [Op:ConcatV2] - error, when axis = 1

            tf_pre_mv = []
            for i in range(len(tf_out0)):
                m = []
                for j in range(len(tf_out0[i])):
                    if np.isin(j, tf_ipass[0]):
                        m.append(tf_out0[i][j])
                tf_pre_mv.append(m)

            tf_mv = tf.zeros([0])
            for m in tf_pre_mv:
                tf_mv = tf.concat([tf_mv, m], 0)
            tf_mv = tf.reshape(tf_mv, [4, -1])

            """ custom TensorFlow code adaptation end """
            out = self._onet(tempimg1)
            out0 = np.transpose(out[0])

            out2 = np.transpose(out[2])
            score = out2[1, :]  # relevant

            out1 = np.transpose(out[1])

            points = out1  # points influences only points

            ipass = np.where(score > self._steps_threshold[2])  # relevant

            points = points[:, ipass[0]]  # not relevant

            total_boxes = np.hstack(
                [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])  # relevant

            mv = out0[:, ipass[0]]  # relevant

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1  # not relevant
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1  # not relevant

            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0],
                                                                           (5, 1)) - 1  # not relevant
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1],
                                                                             (5, 1)) - 1  # not relevant

            # Comment for Tensorflow, uncomment for Regular Numpy
            if total_boxes.shape[0] > 0:
                total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
                pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]

            tf_points = points

            """ tensorflow adaptation - original code commented out to save resources"""
            if tf_total_boxes.shape[0] > 0:
                tf_total_boxes = self.__tf_bbreg(tf_total_boxes, tf.transpose(
                    tf_mv))  # didn't copy tf_total_boxes, let's see how it goes
                # tf_pick = self.__tf_nms(tf_total_boxes, 0.7, 'Min')  # also didn't copy
                tf_pick = tf.image.non_max_suppression(tf_total_boxes[:, 0:4], tf_total_boxes[:, 4],
                                                       max_output_size=100, iou_threshold=0.7)  # also didn't copy
                tf_pick = tf.cast(tf_pick,
                                  dtype=tf.int32)  # initially is type int16 which tensorflow doesn't support ... for some reason
                # tf_pick = tf_pick.astype('int32')  # initially is type int16 which tensorflow doesn't support ... for some reason
                tf_total_boxes = tf.gather(tf_total_boxes, tf_pick)
                # tf_points = tf_points[:, tf_pick]  # have to comment out because during comparison to above it breaks stuff  ----- SOMETIMES throws errors: IndexError: index 3 is out of bounds for axis 1 with size 3

            # print(max(abs(np.round(np.amax(tf.transpose(self._onet(tf_tempimg2)[2])[1, :] - score), 2)), abs(np.round(np.amin(tf.transpose(self._onet(tf_tempimg2)[2])[1, :] - score), 2))))

            indices = []
            tf_outie = tf.transpose(tf_out[2])[1, :]
            tf_loss = 0

            if tf_total_boxes.shape[0] > 0:
                for score in tf_total_boxes[:, 4]:
                    for i in range(len(tf_outie)):
                        if score == tf_outie[i]:
                            indices.append(i)
            # print(tf.gather(tf.transpose(self._onet(tf_tempimg2)[2])[1, :], indices).shape[0])

            if len(indices):
                tf_loss = tf.math.negative(tf.divide(
                    tf.math.reduce_sum(tf.math.log(tf.gather(tf.transpose(self._onet(tf_tempimg2)[2])[1, :], indices))),
                    len(indices)))

            # tf_outie = tf.gather(tf.transpose(self._onet(tf_tempimg2)[2])[1, :], indices)
            """
            # Comment for Tensorflow, uncomment for Regular Numpy
            return total_boxes, points
            """

            # print([var.name for var in tape.watched_variables()])
            # print(tf_score)
            # print(tf_total_boxes[:,4], tf.math.reduce_max(tf_score))

            # print(tf_tempimg2.shape)

            # print(indices, tf.gather(tf_score, indices))

            # print(a.shape, b.shape, type(a), type(b), a, b, a is b)
        # print(total_boxes, tf_total_boxes)
        # loss = tf.divide(tf.math.reduce_sum(tf.math.log(tf_outie)), tf_outie.shape[0])

        if tf_loss:
            gradient = tape.gradient(tf_loss,
                                     self.patch)  # self.patch -> tf_img -> MANY tf_tmp -> tf_tempimg -> / DOESN'T KNOW HOW TO DO GRADIENT OF 'AREA' RESIZE / -> tf_tempimg2 -> tf_out

            if gradient is not None:
                self.patch.assign(tf.clip_by_value((self.patch + gradient * amplification_factor), clip_value_min=0,
                                                   clip_value_max=255))

        # self.adv_img = tf.cast(tf_tempimg1[0,:,:,:], dtype=tf.float32).numpy()
        # print(type(self.adv_img), self.adv_img.shape)

        # print(self.adv_img)
        # print(gradient)
        """
        if True:
          print('wizard')
          #self.patch.assign(tf.keras.preprocessing.image.img_to_array(tf.keras.utils.load_img("wizards.jpg", target_size = (128, 128))))
        else:
          print('lesser')
        """

        # storePatch = self.patch.numpy()
        # cv2.imwrite('Face_Control/'+str(self.i)+'_patch_.jpg', storePatch*10000000) #[:, :, [2, 1, 0]])
        # self.i += 1

        # print(gradient.shape, self.patch.shape)
        # gradient = tape.gradient(tf_total_boxes[:,4], tempimg2)
        # print(gradient.shape, tf_tempimg2.shape)
        # print(gradient)
        return tf_total_boxes, tf_points