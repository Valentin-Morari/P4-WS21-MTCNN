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

import copy #used for creating deep copies where changes won't affet the original files

import cv2 #image processing framework
import numpy as np #extend math library in python
import pkg_resources #weights loading
import tensorflow as tf #machine learning framework

import math

from mtcnn.exceptions import InvalidImage
from mtcnn.network.factory import NetworkFactory

__author__ = "Iván de Paz Centeno" #Adapted for P4 course by Valentin Morari and Eduard Bersch

def IoU(boxA, boxB):
    """
    This code was taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    It does Intersection over Union with the two bounding boxes.

    :param boxA: list
        representing a bounding box
    :param boxB: list
        representing a bounding box
    :return: float
        representing the result of the Intersection over Union from boxA and boxB
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    lambda_IoU = 0.6  # To match paper's concrete parameters - adjust for allowing smaller faces to be detected, compared to the original

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # MTCNN doesn't give the end points in [2] and [3] but the distance from the start point to the end point
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth bounding boxes
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union result 
    return iou > lambda_IoU

def not_negative(x): 
  """
  Simple function to ensure the value is either positive or zero
  
  :param x: number
    variable that we want to ensure is non-negative
  :return: x
    non-negative value of x
  """
  
  if x<0:
    return 0
  else:
    return x

class StageStatus(object):
    """
    Keeps information between MTCNN stages
    
    :param object: base class
      used exclusively for compatibility with python 2.x
    """

    def __init__(self, pad_result: tuple = None, width=0, height=0):
        """
        Performs initialization of a StageStatus instance
        
        :param pad_result: tuple
          bounded boxes padded to a square shape; defaults to None
        :param width: int
          total width of the image; defaults to 0
        :param height: int
          total height of the image; defaults to 0
        """
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        """
        Updates variables of an existing StageStatus instance
        
        :param pad_result: tuple
          bounded boxes padded to a square shape
        """
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class MTCNN(object):
    """
    Allows to perform MTCNN detection ->
        a) Detection of faces (with the confidence probability) - Only faces are used for the P4 project
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    
    :param object: base class
      used exclusively for compatibility with python 2.x
    """

    def __init__(self, weights_file: str = None, min_face_size: int = 20, steps_threshold: list = None,
                 scale_factor: float = 0.709):
        """
        Initializes the MTCNN.
        
        :param weights_file: string
          file uri with the weights of the P, R and O networks from MTCNN. When not specified,
        it loads the ones bundled with the package.
        :param min_face_size: int
          minimum size of the face to detect. Defaults to 20
        :param steps_threshold: list
          step threshold values
        :param scale_factor: float
          scale factor. Defaults to 0.709
        """
        
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        if weights_file is None:
            weights_file = pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy')
        
        self.tape = tf.GradientTape() # Used to compute gradients
        
        self._min_face_size = min_face_size 
        self._steps_threshold = steps_threshold
        self._scale_factor = scale_factor
        self.amplification = 1000000 # default value, can be specfiied when calling new_detect_faces
        
        self.resize_method = 'lanczos5' # tensorflow resize method used during adversarial patch training
        # NOTE: it is not AREA, as it was originally, because the AREA resize method is the only one without an automatic gradient. 
        # Until it is added, we use lanczos5
        self.antialias = True # antialiasing to use in tensorflow image resizing. default behavior: on
        
        self._pnet, self._rnet, self._onet = NetworkFactory().build_P_R_O_nets_from_file(weights_file)
    
    #following functions are left unchanged

    @property
    def min_face_size(self):
        """
        Returns minimum face size parameter of MTCNN instance.
        """ 
        return self._min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=20):
        """
        Attempts to set minimum face size parameter of MTCNN instance. Reverts to a default of 20 if unsuccessful.
        
        :param mfc: int
          new minimum face size value. Defaults to 20
        """
        try:
            self._min_face_size = int(mfc)
        except ValueError:
            self._min_face_size = 20

    def __compute_scale_pyramid(self, m, min_layer):
        """
        Computes scale pyramid to be fed into PNET during stage 1.
        
        :param m: float
          maximum scale of the pyramid
        :param min_layer: float 
          minimum face size that the P network can detect
        """
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
        
        :param image: nd-array
          image data to be rescaled
        :param scale: float
          scale used for image rescaling
        :return: nd-array
          rescaled image that is normalized to the interval (-1, 1)
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels to the interval (-1, 1)
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(imap, reg, scale, t): #left unmodified
        """
        Generates bounding box based on PNET results 
        
        :param imap: nd-array
          mapped indexes of PNET face detection result
        :param reg: nd-array
          bounding box regression vector 
        :param scale: float
          scale to which the original image was resized
        :paramt t: float
          step threshold to surpass
          
        :return 
          boundingbox: nd-array
            generated boundingbox for detected faces
          reg: nd-array 
            new bounding box regression vector
        """
        
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
    def __tf_nms(boxes, threshold, method): #converted to tensorflow
        """
        Non Maximum Suppression.

        :param boxes: tensor 
          bounding boxes to perform nms on
        :param threshold: float
          number from 0 to 1, to serve as minimum value to reach by NMS
        :param method: str
          NMS method to apply. Available values ('Min', 'Union')
        :return: nd-array 
          index list for which bounding boxes to process on after nms is done
        """
        
        if tf.size(boxes) == 0:           #start here
            print("tf_nms encountered an error, outputs zero-tensor")
            return tf.reshape(tf.convert_to_tensor(()), (0,3))

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
            
            x1idx = []
            for j in idx:
              x1idx.append(x1[j])
            xx1 = tf.math.maximum(x1[i], x1idx)
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
    def __nms(boxes, threshold, method): #kept unchanged to support functions not converted to tensorflow that need it
        """
        Non Maximum Suppression.

        :param boxes: nd-array 
          bounding boxes to perform nms on
        :param threshold: float
          number from 0 to 1, to serve as minimum value to reach by NMS
        :param method: str
          NMS method to apply. Available values ('Min', 'Union')
        :return: nd-array 
          which bounding boxes remain after nms
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
            
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

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
    def __tf_pad(total_boxes, w, h): #converted to tensorflow
        """
        Compute the padding coordinates (pad the bounding boxes to squares)
        
        :param total_boxes: tensor
          all bounding boxes taken over from stage 2
        :param w: int
          width of image
        :param h: int
          height of image
        :return: 
          padding data to be kept track of by StageStatus
        """
        
        tmpw = tf.cast((total_boxes[:, 2] - total_boxes[:, 0] + 1), dtype=tf.int32)
        tmph = tf.cast((total_boxes[:, 3] - total_boxes[:, 1] + 1), dtype=tf.int32)
        numbox = total_boxes.shape[0]

        dx = tf.ones(numbox, dtype=tf.int32)
        dy = tf.ones(numbox, dtype=tf.int32) #potential error/mistake -> works in our case for P4 (patch is a square) but otherwise should be adjusted
        edx = tf.cast(tmpw, dtype=tf.int32)
        edy = tf.cast(tmph,dtype=tf.int32)

        x = tf.cast(total_boxes[:, 0], dtype=tf.int32)
        y = tf.cast(total_boxes[:, 1], dtype=tf.int32)
        ex = tf.cast(total_boxes[:, 2], dtype=tf.int32)
        ey = tf.cast(total_boxes[:, 3], dtype=tf.int32)
        
        tmp = tf.where(ex > w)
        tf_edx_flat = tf.expand_dims(tf.reshape(edx,[-1]), 1) 
        tf_edx_mod = tf.gather(-ex, tmp) + w + tf.gather(tmpw, tmp)
        tf_edx = tf.tensor_scatter_nd_update(tf_edx_flat, tmp, tf_edx_mod)
        ex = tf.clip_by_value(ex, 0, w)
        
        tmp = tf.where(ey > h)
        tf_edy_flat = tf.expand_dims(tf.reshape(edy,[-1]), 1) 
        tf_edy_mod = tf.gather(-ey, tmp) + h + tf.gather(tmph, tmp)
        tf_edy = tf.tensor_scatter_nd_update(tf_edy_flat, tmp, tf_edy_mod)
        ey = tf.clip_by_value(ey, 0, h)

        tmp = tf.where(x < 1)
        tf_dx_flat = tf.expand_dims(tf.reshape(dx,[-1]), 1) 
        tf_dx_mod = 2 - tf.gather(x, tmp)
        tf_dx = tf.tensor_scatter_nd_update(tf_dx_flat, tmp, tf_dx_mod)
        x = tf.clip_by_value(x, 1, tf.int32.max)
        
        tmp = tf.where(y < 1)
        tf_dy_flat = tf.expand_dims(tf.reshape(dy,[-1]), 1)
        tf_dy_mod = 2 - tf.gather(y, tmp)
        tf_dy = tf.tensor_scatter_nd_update(tf_dy_flat, tmp, tf_dy_mod)
        y = tf.clip_by_value(y, 1, tf.int32.max)
        
        return tf_dy, tf_edy, tf_dx, tf_edx, y, ey, x, ex, tmpw, tmph


    @staticmethod
    def __pad(total_boxes, w, h): #not converted to tensorflow, left to support functions that have not yet been ported and need it
        """
        Compute the padding coordinates (pad the bounding boxes to squares)
        
        :param total_boxes: nd-array
          all bounding boxes taken over from stage 2
        :param w: int
          width of the image
        :param h: int
          height of the image
        :return: 
          padding data to be kept track of by StageStatus
        """
        
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
    def __rerec(bbox): # left unchanged
        """
        Convert bbox to a square
        
        :param bbox: nd-array
          bounding box
        :return: nd-array
          bounding box
        """
        
        height = bbox[:, 3] - bbox[:, 1]
        width = bbox[:, 2] - bbox[:, 0]
        max_side_length = np.maximum(width, height)
        bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
        bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(boundingbox, reg): #unconverted, as stage2 needs it, and no plans to convert it to tensorflow have yet been made
        """
        Calibrate bounding boxes through bounding box regression
        
        :param boundingbox: nd-array
          all bounding boxes obtained so far as part of the face detection process
        :param reg: nd-array
          regression vector
        :return: nd-array
          calibrated bounding boxes
        """
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
    def __tf_bbreg(boundingbox, reg): #used in stage 3, so was converted to tensorflow
        """
        Calibrate bounding boxes through bounding box regression
        
        :param boundingbox: tensor
          all bounding boxes obtained so far as part of the face detection process
        :param reg: tensor
          regression vector
        :return: tensor
          calibrated bounding boxes
        """
        
        if reg.shape[1] == 1:
            reg = tf.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        
        indices = []
        
        for j in range(boundingbox.shape[1]-1):
          for i in range(boundingbox.shape[0]):
            indices.append([i,j])
        
        t = tf.tensor_scatter_nd_update(boundingbox, indices, tf.transpose(tf.concat([b1, b2, b3, b4], axis = 0)))        
        
        return t

    @staticmethod
    def __tf_apply_patch(self, img, patch, ground_truths_of_image): #converted to tensorflow
        """
        Applies the patch to the image. Please use this Function only in new_detect_faces().

        :param img: nd-array
          The original image
        :param patch: tensor
          The adversarial patch
        :param ground_truths_of_image: list
          Ground truth bounding boxes of the faces on the image
        :return: tensor
          The original image but with the patch placed, relative to the ground truths 
        """
        alpha = 0.5
        for bounding_box in ground_truths_of_image:  # ground truth loop

            resize_value = alpha * math.sqrt(bounding_box[2] * bounding_box[3]) #as per the paper - careful, alpha here is hardcoded, independently of the ones used elsewhere
            tf_resized_P = tf.image.resize(patch, (round(resize_value), round(resize_value)), method = self.resize_method, antialias = self.antialias) #tf image resize LANCZOS instead of cv2.AREA. 
            #Reason: method='area' doesn't have built-in gradient approximation for backpropagation. Until it is added we use lanczos5 with antialiasing to approximate the results of cv2.AREA
            #lanczos5 should be suitable approximation for cv2 resize method with method parameter set to area (each value is 1-5 pixels different at most, when running comparisons).
            
            x_P = round(bounding_box[2] / 2)
            y_P = round(resize_value / 2)
            
            adv_img_rows = img.shape[0]
            adv_img_cols = img.shape[1]
            
            #coordinates of the patch, as per the paper's
            y_start = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2) #bounding_box[0]
            x_start = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2) #bounding_box[1]
            y_end = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2) + tf_resized_P.shape[1] # x_start + tf_resized_p.shape[0]
            x_end = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2) + tf_resized_P.shape[0] # y_start + tf_resized_p.shape[1]
            p_shape = tf.shape(tf_resized_P)

            #covers niche cases where after calculations we'd end up with a negative pixel value (impossible)
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
            
            #place patch over image at the correct coordinates
            overlay = tf_resized_P - img[y_start:y_end, x_start:x_end]
            overlay_pad = tf.pad(overlay, [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]]) 
            img = img + overlay_pad
            
        return img

    def new_detect_faces(self, img, patch, ground_truths_of_image, amplification) -> list:
        """
        Detects bounding boxes from the specified image.
        
        :param img: nd-array
          image to process
        :param patch: tensor
          adversarial patch we use to lower face detection confidence scores
        :param ground_truths_of_image: list
          ground truth list to compare our results to -> passed to __tf_apply_patch
        :amplification: int
          gradient amplification factor, used to offset working with smaller image number
        :return: list
          list containing all the bounding boxes detected with their keypoints.
        """
        self.amplification = amplification
        self.patch = tf.Variable(patch, dtype=tf.float32) #convert patch to a tensorflow variable
        self.ground_truths_of_image = ground_truths_of_image

        tf_adv_img = self.__tf_apply_patch(self, img, self.patch, ground_truths_of_image) #apply patch to the images
        self.adv_img = tf_adv_img.numpy() #store a numpy copy of the patched images for compatibility purposes, this variable will be used now instead of tf_adv_img
        if self.adv_img is None or not hasattr(self.adv_img, "shape"):
            raise InvalidImage("Image not valid.")

        height, width, _ = self.adv_img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self._min_face_size # 12 is the face size the P network can detect
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__new_stage1, self.__new_stage2, self.__new_stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in range(len(stages)):
            result = stages[stage](self.adv_img, result[0], result[1])
            if stage is 1: #if it's self.__stage2 -> dependency on immutability of stages list
              old_result = result
              result = [None, None] #for re-entry into stage3
              result[0] = tf.cast(old_result[0], dtype=tf.float32) #convert to Tensor so we can use it immediately upon entry to stage3, which is ONET
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
            
        return (bounding_boxes, self.adv_img, self.patch)

    
    def __new_stage1(self, image, scales: list, stage_status: StageStatus): #left unchanged - responsible for PNET
        """
        First stage of the MTCNN.
        
        :param image: nd-array
          image to process
        :param scales: list
          list of scales used for image rescaling
        :param stage_status: StageStatus
          padding data kept track of by a StageStatus instance
        
        :return total_boxes: nd-array
          face bounding boxes detected by PNET
        :return status: StageStatus
          modified padding data kept track of by a StageStatus instance
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

    def __new_stage2(self, img, total_boxes, stage_status: StageStatus): #left unchanged - responsible for RNET
        """
        Second stage of the MTCNN.
        
        :param img: nd-array
          image to be processed
        :param total_boxes: nd-array
          face bounding boxes detected by PNET
        :param stage_status: StageStatus
        
        :return total_boxes: nd-array
          face bounding boxes detected by RNET
        :return status: StageStatus
          modified padding data kept track of by a StageStatus instance
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

    def __new_stage3(self, img, total_boxes, stage_status: StageStatus): #converted to tensorflow - responsible for ONET
      
      """
        Third stage of the MTCNN. Converted to tensorflow to be able to automatically backpropagate gradient generation over ONET.

        :param img: nd-array
          image used for face detection and also adversarial patch generation
        :param total_boxes: tensor
          all bounding boxes taken over from stage 2, converted into a tensor between stages 2 and 3
        :param stage_status: StageStatus
          local MTCNN parameters for padding saved through a StageStatus instance
          
        :return tf_total_boxes: tensor
          final set of face detection results in the form of bounding boxes, coupled with confidence scores
        :return points: nd-array
          positions of key points on the face 
      """
      with self.tape as tape: #used to calculate the gradient
        tape.watch(self.patch)
        tf_img = self.__tf_apply_patch(self, img, self.patch, self.ground_truths_of_image) # (re)apply patch to image: needed so the gradient Tape "sees" it and can backpropagate over it
        
        new_total_boxes = []
        num_boxes = len(total_boxes)
        tf_total_boxes = tf.cast(total_boxes, dtype=tf.float32)

        if num_boxes == 0:
            print('no faces detected')
            return tf_total_boxes, np.empty(shape=(0,))  # it's just face points - can be left in np, as gradient doesn't need it

        total_boxes = np.fix(total_boxes).astype(np.int32) #both numpy and tensorflow versions are kept and used throughout, so sadly the code is doubled
        status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                             width=stage_status.width, height=stage_status.height)
        
        tempimg = np.zeros((48, 48, 3, num_boxes))
        tf_tempimg = tf.zeros((48, 48, 3, num_boxes))
        
        tf_total_boxes = tf.experimental.numpy.fix(tf_total_boxes)
        tf_total_boxes = tf.cast(tf_total_boxes, dtype=tf.int32)

        tf_status = StageStatus(self.__tf_pad(tf_total_boxes, stage_status.width, stage_status.height),
                             width=stage_status.width, height=stage_status.height)
        
        tf_tempimgs = []

        for k in range(0, num_boxes):
            tf_tmp = tf_img[tf_status.y[k] - 1:tf_status.ey[k], tf_status.x[k] - 1:tf_status.ex[k], :]
            if tf_tmp.shape[0] > 0 and tf_tmp.shape[1] > 0 or tf_tmp.shape[0] == 0 and tf_tmp.shape[1] == 0:
                #tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48), method = 'area')) # - The correct one - unfortunately doesn't have a defined automatic gradient
                #tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48))) #method = 'bilinear' is sometimes very close, 'lanczos5' is one significant digit off, so better on average 
                tf_tempimgs.append(tf.image.resize(tf_tmp, (48, 48), method = self.resize_method, antialias = self.antialias))
            else:
                print('tf_tmp is misshapen')
                return tf.empty(shape=(0,))
        
        tf_tempimg = tf.stack(tf_tempimgs, 3)
        
        for k in range(0, num_boxes):
            tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))
            tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:     
                tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                print('np_tmp is misshapen')
                return tf.empty(shape=(0,)), np.empty(shape=(0,))
        
        tf_tempimg = (tf_tempimg - 127.5) * 0.0078125
        tf_tempimg2 = tf.transpose(tf_tempimg, (3, 1, 0, 2))
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        tf_total_boxes = tf.cast(total_boxes, dtype=tf.float32)

        tf_out = self._onet(tf_tempimg2) #get ONET result 
        tf_out2 = tf.transpose(tf_out[2])
        tf_out0 = tf.transpose(tf_out[0])
        tf_score = tf_out2[1, :]

        tf_ipass = tf.transpose(tf.where(tf_score > self._steps_threshold[2]))
        tf_ipass = tf.cast(tf_ipass, dtype=tf.float32)
        
        a = []
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
          return [], np.transpose(tf_out[1]) #points (second result) should always be left in numpy format
          
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

        """
           np segment - most things here were ported to tensorflow above -
           to ensure automatic gradient differentiation & backpropagation compatibility
        """ 
        out = self._onet(tempimg1)
        out0 = np.transpose(out[0])

        out2 = np.transpose(out[2])
        score = out2[1, :] 

        out1 = np.transpose(out[1])

        points = out1  # points doesn't influence image or patch, can be kept in numpy format

        ipass = np.where(score > self._steps_threshold[2]) 

        points = points[:, ipass[0]]

        total_boxes = np.hstack(
            [total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])  

        mv = out0[:, ipass[0]] 
        
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1  
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1  

        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0],
                                                                       (5, 1)) - 1  
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1],
                                                                         (5, 1)) - 1  
        
        if total_boxes.shape[0] > 0:
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
            pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]
        
        tf_points = points
        
        # more doubling
        if tf_total_boxes.shape[0] > 0:
            tf_total_boxes = self.__tf_bbreg(tf_total_boxes, tf.transpose(
                tf_mv))
            tf_pick = self.__tf_nms(tf_total_boxes, 0.7, 'Min')
            tf_pick = tf.cast(tf_pick, dtype=tf.int32)  # initially is type int16 which tensorflow doesn't support ... for some reason
            tf_total_boxes = tf.gather(tf_total_boxes, tf_pick)

        indices = []
        tf_outie = tf.transpose(tf_out[2])[1, :] #needed to extract correct indices in result -> used to create the tf_loss oneliner
        tf_loss = 0
        
        if tf_total_boxes.shape[0] > 0:
          for score in tf_total_boxes[:, 4]:
              for i in range(len(tf_outie)):
                  if score == tf_outie[i]:
                      indices.append(i)
        
        """
        Why we need the one-liner below: for some unkown reason, Tensorflow in its current version refuses to correctly calculate the gradient if all the operations aren't sequentially done in one line.
        Could be because assigning Tensors to a variable creates unique copies, rather than references - whatever the reason, this needs to be kept in one line to ensure the gradient won't return None.
        """
        
        if len(indices):
          tf_loss = tf.math.negative(tf.divide(tf.math.reduce_sum(tf.math.log(tf.gather(tf.transpose(self._onet(tf_tempimg2)[2])[1, :], indices))), len(indices))) 
      
      # exiting gradientTape-watching area  
      if tf_loss:
        gradient = tape.gradient(tf_loss, self.patch) # calculate gradient based on loss. self.patch -> tf_img -> MANY tf_tmp -> tf_tempimg -> / DOESN'T KNOW HOW TO DO GRADIENT OF 'AREA' RESIZE / -> tf_tempimg2 -> tf_loss
        self.patch.assign(tf.clip_by_value(self.patch+gradient*self.amplification, clip_value_min = 0, clip_value_max = 255)) #update patch and normalize value to within the RGB spectrum
      return tf_total_boxes, points
