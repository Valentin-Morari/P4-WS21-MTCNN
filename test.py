import tensorflow as tf
import numpy as np
import mtcnn
from mtcnn.network.factory import NetworkFactory
import matplotlib.pyplot as plt
import cv2
import pkg_resources
import matplotlib.patches as mpatches
import os
import math

# keras_scratch_graph problem
# gpu version
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

detector = mtcnn.MTCNN()


def scale_image(image, scale):
    height, width, _ = image.shape
    width_scaled = int(np.ceil(width * scale))
    height_scaled = int(np.ceil(height * scale))
    scaleimage = cv2.resize(
        image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

    '''MTCNN normalized the image's pixels here, but Attack doesn't. 
        They moved it to imageChangeToFitPnet() 
        because they put the scaled image into a tensor.#'''

    return scaleimage


def generate_bounding_box(imap, reg, scale, t):
        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        imap = tf.transpose(imap)

        dx1tf = tf.transpose(reg[:, :, 0])
        dy1tf = tf.transpose(reg[:, :, 1])
        dx2tf = tf.transpose(reg[:, :, 2])
        dy2tf = tf.transpose(reg[:, :, 3])

        positives = tf.where(tf.greater_equal(imap, t))

        scoretf = tf.gather_nd(imap, positives)

        regtf = tf.concat([[tf.gather_nd(dx1tf, positives)], [tf.gather_nd(dy1tf, positives)],
                           [tf.gather_nd(dx2tf, positives)], [tf.gather_nd(dy2tf, positives)]],
                          axis=0) # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn
        regtf = tf.transpose(regtf)


        if tf.size(regtf) == 0:
            regtf = tf.reshape(tf.convert_to_tensor(()), (0, 3)) # not sure if I can port this to tensorflow

        bbtf = positives

        q1tf = tf.floor((stride * tf.cast(bbtf, dtype=tf.float32) + 1.) / scale)
        q2tf = tf.floor((stride * tf.cast(bbtf, dtype=tf.float32) + cellsize) / scale)

        boundingboxtf = tf.concat([q1tf, q2tf, tf.expand_dims(scoretf, 1), regtf], axis=1) # tf.concat(value, axis=1) is counterpart of the numpy function np.hstack() in Tensorflow

        return boundingboxtf, regtf


def nms(boxes, threshold, method):
    """
    Non Maximum Suppression.

    :param boxes: Tensorflow np array with bounding boxes.
    :param threshold:
    :param method: NMS method to apply. Available values ('Min', 'Union')
    :return:
    """

    '''
    I think that this creates an NONE gradient.
    '''
    if tf.size(boxes) == 0:  # start here
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
    #pick = tf.cast(pick, dtype=tf.int32) # I have to use the type tf.int32 and not tf.int16
    return pick

def rerec(bbox):
    # convert bbox to square

    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    max_side_length = tf.maximum(width, height)

    v21 = bbox[:, 0] + width * 0.5 - max_side_length * 0.5 # This should be x of a bounding box
    v22 = bbox[:, 1] + height * 0.5 - max_side_length * 0.5 # This should be y of a bounding box
    v23 = bbox[:, 0:2] + tf.transpose(tf.tile([max_side_length], [2, 1]))
    confidences = bbox[:, 4] # This are the confidence scores

    bbox = tf.stack([v21, v22, v23[:, 0], v23[:, 1], confidences], axis=1) ### !!!!!!!!!!!!! look again

    return bbox


def process_pnet_result(pnet_result):
    total_boxes = tf.reshape(tf.convert_to_tensor(()), (0, 9))

    scale_factor = 0.709 # Set like mtcnn __init__ did
    steps_threshold = [0.6, 0.7, 0.7] # Set like mtcnn __init__ did

    out0 = tf.transpose(pnet_result[0], perm=[0, 2, 1, 3])
    out1 = tf.transpose(pnet_result[1], perm=[0, 2, 1, 3])

    boxes, _ = generate_bounding_box(tf.identity(out1[0, :, :, 1]), tf.identity(out0[0, :, :, :]), scale_factor, steps_threshold[0]) # tf.identity should be the counterpart of numpy.copy

    pick = nms(tf.identity(boxes), 0.5, 'Union') # tf.identity should be the counterpart of numpy.copy

    picktf = tf.cast(pick, dtype=tf.int32) #I have to use the type tf.int32 and not tf.int16

    tf_pick = tf.image.non_max_suppression(tf.identity(boxes[:, 0:4]), tf.identity(boxes[:, 4]), 100, iou_threshold=0.5)

    if tf.size(boxes) > 0 and tf.size(tf_pick) > 0:
        boxes = tf.gather(boxes, indices=tf_pick)
        total_boxes = tf.concat([total_boxes, boxes], axis=0)

    numboxes = total_boxes.shape[0]

    if numboxes > 0:
        pick = nms(tf.identity(total_boxes), 0.7, 'Union') # tf.identity should be the counterpart of numpy.copy
        picktf = tf.cast(pick, dtype=tf.int32) #I have to use the type tf.int32 and not tf.int16
        total_boxes = tf.gather(total_boxes, indices=picktf)

        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]

        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = tf.transpose(tf.concat([[qq1], [qq2], [qq3], [qq4], [total_boxes[:, 4]]],
                                             axis=0))  # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn

        total_boxes = rerec(tf.identity(total_boxes))

        total_boxes_03 = tf.floor(total_boxes[:, 0:4])  # np.fix(total_boxes[:, 0:4]).astype(np.int32)
        total_boxes_4 = total_boxes[:, 4]
        total_boxes = tf.stack(
            [total_boxes_03[:, 0], total_boxes_03[:, 1], total_boxes_03[:, 2], total_boxes_03[:, 3], total_boxes_4],
            axis=1)

    return total_boxes

def loss_object(label, predict_box):
    #print(label)
    # MSE loss
    #loss = tf.math.reduce_mean(tf.math.square(tf.subtract(label, predict_box)))
    loss = tf.divide(tf.math.reduce_sum(tf.math.log(predict_box)), len(predict_box)) #THIS DOES ASCENT WITH TENSORFLOW DESCENT
    #loss = tf.negative(tf.divide(tf.math.reduce_sum(tf.math.log(predict_box)), len(predict_box))) #THIS DOES DESCENT WITH TENSORFLOW DESCENT
    #print(loss)
    return loss


def createPnet():
    weight_file = 'mtcnn_weights.npy'
    weights = np.load(weight_file, allow_pickle=True).tolist()
    pnet = NetworkFactory().build_pnet()
    pnet.set_weights(weights['pnet'])
    return pnet


pnet_attacked = createPnet()



def imageChangeToFitPnet(image, scale):
    image = scale_image(image, scale)
    image = tf.cast(image, dtype=tf.float32)
    # Normalize image
    image = (image - 127.5) * 0.0078125
    image = image[tf.newaxis, ...]
    image = tf.transpose(image, (0, 2, 1, 3))
    return image


def createLabel(image, scale):
    image = imageChangeToFitPnet(image, scale)
    label = pnet_attacked(image)
    return label[0][0, :, :, :]


def create_adversarial_pattern(image, label):
    with tf.GradientTape() as tape:
        tape.watch(image)

        pnet_probe = pnet_attacked(image)
        probe = process_pnet_result(pnet_probe)

        confidence_scores = tf.convert_to_tensor(probe[:, 4], dtype=tf.float32)
        loss = loss_object(label, confidence_scores) #label will be needed for the IOU later
        print("LOSS: ")
        print(loss)
    # Get the gradients of the loss to the input image
    gradient = tape.gradient(loss, image)
    '''
    if gradient.numpy().any() != 0:
        print(True)
    else:
        print(False)
    '''
    #print("GRADIANT: ")
    #print(gradient)
    # Normalization, all gradient divide the max gradient's absolute value
    #norm_gradient = tf.math.divide(gradient, tf.math.reduce_max(tf.math.abs(gradient)))
    return gradient, loss#norm_gradient, loss


def createMask(results, image):
    mask = np.zeros((image.shape[0], image.shape[1]))

    b = results[0]  # first bounding box
    x1, y1, width, height = b['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    left_eye_y = b['keypoints']['left_eye'][1]
    right_eye_y = b['keypoints']['right_eye'][1]
    left_mouth_y = b['keypoints']['mouth_left'][1]
    right_mouth_y = b['keypoints']['mouth_right'][1]
    nose_y = b['keypoints']['nose'][1]

    # Patch coordinate
    mask_x1 = x1 + int(width/20)
    mask_x2 = x2 - int(width/20)
    mask_y1 = nose_y
    mask_y2 = y2

    if mask_y1 < 0:
        mask_y1 = 0
    if mask_x1 < 0:
        mask_x1 = 0

    mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1
    # frame_patch
    # mask[y1:y2, x1:x2] = 0

    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    return mask, mask_x1, mask_x2, mask_y1, mask_y2


def my_createMask(results, image):
    '''

    :param results: result of mtcnn for the given image
    :param image: image to which the patch will be applied to in future functions
    :return: the adversial "mask" with it's coordinates mask_x1, mask_x2, mask_y1, mask_y2
    '''

    '''
    Originally attack only uses Images with only one face. This addition adds patches to all found faces.
    '''
    if len(results) == 1: # With this bounding_boxes is always 2 dimensional
        bounding_boxes = np.array([results[0]['box']])
    else:
        bounding_boxes = np.empty((0, 4), int)  # CHANGE I'm not sure if using np.empty is safe

        for i in range(len(result)):
            bounding_boxes = np.append(bounding_boxes, np.array([result[i]['box']]), axis=0)

    mask = np.zeros((image.shape[0], image.shape[1]))

    alpha = 0.5

    for bounding_box in bounding_boxes:
        x, y, width, height = bounding_box

        resize_value = alpha * math.sqrt(width * height)

        x_P = round(width / 2)
        y_P = round(resize_value/ 2)

        # Patch coordinate
        mask_x1 = x_P + x - round(resize_value / 2)
        mask_x2 = x_P + x - round(resize_value / 2) + round(resize_value) # have to round resize_value because its a float
        mask_y1 = y_P + y - round(resize_value / 2)
        mask_y2 = y_P + y - round(resize_value / 2) + round(resize_value) # have to round resize_value because its a float

        '''
        We need to look if this also happens with our Patch!!!!! This should not happen using the real ground truths.
        '''
        if mask_y1 < 0:
            mask_y1 = 0
        if mask_x1 < 0:
            mask_x1 = 0

        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1
    # frame_patch
    # mask[y1:y2, x1:x2] = 0

    mask = np.repeat(mask[..., np.newaxis], 3, 2)
    return mask, mask_x1, mask_x2, mask_y1, mask_y2


def tf_apply_patch(img, patch, ground_truths_of_image):
    """
    Applies the patch to the image. Please use this Function only in new_detect_faces().

    :param img: The original image
    :param patch: The adversial patch
    :param ground_truths_of_image: Ground truth bounding boxes of the image, i.e. the marked faces
    :return: The original image but with the patch placed, dependet on where the ground truths are given
    """
    #adv_img = copy.deepcopy(img)
    tf_adv_img = tf.cast(img, dtype=tf.float32)

    alpha = 0.5

    # draw detected face + plaster patch over source
    for bounding_box in ground_truths_of_image:  # ground truth loop

        resize_value = alpha * math.sqrt(bounding_box[2] * bounding_box[3]) #CHANGE! need to remove math
        tf_resized_P = tf.image.resize(patch, (round(resize_value), round(resize_value)))

        x_P = round(bounding_box[2] / 2)
        y_P = round(resize_value / 2)

        adv_img_rows = tf_adv_img.shape[0]
        adv_img_cols = tf_adv_img.shape[1]

        # Finding the indices where to put the patch
        y_start = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2)  # bounding_box[0]
        x_start = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2)  # bounding_box[1]
        y_end = y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2) + tf_resized_P.shape[
            1]
        x_end = x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2) + tf_resized_P.shape[
            0]

        tf_overlay = tf_resized_P - tf_adv_img[y_start:y_end, x_start:x_end]
        tf_overlay_pad = tf.pad(tf_overlay, [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]])
        tf_adv_img = tf_adv_img + tf_overlay_pad

    return tf_adv_img


def getPerturbationRGB(image, imageWithPerturbations):
    temp = tf.squeeze(imageWithPerturbations)
    temp = temp / 0.0078125 + 127.5
    temp = tf.transpose(temp, (1, 0, 2))
    temp = cv2.resize(temp.numpy(), (image.shape[1], image.shape[0]))

    return temp * mask


def iterative_attack(adv_image, label, mask, scale, learning_rate, scaleNum): #CHANGED: Removed GRAYSCALE

    upperBound = (255 - 127.5) * 0.0078125 # ~ +1
    lowerBound = (0 - 127.5) * 0.0078125 # ~ -1

    my_mask = scale_image(mask, scale)
    my_mask = tf.cast(my_mask, dtype=tf.float32)

    my_mask = my_mask[tf.newaxis, ...]
    my_mask = tf.transpose(my_mask, (0, 2, 1, 3))

    maxloss = 0.0
    count = 0

    for epoch in range(100):
        print("_____")
        print("Epoch %d", epoch)
        tf.dtypes.cast(adv_image, tf.int32)
        tf.dtypes.cast(adv_image, tf.float32)

        perturbations, loss = create_adversarial_pattern(adv_image, label)
        perturbations = perturbations.numpy()

        adv_image = adv_image + \
            learning_rate[scaleNum] * (perturbations * my_mask)
        adv_image = tf.where(adv_image < lowerBound, lowerBound, adv_image) # everything smaller than -1 is cast to -1
        adv_image = tf.where(adv_image > upperBound, upperBound, adv_image) # everything bigger than +1 is cast to +1


        '''
        # Original Optimizer of attack.py
        if maxloss < loss:
            maxloss = loss
            count = 0
        else:
            count += 1
        if count > 2:
            if learning_rate[scaleNum] > 0.01:
                learning_rate[scaleNum] /= 2
            count = 0
        '''

        if epoch == 60:
            learning_rate = learning_rate * 0.1
        if epoch == 80:
            learning_rate = learning_rate * 0.1

    return adv_image


# One meter to five meter (only 1 meter)
for distance in range(1, 2): # CHANGED
    allFileList = os.listdir('./patch_img')
    pictureList = os.listdir('./picture/{}M/normal'.format(distance))
    true_box_info = np.load('./picture/{}M/normal/info_changed.npy'.format(distance), allow_pickle=True)

    for file in allFileList:
        patch_name = file.split('.')[0]

        for pic in pictureList:

            pic_name = pic.split('.')[0]
            if pic_name == 'info':
                break
            which_pic = int(pic_name[0]) - 1

            image = tf.keras.preprocessing.image.load_img('./picture/{}M/normal/'.format(distance) + pic)
            patch = tf.keras.preprocessing.image.load_img('patch_img/'+file)

            image = tf.keras.preprocessing.image.img_to_array(image)
            patch = tf.keras.preprocessing.image.img_to_array(patch)

            result = detector.detect_faces(image)

            mask, mask_x1, mask_x2, mask_y1, mask_y2 = my_createMask(result, image)
            
            # Add code to mtcnn.py to output all scales
            # Pack them to 'scales.npy'
            scales = np.load('scales.npy')

            image2 = image.copy()

            patch = cv2.resize(patch, (abs(mask_x1-mask_x2), abs(mask_y1-mask_y2)))

            storePatch = patch
            cv2.imwrite(
                'result/patch/rgb/mouth/normal/{}M/'.format(distance) + "_supp_" + patch_name + '_' + pic_name + '.jpg',
                storePatch[:, :, [2, 1, 0]])

            image2[mask_y1:mask_y2, mask_x1:mask_x2] = patch
            # frame_patch
            '''x1, y1, width, height = result[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            image2[y1:y2,x1:x2] = image[y1:y2,x1:x2]'''
            ##########

            show_img = image2
            # 1M scales[5:8]
            # 2M scales[4:7]
            # 3M scales[2:5]
            # 4M scales[1:4]
            # 5M scales[0:3]
            scaleStartIndex = [5, 4, 2, 1, 0]
            learning_rate = np.array([0.01, 0.01, 0.01])#[1.0, 1.0, 1.0]

            for i in range(10):
                print("*****")
                print("Iteration: %d", i, "of Picture:", pic_name)
                # Pick three scale to train for each distance
                for scaleNum in range(3):
                    print('start')
                    which_scale = scaleStartIndex[distance - 1] + scaleNum

                    scale = scales[which_scale]
                    # scale = 1

                    tempImg = show_img - show_img * mask

                    cv2.imwrite('result/patch/rgb/mouth/normal/{}M/'.format(
                        distance) + "_tempIMG_" + patch_name + '_' + pic_name + '.jpg',
                                tempImg[:, :, [2, 1, 0]])

                    adv_image = imageChangeToFitPnet(show_img, scale)

                    label = createLabel(image, scale)

                    perturbations = iterative_attack(adv_image, label, mask, scale, learning_rate, scaleNum)

                    perturbationsRGB = getPerturbationRGB(image, perturbations.numpy())

                    show_img = tempImg + perturbationsRGB

                    cv2.imwrite('result/patch/rgb/mouth/normal/{}M/'.format(
                        distance) + "_show_img_" + patch_name + '_' + pic_name + '.jpg',
                                show_img[:, :, [2, 1, 0]])

                    show_img = np.where(show_img < 0, 0, show_img)
                    show_img = np.where(show_img > 255, 255, show_img)

            image3 = show_img
            image3 = image3.astype(np.int32)
            image3 = image3.astype(np.float32)
            results = detector.detect_faces(image3)

            storePatch = image3[mask_y1:mask_y2, mask_x1:mask_x2]

            storePatch = storePatch.astype(np.int32)
            storePatch = storePatch.astype(np.float32)

            plt.figure(figsize=(image3.shape[1]*6//image3.shape[0], 6))
            plt.imshow(image3/255)
            plt.axis(False)
            ax = plt.gca()

            # Count iou
            iou_mat = np.zeros((image.shape[0], image.shape[1]))
            t_x1, t_y1, t_x2, t_y2 = true_box_info[which_pic]['box']
            iou_mat[t_y1:t_y2, t_x1:t_x2] += 1

            for b in results:
                x1, y1, width, height = b['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                if b == results[0]:
                    iou_mat[y1:y2, x1:x2] += 1

                plt.text(x1, y1, '{:.2f}'.format(b['confidence']), color='red')
                ax.add_patch(mpatches.Rectangle((x1, y1), width, height, ec='red', alpha=1, fill=None))

            iou = round(len(iou_mat[np.where(iou_mat == 2)]) /len(iou_mat[np.where(iou_mat > 0)]), 2)

            plt.savefig('result/image/rgb/mouth/normal/{}M/'.format(distance) + patch_name+'_' + pic_name + '_iou=' + str(iou)+'.jpg')

            cv2.imwrite('result/patch/rgb/mouth/normal/{}M/'.format(distance) + patch_name+'_' + pic_name + '.jpg', storePatch[:, :, [2, 1, 0]])
