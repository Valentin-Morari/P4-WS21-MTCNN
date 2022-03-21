import tensorflow as tf
import numpy as np
import mtcnn
from mtcnn.network.factory import NetworkFactory
import matplotlib.pyplot as plt
import cv2
import pkg_resources
import matplotlib.patches as mpatches
import os

# keras_scratch_graph problem
# gpu version
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

detector = mtcnn.MTCNN()


def tf_scale_image(img, scale):
    height, width, _ = img.shape
    width_scaled = tf.math.ceil(width * scale)
    height_scaled = tf.math.ceil(height * scale)
    scaleimage = tf.image.resize(
        img, (width_scaled, height_scaled), method='lanczos5', antialias=True)

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
                      axis=0)  # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn
    regtf = tf.transpose(regtf)

    if tf.size(regtf) == 0:
        regtf = tf.reshape(tf.convert_to_tensor(()), (0, 3))  # not sure if I can port this to tensorflow

    bbtf = positives

    #TODO
    q1tf = tf.experimental.numpy.fix((stride * tf.cast(bbtf, dtype=tf.float32) + 1.) / scale)
    q2tf = tf.experimental.numpy.fix((stride * tf.cast(bbtf, dtype=tf.float32) + cellsize) / scale)

    boundingboxtf = tf.concat([q1tf, q2tf, tf.expand_dims(scoretf, 1), regtf],
                              axis=1)  # tf.concat(value, axis=1) is counterpart of the numpy function np.hstack() in Tensorflow

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
    # pick = tf.cast(pick, dtype=tf.int32) # I have to use the type tf.int32 and not tf.int16
    return pick


def rerec(bbox):
    # convert bbox to square

    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    max_side_length = tf.maximum(width, height)

    v21 = bbox[:, 0] + width * 0.5 - max_side_length * 0.5  # This should be x of a bounding box
    v22 = bbox[:, 1] + height * 0.5 - max_side_length * 0.5  # This should be y of a bounding box
    v23 = bbox[:, 0:2] + tf.transpose(tf.tile([max_side_length], [2, 1]))
    confidences = bbox[:, 4]  # This are the confidence scores

    bbox = tf.stack([v21, v22, v23[:, 0], v23[:, 1], confidences], axis=1)  ###TODO

    return bbox


def process_pnet_result(pnet_result):
    total_boxes = tf.reshape(tf.convert_to_tensor(()), (0, 9))

    scale_factor = 0.709  # Set like mtcnn __init__ did
    steps_threshold = [0.6, 0.7, 0.7]  # Set like mtcnn __init__ did

    out0 = tf.transpose(pnet_result[0], perm=[0, 2, 1, 3])
    out1 = tf.transpose(pnet_result[1], perm=[0, 2, 1, 3])

    boxes, _ = generate_bounding_box(tf.identity(out1[0, :, :, 1]), tf.identity(out0[0, :, :, :]), scale_factor,
                                     steps_threshold[0])  # tf.identity should be the counterpart of numpy.copy

    # pick = nms(tf.identity(boxes), 0.5, 'Union')  # tf.identity should be the counterpart of numpy.copy

    # picktf = tf.cast(pick, dtype=tf.int32)  # I have to use the type tf.int32 and not tf.int16

    tf_pick = tf.image.non_max_suppression(tf.identity(boxes[:, 0:4]), tf.identity(boxes[:, 4]), 100, iou_threshold=0.5)

    if tf.size(boxes) > 0 and tf.size(tf_pick) > 0:
        boxes = tf.gather(boxes, indices=tf_pick)
        total_boxes = tf.concat([total_boxes, boxes], axis=0)

    numboxes = total_boxes.shape[0]

    if numboxes > 0:
        # pick = nms(tf.identity(total_boxes), 0.7, 'Union')  # tf.identity should be the counterpart of numpy.copy
        # picktf = tf.cast(pick, dtype=tf.int32)  # I have to use the type tf.int32 and not tf.int16
        '''I'm using NMS from TensorFlow instead of our NSMS.'''
        tf_pick = tf.image.non_max_suppression(tf.identity(total_boxes[:, 0:4]), tf.identity(total_boxes[:, 4]), 100,
                                               iou_threshold=0.7)
        total_boxes = tf.gather(total_boxes, indices=tf_pick)

        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]

        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = tf.transpose(tf.concat([[qq1], [qq2], [qq3], [qq4], [total_boxes[:, 4]]],
                                             axis=0))  # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn

        total_boxes = rerec(tf.identity(total_boxes))

        total_boxes_03 = tf.experimental.numpy.fix(total_boxes[:, 0:4])
        total_boxes_4 = total_boxes[:, 4]
        total_boxes = tf.stack(
            [total_boxes_03[:, 0], total_boxes_03[:, 1], total_boxes_03[:, 2], total_boxes_03[:, 3], total_boxes_4],
            axis=1)

    return total_boxes


def loss_object(label, predict_box):
    # MSE loss
    # loss = tf.math.reduce_mean(tf.math.square(tf.subtract(label, predict_box)))

    # loss = tf.divide(tf.math.reduce_sum(tf.math.log(predict_box)), len(predict_box)) #THIS DOES ASCENT WITH TENSORFLOW DESCENT
    loss = tf.negative(tf.divide(tf.math.reduce_sum(tf.math.log(predict_box)),
                                 len(predict_box)))  # THIS DOES DESCENT WITH TENSORFLOW DESCENT
    # print(loss)
    return loss


def createPnet():
    weight_file = 'mtcnn_weights.npy'
    weights = np.load(weight_file, allow_pickle=True).tolist()
    pnet = NetworkFactory().build_pnet()
    pnet.set_weights(weights['pnet'])
    return pnet


pnet_attacked = createPnet()


def imageChangeToFitPnet(image, scale):
    image = tf_scale_image(image, scale)
    # image = tf.cast(image, dtype=tf.float32)
    # Normalize image
    # image = (image - 127.5) * 0.0078125 # using it know before this function
    image = image[tf.newaxis, ...]
    image = tf.transpose(image, (0, 2, 1, 3))
    return image


def createLabel(image, scale):
    image = imageChangeToFitPnet(image, scale)
    label = pnet_attacked(image)
    return label[0][0, :, :, :]


def tf_IoU(boxA, boxB):

    xA = tf.math.maximum(boxA[0], boxB[0])
    yA = tf.math.maximum(boxA[1], boxB[1])

    xB = tf.math.minimum(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = tf.math.minimum(boxA[1] + boxA[3], boxB[1] + boxB[3])

    print("xA:")
    print(xA)
    print("yA:")
    print(yA)
    print("xB:")
    print(xB)
    print("yB:")
    print(yB)

    # compute the area of intersection rectangle
    interArea = tf.math.maximum(0, xB - xA + 1) * tf.math.maximum(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # print("interArea = %d -- boxAArea = %d -- boxBArea = %d" % (interArea, boxAArea, boxBArea))
    print("InterArea:")
    print(interArea)
    print("BoxA:")
    print(boxAArea)
    print("BoxB")
    print(boxBArea)
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    print(iou)
    return iou

def tf_IoU_multiple_boxes(bounding_boxes, ground_truths):
    iou_result = []

    for bb in bounding_boxes:
        print(bb)
        iou = 0

        for gt in ground_truths:
            print(gt)
            new_iou = tf_IoU(bb[0:4], gt)

            if new_iou > iou:
                iou = new_iou

        if iou > 0.6:
            iou_result.append(bb)
    tf_iou_result = tf.cast(iou_result, dtype=tf.float32)
    print("IoU PASSED")
    print(tf_iou_result)

    return tf_iou_result

def create_adversarial_pattern(image, label, ground_truth_boxes, scale):
    with tf.GradientTape() as tape:
        # tape.watch(var_patch)

        # print("IMAGE:")
        # print(image)
        adv_image = tf_apply_patch(image, var_patch, ground_truth_boxes)
        if adv_image == None:
            print("ATTENTION PATCH WASN'T ADDED TO THE IMAGE!!!")
            print(ground_truth_boxes)

        adv_image = imageChangeToFitPnet(adv_image, scale)

        pnet_probe = pnet_attacked(adv_image)

        probe = process_pnet_result(pnet_probe)
        '''
        # Right now I'm trying to implement IoU from our paper into the tape,
            but I'm only getting a IoU of zero, at the moment. 
        #We need to resize our ground truth boxes, since the image was also resized
        resized_ground_truth_boxes = ground_truth_boxes * scale

        iou_probe = tf_IoU_multiple_boxes(probe, resized_ground_truth_boxes)

        if tf.size(iou_probe) == 0:
            confidence_scores = tf.cast(np.array([0]), dtype=tf.float32)
        else:
            confidence_scores = iou_probe[:, 4]
        '''
        confidence_scores = probe[:, 4]
        # TODO
        loss = ascent_or_descent * loss_object(label, confidence_scores)  # label will be needed for the IOU later

        # TO_DO
        # loss = loss_object(label, process_pnet_result(pnet_attacked(imageChangeToFitPnet(
        #    tf_apply_patch(image, var_patch, ground_truth_boxes), scale)))[:, 4])
        print("LOSS: ")
        print(loss)
    # TODO
    gradient = tape.gradient(loss, var_patch)
    #print(gradient)

    return gradient, loss

#TODO
def not_negative(x):
  if x<0:
    return 0
  else:
    return x

def tf_apply_patch(img, patch, ground_truths_of_image):
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
    #TODO
    tf_adv_img = None

    # draw detected face + plaster patch over source
    for bounding_box in ground_truths_of_image:  # ground truth loop

        # TODO
        resize_value = alpha * tf.math.sqrt(bounding_box[2] * bounding_box[3])
        tf_resized_P = tf.image.resize(patch, (resize_value, resize_value), method='lanczos5', antialias=True)

        x_P = tf.math.round(bounding_box[2] / 2.0)
        y_P = tf.math.round(resize_value / 2.0)

        adv_img_rows = img.shape[0]
        adv_img_cols = img.shape[1]

        # Finding the indices where to put the patch
        y_start = tf.cast(y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2.0), dtype=tf.int32)  # bounding_box[0]
        x_start = tf.cast(x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2.0), dtype=tf.int32)  # bounding_box[1]

        y_end = tf.cast(y_P + bounding_box[1] - round(tf_resized_P.shape[1] / 2.0) + tf_resized_P.shape[
            1], dtype=tf.int32)
        x_end = tf.cast(x_P + bounding_box[0] - round(tf_resized_P.shape[0] / 2.0) + tf_resized_P.shape[
            0], dtype=tf.int32)

        tf_overlay = tf_resized_P - img[y_start:y_end, x_start:x_end]
        tf_overlay_pad = tf.pad(tf_overlay, [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]])
        tf_adv_img = img + tf_overlay_pad
    if tf_adv_img == None:
        print("ATTENTION no GroundTruth *************************************************")
        print(ground_truths_of_image)
        tf_adv_img = img

    return tf_adv_img


'''Instead of this method getRGB is used right now'''
def getPerturbationRGB(image, imageWithPerturbations):
    temp = tf.squeeze(imageWithPerturbations)
    temp = temp / 0.0078125 + 127.5
    temp = tf.transpose(temp, (1, 0, 2))
    temp = cv2.resize(temp.numpy(), (image.shape[1], image.shape[0]))

    return temp  # * mask


def getRGB(image):
    temp = image / 0.0078125 + 127.5
    return temp


def iterative_attack(target_image, label, ground_truth_boxes, scale, learning_rate,
                     scaleNum):  # CHANGED: Removed GRAYSCALE

    upperBound = (255 - 127.5) * 0.0078125  # ~ +1
    lowerBound = (0 - 127.5) * 0.0078125  # ~ -1

    # print("_____")
    # print("Epoch %d", epoch)

    '''
    #Don't need this right now.
    tf.dtypes.cast(adv_image, tf.int32)
    tf.dtypes.cast(adv_image, tf.float32)
    '''

    gradient, loss = create_adversarial_pattern(target_image, label, ground_truth_boxes, scale)

    # print("GRADIENT:")
    # print(gradient)
    # TODO (sometimes gradient is NONE)
    if (gradient != None):
        opt.apply_gradients(zip([gradient], [var_patch]))
        var_patch.assign(tf.clip_by_value(var_patch, lowerBound, upperBound))
    else:
        print("GRADIENT NONE")

    adv_image = tf_apply_patch(target_image, var_patch, ground_truth_boxes)
    clipped_adv_image = tf.clip_by_value(adv_image, lowerBound, upperBound)

    return clipped_adv_image


def picture_images(learning_rate):

    storePatch = getRGB(var_patch).numpy()

    storePatch = storePatch.astype(np.int32)
    storePatch = storePatch.astype(np.float32)

    cv2.imwrite(
        'result/patch/rgb/mouth/normal/Patch/' + '_INIT' + '_' + patch_name + '_LR= ' + str(learning_rate) + '.jpg'
        , storePatch[:, :, [2, 1, 0]])
    for epoch in range(61):
        # One meter to five meter
        for distance in range(1, 6):
            pictureList = os.listdir('./picture/{}M/normal'.format(distance))
            true_box_info = np.load('./picture/{}M/normal/info.npy'.format(distance), allow_pickle=True)

            for pic in pictureList:

                pic_name = pic.split('.')[0]
                if pic_name == 'info':
                    break
                which_pic = int(pic_name[0]) - 1

                image = tf.keras.preprocessing.image.load_img('./picture/{}M/normal/'.format(distance) + pic)
                image = tf.keras.preprocessing.image.img_to_array(image)

                result = detector.detect_faces(image)
                image = tf.cast(image, dtype=tf.float32)

                if len(result) == 1:  # With this bounding_boxes is always 2 dimensional
                    bounding_boxes = np.array([result[0]['box']])
                else:
                    bounding_boxes = np.empty((0, 4), int)  # CHANGE I'm not sure if using np.empty is safe

                    for i in range(len(result)):
                        bounding_boxes = np.append(bounding_boxes, np.array([result[i]['box']]), axis=0)

                bounding_boxes = tf.cast(bounding_boxes, dtype=tf.float32)

                show_img = image
                # 1M scales[5:8]
                # 2M scales[4:7]
                # 3M scales[2:5]
                # 4M scales[1:4]
                # 5M scales[0:3]
                scales = np.load('scales.npy')
                scaleStartIndex = [5, 4, 2, 1, 0]

                print("*****")
                print("Epoch:", epoch, "of Picture:", pic_name, "of Directory", str(distance) + "M")
                # Pick three scale to train for each distance
                for scaleNum in range(1):  # CHANGED 3 -> 1
                    # print('start', " Scalenum: ", scaleNum)
                    #TODO
                    #opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)  # learning_rate[scaleNum])
                    which_scale = scaleStartIndex[distance - 1] + scaleNum

                    scale = scales[which_scale]

                    label = createLabel(image, scale)

                    '''
                    Preparing the image before the tape.
                    '''
                    normalized_image = (image - 127.5) * 0.0078125  # NORMALIZING the image used

                    attacked_image = iterative_attack(normalized_image, label, bounding_boxes, scale,
                                                      learning_rate, scaleNum)
            if epoch % 10 == 0:
                image3 = attacked_image
                image3 = getRGB(image3).numpy()
                image3 = image3.astype(np.int32)
                image3 = image3.astype(np.float32)

                results = detector.detect_faces(image3)

                storePatch = getRGB(var_patch).numpy()

                storePatch = storePatch.astype(np.int32)
                storePatch = storePatch.astype(np.float32)

                plt.figure(figsize=(image3.shape[1] * 6 // image3.shape[0], 6))
                plt.imshow(image3 / 255)
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

                iou = round(len(iou_mat[np.where(iou_mat == 2)]) / len(iou_mat[np.where(iou_mat > 0)]), 2)

                plt.savefig(
                    'result/image/rgb/mouth/normal/{}M/'.format(
                        distance) + aod + '_NEW' + '_' + patch_name + '_' + pic_name + '_iou=' + str(
                        iou) + '_' + str(epoch) + '_LR= ' + str(learning_rate) + '.jpg')

                cv2.imwrite(
                    'result/patch/rgb/mouth/normal/Patch/' + aod + '_NEW' + '_' + patch_name + '_' + pic_name
                    + '_' + str(epoch) + '_LR= ' + str(learning_rate) + '.jpg', storePatch[:, :, [2, 1, 0]])

        if epoch == 60:
            learning_rate = learning_rate * 0.1
        if epoch == 80:
            learning_rate = learning_rate * 0.1


def face_Control_all(learning_rate):
    storePatch = getRGB(var_patch).numpy()

    storePatch = storePatch.astype(np.int32)
    storePatch = storePatch.astype(np.float32)

    cv2.imwrite(
        './Face_Control/results/patches/' + aod + '_INIT' + patch_name + '_LR= ' + str(learning_rate) + '.jpg', storePatch[:, :, [2, 1, 0]])

    for epoch in range(21):
        # One meter to five meter (only 1 meter)
        for distance in range(1, 2):  # CHANGED
            pictureList = os.listdir('./Face_Control/0--Parade')
            true_box_info = np.load('./Face_Control/0--Parade/info_me.npy', allow_pickle=True)
            pn = 0  # picture count(how many images already passed)

            for pic in pictureList:
                pn += 1

                pic_name = pic.split('.')[0]
                if pic_name == 'info_me':  # CHANGED
                    break
                # which_pic = int(pic_name[0]) - 1

                image = tf.keras.preprocessing.image.load_img('./Face_Control/0--Parade/' + pic)
                image = tf.keras.preprocessing.image.img_to_array(image)

                result = detector.detect_faces(image)
                image = tf.cast(image, dtype=tf.float32)

                if len(result) == 1:  # With this bounding_boxes is always 2 dimensional
                    bounding_boxes = np.array([result[0]['box']])
                else:
                    bounding_boxes = np.empty((0, 4), int)  # CHANGE I'm not sure if using np.empty is safe

                    for j in range(len(result)):
                        bounding_boxes = np.append(bounding_boxes, np.array([result[j]['box']]), axis=0)

                bounding_boxes = tf.cast(bounding_boxes, dtype=tf.float32)

                # 1M scales[5:8]
                # 2M scales[4:7]
                # 3M scales[2:5]
                # 4M scales[1:4]
                # 5M scales[0:3]
                scales = np.load('scales.npy')
                scaleStartIndex = [5, 4, 2, 1, 0]

                print("*****")
                print("Epoch: ", epoch, "of Picture:", pic_name, "Image_NR:",
                      pn)  # , "of Directory", str(distance) + "M")
                # Pick three scale to train for each distance
                for scaleNum in range(1):  # CHANGED 3 -> 1
                    # print('start', " Scalenum: ", scaleNum)
                    #TODO
                    #opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)  # learning_rate[scaleNum])
                    which_scale = scaleStartIndex[distance - 1] + scaleNum

                    scale = scales[which_scale]

                    label = createLabel(image, scale)

                    '''
                    Preparing the image before the tape.
                    '''
                    normalized_image = (image - 127.5) * 0.0078125  # NORMALIZING the image used

                    attacked_image = iterative_attack(normalized_image, label, bounding_boxes, scale,
                                                      learning_rate, scaleNum)
            if epoch % 10 == 0:
                '''
                image3 = attacked_image
                image3 = getRGB(image3).numpy()
                image3 = image3.astype(np.int32)
                image3 = image3.astype(np.float32)

                results = detector.detect_faces(image3)
                '''
                storePatch = getRGB(var_patch).numpy()

                storePatch = storePatch.astype(np.int32)
                storePatch = storePatch.astype(np.float32)

                # TODO
                '''
                plt.figure(figsize=(image3.shape[1] * 6 // image3.shape[0], 6))
                plt.imshow(image3 / 255)
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

                iou = round(len(iou_mat[np.where(iou_mat == 2)]) / len(iou_mat[np.where(iou_mat > 0)]), 2)

                plt.savefig(
                    './Face_Control/results/images' + aod + '_NEW' + '_' + patch_name + '_' + pic_name + '_iou=' + str(
                        iou) + '_' + str(i) + '_LR= ' + str(learning_rate) + '.jpg')
                '''

                cv2.imwrite(
                    './Face_Control/results/patches/' + aod + '_NEW' + '_' + patch_name
                    + '_' + str(epoch) + '_LR= ' + str(learning_rate) + '.jpg', storePatch[:, :, [2, 1, 0]])

        if epoch == 60:
            learning_rate = learning_rate * 0.1
        if epoch == 80:
            learning_rate = learning_rate * 0.1


def face_control_variable(learning_rate):
    # For now, we need to load the names of the images and their ground truths only ones, sinc at the momente we only use one directory
    img_folder = './Face_Control'
    labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")
    pictureList = []  # names of the images
    ground_truths = {}  # dictionary of ground truths
    pCount = 0  # how many images we are using
    while labels:
        #TODO pCount througths an error if pCount = 66
        if pCount == 65:  # number of images being processed
            break

        pCount += 1
        img_name = labels.readline().rstrip("\n")
        if img_name == "":
            labels.close()
            break

        pictureList.append(img_name)
        ground_truth_count = int(labels.readline().rstrip("\n"))

        ground_truths[img_name] = []

        for i in range(ground_truth_count):
            ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][
                                           0:4])  # take only first 4 values for box size

    storePatch = getRGB(var_patch).numpy()

    storePatch = storePatch.astype(np.int32)
    storePatch = storePatch.astype(np.float32)

    cv2.imwrite(
        './Face_Control/results/patches/' + aod + '_' + patch_name + '_INIT' + '_LR= ' + str(learning_rate)
        + '_PCount=' + str(pCount) + '.jpg', storePatch[:, :, [2, 1, 0]])

    for epoch in range(121):
        # One meter to five meter (only 1 meter)
        for distance in range(1, 2):  # For now, this stands how it is, since later we want to use multiple directories

            pn = 0
            for pic in pictureList:
                pn += 1

                pic_name = pic.split('.')[0]

                image = tf.keras.preprocessing.image.load_img(img_folder + "/" + pic)
                image = tf.keras.preprocessing.image.img_to_array(image)

                ground_truths_of_pic = tf.cast(ground_truths[pic], dtype=tf.float32)  # ground truths of the image

                # 1M scales[5:8]
                # 2M scales[4:7]
                # 3M scales[2:5]
                # 4M scales[1:4]
                # 5M scales[0:3]
                scales = np.load('scales.npy')
                scaleStartIndex = [5, 4, 2, 1, 0]

                print("*****")
                print("Epoch:", epoch, "of Picture:", pic_name, "Image_NR:",
                      pn)  # , "of Directory", str(distance) + "M")
                # Pick three scale to train for each distance
                for scaleNum in range(1):  # CHANGED 3 -> 1
                    # print('start', " Scalenum: ", scaleNum)
                    #TODO
                    #opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)  # learning_rate[scaleNum])
                    which_scale = scaleStartIndex[distance - 1] + scaleNum

                    scale = scales[which_scale]

                    label = createLabel(image, scale)

                    '''
                    Preparing the image before the tape.
                    '''
                    normalized_image = (image - 127.5) * 0.0078125  # NORMALIZING the image used

                    attacked_image = iterative_attack(normalized_image, label, ground_truths_of_pic, scale,
                                                      learning_rate, scaleNum)
            if epoch % 10 == 0:
                '''
                image3 = attacked_image
                image3 = getRGB(image3).numpy()
                image3 = image3.astype(np.int32)
                image3 = image3.astype(np.float32)

                results = detector.detect_faces(image3)
                '''
                storePatch = getRGB(var_patch).numpy()

                storePatch = storePatch.astype(np.int32)
                storePatch = storePatch.astype(np.float32)

                # TODO
                '''
                plt.figure(figsize=(image3.shape[1] * 6 // image3.shape[0], 6))
                plt.imshow(image3 / 255)
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

                iou = round(len(iou_mat[np.where(iou_mat == 2)]) / len(iou_mat[np.where(iou_mat > 0)]), 2)

                plt.savefig(
                    './Face_Control/results/images' + aod + '_NEW' + '_' + patch_name + '_' + pic_name + '_iou=' + str(
                        iou) + '_' + str(i) + '_LR= ' + str(learning_rate) + '.jpg')
                '''

                cv2.imwrite(
                    './Face_Control/results/patches/' + aod + '_NEW' + '_' + patch_name + '_' + str(i)
                    + '_LR= ' + str(learning_rate) + '_PCount=' + str(pCount) + '.jpg', storePatch[:, :, [2, 1, 0]])

        if epoch == 60:
            learning_rate = learning_rate * 0.1
        if epoch == 80:
            learning_rate = learning_rate * 0.1


# TODO
ascent_or_descent = 1  # -1 = ascent and +1 = descent [I think]
if ascent_or_descent == 1:
    aod = 'DESCENT'
elif ascent_or_descent == -1:
    aod = 'ASCENT'

allFileList = os.listdir('./patch_img')
for file in allFileList:
    '''
    # ATTENTION THE PATCH IMAGES AREN'T USED RIGHT KNOW
    patch_name = file.split('.')[0]
    patch = tf.keras.preprocessing.image.load_img('patch_img/' + file)
    patch = tf.keras.preprocessing.image.img_to_array(patch)
    '''

    '''
    Random patch is used right know

    Right now, I'm trying to normalize our RGB patch to their relative numbers between
    -1 and 1, so I could use the float type without loosing information.
    '''
    patch_name = "RANDOM"
    patch = np.random.randint(255, size=(128, 128, 3))  # RANDOM PATCH
    patch = (patch - 127.5) * 0.0078125  # NORMALIZING of the patch
    var_patch = tf.Variable(patch, dtype=tf.float32)  # RANDOM PATCH to TF VARIABLE

    # TODO
    # Learning rate for having for each scale factore one special learning rate
    learning_rate = np.array([1, 1, 1])  # [1.0, 1.0, 1.0]
    # Use lr for testing
    lr = 100  # learning rate/ amplification rate used right know
    opt = tf.keras.optimizers.SGD(learning_rate=lr)  # learning_rate[scaleNum])
    #picture_images(lr)
    #face_Control_all(lr)
    face_control_variable(lr)
