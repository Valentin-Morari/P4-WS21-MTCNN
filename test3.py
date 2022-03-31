import tensorflow as tf
import numpy as np
import mtcnn
from mtcnn.network.factory import NetworkFactory
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as mpatches
import os

detector = mtcnn.MTCNN()


def tf_scale_image(img, scale):
    """
    This function was taken from MTCNN  (__scale_image) and ported to tensorflow 2.x

    :param img: Image to resized
    :param scale: Scale factor
    :return: Given image scaled with the specified scale factor
    """
    height = img.shape[0]
    width = img.shape[1]

    width_scaled = tf.cast(tf.math.ceil(width * scale), dtype=tf.int32)
    height_scaled = tf.cast(tf.math.ceil(height * scale), dtype=tf.int32)

    # TensorFlow first take the height and than the width. CV2 does the opposite
    scaled_image = tf.image.resize(img, [height_scaled, width_scaled], method='lanczos5', antialias=True)

    # Normalizing the RGB values to be between -1 and 1
    scaled_image_normalized = (scaled_image - 127.5) * 0.0078125

    return scaled_image_normalized


def tf_rerec(bbox):
    """
    This function was taken from MTCNN  (__rerec) and ported to tensorflow 2.x
    It reshapes the bounding box to a square by elongating the shorter sides.

    :param bbox: Bounding box
    :return: The bounding box converted to a square
    """

    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]

    max_side_length = tf.maximum(width, height)

    v21 = bbox[:, 0] + width * 0.5 - max_side_length * 0.5  # This should be x of a bounding box
    v22 = bbox[:, 1] + height * 0.5 - max_side_length * 0.5  # This should be y of a bounding box
    v23 = bbox[:, 0:2] + tf.transpose(tf.tile([max_side_length], [2, 1]))
    confidences = bbox[:, 4]  # These are the confidence scores

    # Combining the changes back to one bounding box. The type is tf.float, since the confidence score is a float type
    # and a tensor can only have one type.
    bbox = tf.stack([v21, v22, v23[:, 0], v23[:, 1], confidences], axis=1)

    return bbox


def loss_object(predict_boxes, ground_truth_boxes):
    """
    Before the confidence scores are put into the loss function, IoU needs to be applied to their bounding boxes
    with the ground truths of the image, to make sure that the bounding box is from a face.
    The loss function was given by the paper.

    :param predict_box: Predicted bounding boxes in the image with confidence score
    :param ground_truth_boxes: Ground truths of the image
    :return: Result of the loss stated in the paper for Patch_IoU
    """

    # IoU for each bounding box with every ground truth
    iou_probe = tf_IoU_multiple_boxes(predict_boxes, ground_truth_boxes)

    if tf.size(iou_probe) == 0:
        # With this the loss is inf, which results into a None gradient from which the system won't learn
        confidence_scores = tf.cast(np.array([0]), dtype=tf.float32)
    else:
        confidence_scores = iou_probe[:, 4]

    loss = tf.negative(tf.divide(tf.math.reduce_sum(tf.math.log(confidence_scores)),
                                 len(confidence_scores)))

    return loss


def createPnet():
    """
    Sets a PNET taken from MTCNN with the weights from MTCNN

    :return: PNET model with applied weights
    """

    # Loading weights from MTCNN for PNET
    weight_file = 'mtcnn_weights.npy'
    weights = np.load(weight_file, allow_pickle=True).tolist()
    pnet = NetworkFactory().build_pnet()
    pnet.set_weights(weights['pnet'])
    return pnet


pnet_attacked = createPnet()


def imageChangeToFitPnet(image, scale):
    """
    This code originated from MTCNN and was part of the start of the for scales loop in __stage1.
    It was ported to TensorFlow 2.x
    The image is resized and normalized. Then, its dimensions are expanded and then transposed.

    :param image: Image to be put into PNET
    :param scale: Scale factor
    :return: Image prepared for PNET
    """
    # Scaling and normalising the image
    scaled_image = tf_scale_image(image, scale)

    img_x = tf.expand_dims(scaled_image, 0)
    img_y = tf.transpose(img_x, (0, 2, 1, 3))
    return img_y



def tf_IoU(boxA, boxB):
    """
    Does Intersection over Union with the two bounding boxes

    :param boxA: Bounding box
    :param boxB: Bounding box
    :return: Result of the Intersection over Union from boxA and boxB
    """
    xA = tf.math.maximum(boxA[0], boxB[0])
    yA = tf.math.maximum(boxA[1], boxB[1])

    xB = tf.math.minimum(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = tf.math.minimum(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = tf.math.maximum(0, xB - xA + 1) * tf.math.maximum(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    # print("IOU=", iou)
    return iou


def tf_IoU_multiple_boxes(bounding_boxes, ground_truths):
    """
    Does Intersection over Union on each bounding box with each ground truth

    :param bounding_boxes: All found bounding boxes in the image found by PNET (Should surround faces)
    :param ground_truths: Ground truths of the image (True positions of the faces in an image)
    :return: Tensor array containing bounding boxes with their confidence scores. Each of the bounding boxes
    had an IoU value of more than 0.6 with one of the ground truths.
    """
    iou_result = []

    # IoU between each bounding box between every ground truth
    for bb in bounding_boxes:
        iou = 0

        for gt in ground_truths:

            # IoU between one bounding box and one ground truth
            new_iou = tf_IoU(bb[0:4], gt)

            if new_iou > iou:
                iou = new_iou

        # Only bounding boxes with an IoU bigger than 0.6 are used to train the patch like in the paper
        if iou >= 0.6:
            iou_result.append(bb)
    tf_iou_result = tf.cast(iou_result, dtype=tf.float32)

    return tf_iou_result


def tf_generate_bounding_box(imap, reg, scale, t):
    """
    This function was taken from MTCNN  (__generate_bounding_box) and mostly ported to tensorflow 2.x
    It uses heatmap to generate bounding boxes

    :param imap:
    :param reg:
    :param scale: Scale with which the image was resized
    :param t: The steps threshold
    :return: Bounding boxes with their confidence score and
    """

    stride = 2
    cellsize = 12

    imap = tf.transpose(imap)

    dx1tf = tf.transpose(reg[:, :, 0])
    dy1tf = tf.transpose(reg[:, :, 1])
    dx2tf = tf.transpose(reg[:, :, 2])
    dy2tf = tf.transpose(reg[:, :, 3])

    positives = tf.where(tf.greater_equal(imap, t))

    """
    # This code part of the function __generate_bounding_box isn't ported yet to TensorFlow
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    """

    scoretf = tf.gather_nd(imap, positives)

    regtf = tf.concat([[tf.gather_nd(dx1tf, positives)], [tf.gather_nd(dy1tf, positives)],
                       [tf.gather_nd(dx2tf, positives)], [tf.gather_nd(dy2tf, positives)]],
                      axis=0)  # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn
    regtf = tf.transpose(regtf)

    if tf.size(regtf) == 0:
        regtf = tf.cast(np.empty(shape=(0, 3)), dtype=tf.float32)

    bbtf = positives

    q1tf = tf.experimental.numpy.fix((stride * tf.cast(bbtf, dtype=tf.float32) + 1.) / scale)
    q2tf = tf.experimental.numpy.fix((stride * tf.cast(bbtf, dtype=tf.float32) + cellsize) / scale)

    boundingboxtf = tf.concat([q1tf, q2tf, tf.expand_dims(scoretf, 1), regtf],
                              axis=1)  # tf.concat(value, axis=1) is counterpart of the numpy function np.hstack() in Tensorflow

    return boundingboxtf, regtf


# Second half of __stage1
def further_process_pnet_result(total_boxes):
    """
    This code originated from MTCNN and represents the second half of __stage1.
    It was ported to TensorFlow 2.x

    It applies once more Non Maximum Suppression (NMS) on all the bounding boxes from every scale
    to have the best bounding boxes of all the bounding boxes.

    :param total_boxes: All bounding boxes, which were found in an scales loop and
            survived Non Maximum Suppression(NMS)
    :return: Bounding boxes, which should mark faces and have a higher score
            [It would be the result of __stage1]
    """

    numboxes = total_boxes.shape[0]

    if numboxes > 0:
        # Using a high maxOutputSize, since MTCNN returns many boxes. Set threshold set like in MTCNN.
        tf_pick = tf.image.non_max_suppression(total_boxes[:, 0:4], total_boxes[:, 4], 100000, iou_threshold=0.7)
        total_boxes = tf.gather(total_boxes, indices=tf_pick)

        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]

        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = tf.transpose(tf.concat([[qq1], [qq2], [qq3], [qq4], [total_boxes[:, 4]]],
                                             axis=0))  # tf.concat(value, axis=0) is counterpart of the numpy function np.vstack() in Tensorflow and every element is put into brackets to be like the original from mtcnn

        total_boxes = tf_rerec(tf.identity(total_boxes))

        total_boxes_03 = tf.experimental.numpy.fix(total_boxes[:, 0:4])
        total_boxes_4 = total_boxes[:, 4]
        total_boxes = tf.stack(
            [total_boxes_03[:, 0], total_boxes_03[:, 1], total_boxes_03[:, 2], total_boxes_03[:, 3], total_boxes_4],
            axis=1)

    return total_boxes


def create_adversarial_pattern(target_image, ground_truth_boxes, scales_pyramid, steps_threshold):
    """
    Creates an gradient for the patch by adding the patch to the image and executing similar code
    to the function __stage1 in MTCNN. I.e. it tapes __stage1 from MTCNN and tries to generate a
    gradient, which will be applied to the patch to worsen the performance of __stage1 and therefore the
    performance of MTCNN.

    :param target_image: Image which is used to train the patch
    :param ground_truth_boxes: Ground truths of the image
    :param scales_pyramid The scales_pyramid for the creation of the image pyramid of the image
    :param steps_threshold: The steps threshold
    :return: The gradient for the patch and the loss for the image
    """
    with tf.GradientTape() as tape:
        tape.watch(var_patch)

        patched_image = tf_apply_patch(target_image, var_patch, ground_truth_boxes)
        if patched_image is None:
            print("ATTENTION PATCH WASN'T ADDED TO THE IMAGE!!!")
            print(ground_truth_boxes)

        total_boxes = tf.reshape(tf.convert_to_tensor(()), (0, 9))
        iterator = 0
        for scale in scales_pyramid:
            adv_image = imageChangeToFitPnet(patched_image, scale)

            out = pnet_attacked(adv_image)

            out0 = tf.transpose(out[0], perm=[0, 2, 1, 3])
            out1 = tf.transpose(out[1], perm=[0, 2, 1, 3])

            boxes, _ = tf_generate_bounding_box(out1[0, :, :, 1], out0[0, :, :, :], scale, steps_threshold[0])
            boxes = tf.cast(boxes, tf.float32)

            # Using a high maxOutputSize, since MTCNN returns many boxes. Threshold set like in MTCNN.
            tf_pick = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], 100000, iou_threshold=0.5)

            if tf.size(boxes) > 0 and tf.size(tf_pick) > 0:
                boxes = tf.gather(boxes, indices=tf_pick)
                total_boxes = tf.concat([total_boxes, boxes], axis=0)

            iterator += 1

        total_boxes = further_process_pnet_result(total_boxes)

        # Calculating the loss.
        """
        ATTENTION: The loss right now is infinite, since there are no bounding boxes marking a face because
                   the bounding boxes aren't generated properly to MTCNN yet.  
        """
        loss = ascent_or_descent * loss_object(total_boxes, ground_truth_boxes)
        print("LOSS: ")
        print(loss)

    gradient = tape.gradient(loss, var_patch)

    return gradient, loss


def tf_apply_patch(img, p, ground_truths_of_image):
    """
    Applies the patch to the image.

    :param img: The original image
    :param p: The adversarial patch
    :param ground_truths_of_image: Ground truths of the image, i.e. the marked faces
    :return: The original image but with the patch placed, dependet on where the ground truths are given
    """

    alpha = 0.5
    tf_adv_img = None

    # draw detected face + plaster patch over source
    for bounding_box in ground_truths_of_image:  # ground truth loop

        if tf_adv_img is None:
            tf_adv_img = img

        resize_value = tf.math.round(alpha * tf.math.sqrt(bounding_box[2] * bounding_box[3]))
        tf_resized_patch = tf.image.resize(p, (resize_value, resize_value), method='lanczos5', antialias=True)

        x_P = tf.math.round(bounding_box[2] / 2.0)
        y_P = tf.math.round(resize_value / 2.0)

        adv_img_rows = img.shape[0]
        adv_img_cols = img.shape[1]

        # Finding the indices where to put the patch
        y_start = tf.cast(y_P + bounding_box[1] - round(tf_resized_patch.shape[1] / 2.0),
                          dtype=tf.int32)  # bounding_box[0]
        x_start = tf.cast(x_P + bounding_box[0] - round(tf_resized_patch.shape[0] / 2.0),
                          dtype=tf.int32)  # bounding_box[1]

        y_end = tf.cast(y_P + bounding_box[1] - round(tf_resized_patch.shape[1] / 2.0) + tf_resized_patch.shape[
            1], dtype=tf.int32)
        x_end = tf.cast(x_P + bounding_box[0] - round(tf_resized_patch.shape[0] / 2.0) + tf_resized_patch.shape[
            0], dtype=tf.int32)

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

        tf_overlay = tf_resized_patch - img[y_start:y_end, x_start:x_end]
        tf_overlay_pad = tf.pad(tf_overlay, [[y_start, adv_img_rows - y_end], [x_start, adv_img_cols - x_end], [0, 0]])

        tf_adv_img = tf_adv_img + tf_overlay_pad

    if tf_adv_img == None:
        print("ATTENTION no GroundTruth!")
        print(ground_truths_of_image)
        tf_adv_img = img

    return tf_adv_img


def compute_scale_pyramid(m, min_layer, scale_factor):
    """
    This function was taken from MTCNN  (__generate_bounding_box)
    Prepares 12 scales (a Scale pyramid) with which the image will be resized and put into PNET.

    :param m: 12 / min_face_size
    :param min_layer: np.amin([height, width]) * m
    :param scale_factor: Tells how much each scale should be smaller than the other
    :return: Scale pyramid for the image
    """
    scales = []
    factor_count = 0

    while min_layer >= 12:
        scales += [m * np.power(scale_factor, factor_count)]
        min_layer = min_layer * scale_factor
        factor_count += 1

    return scales


def prepare_scales_for_pnet(img, scale_factor, min_face_size):
    """
    This code originated from MTCNN and represents the first half of detect_faces.
    It prepares scales (a scale pyramid), which will be used to resize the image
    to find different sized faces.

    :param img: RGB image put into MTCNN on which we will place the patch
    :param scale_factor: Tells how much each scale should be smaller than the other
    :param min_face_size: 20 set by MTCNN
    :return: Scale pyramid for the image
    """
    height, width, _ = img.shape

    m = 12 / min_face_size
    min_layer = np.amin([height, width]) * m

    scales_from_mtcnn = compute_scale_pyramid(m, min_layer, scale_factor)

    return scales_from_mtcnn


def iterative_attack(target_image, ground_truth_boxes, optimizer):
    """
    Creates an adversarial patch by executing similar code to the functions detect_faces and __stage1 in MTCNN. It tapes
    the execution of __stage1 to create a gradient, which it applies

    :param target_image: Image being used to train the patch
    :param ground_truth_boxes: Ground truth boxes from the image
    :param optimizer: The used Keras optimizer
    :return: Image on which the patch was placed
    """

    scale_factor = 0.709  # Set like mtcnn __init__
    steps_threshold = [0.6, 0.7, 0.7]  # Set like mtcnn __init__
    min_face_size = 20  # Set like mtcnn __init__

    scales_pyramid = prepare_scales_for_pnet(target_image, scale_factor, min_face_size)

    gradient, loss = create_adversarial_pattern(target_image, ground_truth_boxes, scales_pyramid, steps_threshold)

    if gradient is None:
        print("GRADIENT NONE")
    else:
        # Applies the gradient to the patch
        optimizer.apply_gradients(zip([gradient], [var_patch]))
        # Bound the values of the patch to be between 0 and 255
        var_patch.assign(tf.clip_by_value(var_patch, 0, 255))

    # Applies the new patch on the image to output the image
    adv_image = tf_apply_patch(target_image, var_patch, ground_truth_boxes)
    # Bound the values of the image to be between 0 and 255
    clipped_adv_image = tf.clip_by_value(adv_image, 0, 255)

    return clipped_adv_image


def picture_images(amp_factor):
    """
    This function uses the images from picture (which were taken from
    https://github.com/yahi61006/adversarial-attack-on-mtcnn) to train the adversarial patch. And returns
    the patch as well as the images on which the patch was applied as images.

    :param amp_factor: Amplification factor being used in the Keras SGD optimizer for the learning rate
    """

    # Used to output the patch at the start of the function call. To see the initial patch
    store_patch = var_patch.numpy()
    store_patch = store_patch.astype(np.int32)
    store_patch = store_patch.astype(np.float32)

    opt = tf.keras.optimizers.SGD(learning_rate=amp_factor)  # learning_rate[scaleNum])

    cv2.imwrite(
        'result/patch/rgb/mouth/normal/Patch/' + '_INIT' + '_' + patch_name + '_LR= ' + str(amp_factor) + '.jpg'
        , store_patch[:, :, [2, 1, 0]])

    # Going in each epoch through each image in each distance folder in pictures.
    # All images train the same adversarial patch
    for epoch in range(61):
        # One meter to five meter
        for distance in range(1, 6):
            picture_list = os.listdir('./picture/{}M/normal'.format(distance))
            true_box_info = np.load('./picture/{}M/normal/info.npy'.format(distance), allow_pickle=True)

            for pic in picture_list:

                pic_name = pic.split('.')[0]
                # This indicates the end of the folder
                if pic_name == 'info':
                    break
                which_pic = int(pic_name[0]) - 1

                image = tf.keras.preprocessing.image.load_img('./picture/{}M/normal/'.format(distance) + pic)
                image = tf.keras.preprocessing.image.img_to_array(image)

                # Using the results of MTCNN as the ground truths
                result = detector.detect_faces(image)
                image = tf.cast(image, dtype=tf.float32)

                # With this ground_truths_from_result is always 2 dimensional
                if len(result) == 1:
                    ground_truths_from_result = np.array([result[0]['box']])
                else:
                    # This is needed for placing the patch on all the faces, if an image has multiple faces
                    ground_truths_from_result = np.empty((0, 4), int)

                    for i in range(len(result)):
                        ground_truths_from_result = np.append(ground_truths_from_result, np.array([result[i]['box']])
                                                              , axis=0)

                ground_truths_from_result = tf.cast(ground_truths_from_result, dtype=tf.float32)

                # 1M scales[5:8]
                # 2M scales[4:7]
                # 3M scales[2:5]
                # 4M scales[1:4]
                # 5M scales[0:3]
                scales = np.load('scales.npy')
                scale_start_index = [5, 4, 2, 1, 0]

                print("*****")
                print("Epoch:", epoch, "of Picture:", pic_name, "of Directory", str(distance) + "M")

                attacked_image = iterative_attack(image, ground_truths_from_result, opt)

            if epoch % 10 == 0:
                image3 = attacked_image
                image3 = image3.numpy()
                image3 = image3.astype(np.int32)
                image3 = image3.astype(np.float32)

                results = detector.detect_faces(image3)

                store_patch = var_patch.numpy()
                store_patch = store_patch.astype(np.int32)
                store_patch = store_patch.astype(np.float32)

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
                        iou) + '_' + str(epoch) + '_LR= ' + str(amp_factor) + '.jpg')

                cv2.imwrite(
                    'result/patch/rgb/mouth/normal/Patch/' + aod + '_NEW' + '_' + patch_name + '_' + pic_name
                    + '_' + str(epoch) + '_LR= ' + str(amp_factor) + '.jpg', store_patch[:, :, [2, 1, 0]])

        if epoch == 60:
            amp_factor = amp_factor * 0.1
        if epoch == 80:
            amp_factor = amp_factor * 0.1


def face_control_variable(amp_factor):
    """
    This function uses the images from Face_Control (which were taken from the WIDER-FACES dataset)
    to train the adversarial patch. It outputs the adversarial patch into Face_Control/results/patches.

    :param amp_factor: Amplification factor being used in the Keras SGD optimizer for the learning rate
    """
    # For now, we need to load the names of the images and their ground truths only ones,
    # since we only use one directory
    img_folder = './Face_Control'
    labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")
    picture_list = []  # names of the images
    ground_truths = {}  # dictionary of ground truths
    image_count = 0  # how many images we are using
    opt = tf.keras.optimizers.SGD(learning_rate=amp_factor)  # learning_rate[scaleNum])

    # Loading the names of the images used into picture_list
    while labels:
        # number of images being processed
        if image_count == 65:# An error is thrown if pCount = 66
            break

        image_count += 1
        img_name = labels.readline().rstrip("\n")
        if img_name == "":
            labels.close()
            break

        picture_list.append(img_name)
        ground_truth_count = int(labels.readline().rstrip("\n"))

        ground_truths[img_name] = []

        for i in range(ground_truth_count):
            ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][
                                           0:4])  # take only first 4 values for box size

    '''Creating a file for the generated patch'''
    store_patch = var_patch.numpy()

    store_patch = store_patch.astype(np.int32)
    store_patch = store_patch.astype(np.float32)

    cv2.imwrite(
        './Face_Control/results/patches/' + aod + '_' + patch_name + '_INIT' + '_LR= ' + str(amp_factor)
        + '_PCount=' + str(image_count) + '.jpg', store_patch[:, :, [2, 1, 0]])

    for epoch in range(121):

        pn = 0
        for pic in picture_list:
            pn += 1

            pic_name = pic.split('.')[0]

            image = tf.keras.preprocessing.image.load_img(img_folder + "/" + pic)
            image = tf.keras.preprocessing.image.img_to_array(image)

            # In this function the original ground truths are used so this execution is not needed
            # result = detector.detect_faces(image)

            ground_truths_of_pic = tf.cast(ground_truths[pic], dtype=tf.float32)  # ground truths of the image

            # 1M scales[5:8]
            # 2M scales[4:7]
            # 3M scales[2:5]
            # 4M scales[1:4]
            # 5M scales[0:3]

            print("*****")
            print("Epoch:", epoch, "of Picture:", pic_name, "Image_NR:",
                  pn)  # , "of Directory", str(distance) + "M")
            # Pick three scale to train for each distance

            # The image with the applied patch is not returned yet
            attacked_image = iterative_attack(image, ground_truths_of_pic, opt)

        # Output the patch every 10 epochs
        if epoch % 10 == 0:
            store_patch = var_patch.numpy()

            store_patch = store_patch.astype(np.int32)
            store_patch = store_patch.astype(np.float32)

            cv2.imwrite(
                './Face_Control/results/patches/' + aod + '_NEW' + '_' + patch_name + '_' + str(i)
                + '_LR= ' + str(amp_factor) + '_PCount=' + str(image_count) + '.jpg', store_patch[:, :, [2, 1, 0]])

        if epoch == 60:
            amp_factor = amp_factor * 0.1

            opt = tf.keras.optimizers.SGD(learning_rate=amp_factor)  # changing the learning rate like in the paper
        if epoch == 80:
            amp_factor = amp_factor * 0.1

            opt = tf.keras.optimizers.SGD(learning_rate=amp_factor)  # changing the learning rate like in the paper


# -1 = descent and +1 = ascent. This was added since we weren't sure what would be stochastic gradient ascent
ascent_or_descent = 1
if ascent_or_descent == -1:
    aod = 'DESCENT'
elif ascent_or_descent == 1:
    aod = 'ASCENT'

# Loads the patches in patch_img into allFileList
allFileList = os.listdir('./patch_img')
# Attacking each patch separately
for file in allFileList:
    '''
    # ATTENTION THE PATCH IMAGES AREN'T USED RIGHT KNOW
    patch_name = file.split('.')[0]
    patch = tf.keras.preprocessing.image.load_img('patch_img/' + file)
    patch = tf.keras.preprocessing.image.img_to_array(patch)
    '''

    '''
    Random patch is used right know
    '''
    patch_name = "RANDOM"
    patch = np.random.randint(255, size=(128, 128, 3))  # RANDOM PATCH
    var_patch = tf.Variable(patch, dtype=tf.float32)  # RANDOM PATCH to TF VARIABLE

    # We call our learning rate "amplification factor", since it has a huge value
    amplification_factor = 100  # learning rate/ amplification rate used right know
    '''
    Functions for different image dataset to train the patch
    '''
    # picture_images(amplification_factor)
    face_control_variable(amplification_factor)
