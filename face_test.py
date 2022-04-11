import cv2
from mtcnn import original_MTCNN
import numpy as np
import copy
import math
import os
import shutil

original_detector = original_MTCNN()

# loads the images into an array
img_folder = "Test_Faces"
patch_folder = "to_test"
results_folder = "test_results"

images = []  # List of image data used for testing patches
image_names = []  # The file names of images
ground_truths = {}  # The ground truths of faces in loaded images

labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")

n = 0
# Loading images, their names and their ground truths
while labels:
    if n == 0:
        for _ in range(52762):  # skip to the dataset relevant to us - this is the line number where ground truths for our selected dataset start, within the labels file
            next(labels)
    if n == 994:  # number of photos processed
        break

    n += 1
    img_name = labels.readline().rstrip("\n")  # reads image name from list of labels

    if img_name == "":  # stop if we reach the end of the labels file
        labels.close()
        break

    image_names.append(img_name)
    print(str(n), img_folder + "/" + img_name)
    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
    ground_truth_count = int(labels.readline().rstrip("\n"))

    ground_truths[img_name] = []

    if not ground_truth_count:
        next(labels)
    else:
        for i in range(ground_truth_count):
            ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()])

alpha = 0.5  # to match paper's concrete parameters
lambda_IoU = 0.6  # To match paper's concrete parameters


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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # MTCNN doesn't give the end points in [2] and [3] but the distance from the start point to the end point
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def output_image(bounding_boxes_of_image, attacked_image, name_of_image, patch_results_folder
                 , ground_truths_of_image):
    """
    Creates an image file for the given image, on which the patch is added to each face and the bounding boxes for ground truths and
    detected faces are marked in green and blue, respectively.

    :param bounding_boxes_of_image: list
        representing the detected bounding boxes (faces) by MTCNN, which passed the IoU test with the ground truths
    :param attacked_image: nd-array
        image data on which the patch is added to each face
    :param name_of_image: str
        representing the name of the image
    :param patch_results_folder: str
        representing the relative path where the image is placed
    :param ground_truths_of_image: list
        ground truth bounding boxes for faces on the image
    :return:

        bboxes: str
            stating all found bounding boxes in the image
        face_nr: int
            stating how many faces were found
        over_face_nr: int
            stating how many found faces had a confidence score above or equal to 0.99 (were unsuccessfully attacked).
    """

    working_image = copy.deepcopy(attacked_image)  # don't modify the original pictures

    # Number of found faces (bounding boxes) in the image
    face_nr = 0
    # Number of found faces (bounding boxes) with confidence score bigger than or equal to 0.99. These are the faces, which were unsuccessfully attacked
    over_face_nr = 0
    # String containing all found bounding boxes
    bboxes = ""

    for ground_truth_bounding_box in ground_truths_of_image:  # ground truth loop

        cv2.rectangle(working_image,
                      (ground_truth_bounding_box[0], ground_truth_bounding_box[1]),
                      (ground_truth_bounding_box[0] + ground_truth_bounding_box[2],
                       ground_truth_bounding_box[1] + ground_truth_bounding_box[3]),
                      (0, 255, 0),
                      2)

    highest_confidence_score = 0
    for bb in bounding_boxes_of_image:
        face_nr += 1
        bounding_box = bb['box']
        confidence_score = round(bb['confidence'], 3)  # extract confidence score of detected bounding box

        if confidence_score >= 0.99:
            over_face_nr += 1

        print(name_of_image + " has", bounding_box, "with " + str(confidence_score))
        bboxes += str(name_of_image + " has ") + str(bounding_box) + " with " + str(confidence_score) + "\n"
        cv2.rectangle(working_image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        cv2.putText(working_image, str(confidence_score), (bounding_box[0], bounding_box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)  # places confidence score over the detected face boxes

        if confidence_score > highest_confidence_score:
            highest_confidence_score = confidence_score

    # The highest found confidence score in the image
    prefix = str(highest_confidence_score)

    # Creates working_image in which the faces are marked through their bounding boxes
    # HCS (for highest confidence score in the image) and PN (for picture name)
    # Code below is used for writing the output files. This depends on the labels file you use, so adjust where necessary!
    try:
        # Used if the ground truths were taken from wider_face_train_bbx_gt.txt
        cv2.imwrite(
            img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + "HCS=" + prefix + "_PN="
            + name_of_image.split('/')[1], cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR))
    except:
        # Used if the ground truths were taken from labels.txt
        cv2.imwrite(
            img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + "HCS=" + prefix + "_PN=" + name_of_image
            , cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR))

    return bboxes, face_nr, over_face_nr


def apply_patch_multiple(originals, adversarial_patch):
    """
    Applies the adversarial patch onto the original pictures (MULTIPLE)

    :param originals: list
        images on which the adversarial patch is applied to
    :param adversarial_patch: nd-array
        the adversarial patch
    :return: list
        the original images but with the patch placed, relative to the ground truths
    """

    working_images = copy.deepcopy(originals)  # don't modify the original pictures
    for image_nr in range(len(working_images)):  # loop through the copied pictures

        ground_truth_boxes = ground_truths[image_names[image_nr]]  # pull up ground_truth data for this picture
        adv_img_rows = working_images[image_nr].shape[0]
        adv_img_cols = working_images[image_nr].shape[1]

        # draw detected face + plaster patch over source
        for bounding_box in ground_truth_boxes:  # ground truth loop
            resize_value = round(alpha * math.sqrt(bounding_box[2] * bounding_box[
                3]))  # as per the paper (pg. 7 of "Design and Interpretation of Universal Adversarial Patches in Face Detection")

            if resize_value == 0:
                continue

            resized_P = cv2.resize(adversarial_patch, (resize_value,
                resize_value))  # as per the paper (pg. 7 of "Design and Interpretation of Universal Adversarial Patches in Face Detection")

            x_P = round(bounding_box[2] / 2)  # as per the paper, x_P is in the center position of the bounding box
            y_P = round(resize_value / 2)  # as per the paper, y_P is in the center position of the bounding box

            '''Finding the indices where to put the patch'''
            y_start = y_P + bounding_box[1] - round(resized_P.shape[0] / 2)
            x_start = x_P + bounding_box[0] - round(resized_P.shape[1] / 2)

            y_end = y_start + resized_P.shape[0]
            x_end = x_start + resized_P.shape[1]

            '''If the bounding box is outside the image'''
            if x_start < 0:
                x_end -= x_start
                x_start = 0
            if y_start < 0:
                y_end -= y_start
                y_start = 0

            if x_end > adv_img_cols:
                x_start -= x_end - adv_img_cols
                x_end = adv_img_cols
            if y_end > adv_img_rows:
                y_start -= y_end - adv_img_rows
                y_end = adv_img_rows

            # draw patch over source image, by specifying the coordinates to overwrite
            working_images[image_nr][y_start:y_end, x_start:x_end] = resized_P

    return working_images

def apply_patch_one(original, adversarial_patch, ground_truth_boxes):
    """
    Applies the adversarial patch onto the original picture (ONLY ONE)

    :param original: list
        image on which the adversarial patch is applied to
    :param adversarial_patch: nd-array
        the adversarial patch
    :param ground_truths: list
        ground_truths of the image
    :return: list
        the original image but with the patch placed, relative to the ground truths
    """

    working_image = copy.deepcopy(original)  # don't modify the original picture
    adv_img_rows = working_image.shape[0]
    adv_img_cols = working_image.shape[1]

    # draw detected face + plaster patch over source
    for bounding_box in ground_truth_boxes:  # ground truth loop
        resize_value = round(alpha * math.sqrt(bounding_box[2] * bounding_box[
            3]))  # as per the paper (pg. 7 of "Design and Interpretation of Universal Adversarial Patches in Face Detection")

        if resize_value == 0:
            continue

        resized_P = cv2.resize(adversarial_patch, (resize_value,
                                                   resize_value))  # as per the paper (pg. 7 of "Design and Interpretation of Universal Adversarial Patches in Face Detection")

        x_P = round(bounding_box[2] / 2)  # as per the paper, x_P is in the center position of the bounding box
        y_P = round(resize_value / 2)  # as per the paper, y_P is in the center position of the bounding box

        '''Finding the indices where to put the patch'''
        y_start = y_P + bounding_box[1] - round(resized_P.shape[0] / 2)
        x_start = x_P + bounding_box[0] - round(resized_P.shape[1] / 2)

        y_end = y_start + resized_P.shape[0]
        x_end = x_start + resized_P.shape[1]

        '''If the bounding box is outside the image'''
        if x_start < 0:
            x_end -= x_start
            x_start = 0
        if y_start < 0:
            y_end -= y_start
            y_start = 0

        if x_end > adv_img_cols:
            x_start -= x_end - adv_img_cols
            x_end = adv_img_cols
        if y_end > adv_img_rows:
            y_start -= y_end - adv_img_rows
            y_end = adv_img_rows

        # draw patch over source image, by specifying the coordinates to overwrite
        working_image[y_start:y_end, x_start:x_end] = resized_P

    return working_image


def run_test_prepared(attacked_imgs, patch_results_folder):
    """
    Tests an adversarial patch on given already prepared images, i.e. the patch has already been applied to the images.

    :param attacked_imgs: list
        images on which the patch was applied to
    :param patch_results_folder: str
        for the relative path where test results will be placed
    :return:
        None
    """

    # creates a new folder in which the test results will be placed (if it exists it will overwrite it)
    if os.path.exists(img_folder + "/" + results_folder + "/" + patch_results_folder):
        shutil.rmtree(img_folder + "/" + results_folder + "/" + patch_results_folder)
    os.makedirs(img_folder + "/" + results_folder + "/" + patch_results_folder)

    # string containing all found bounding boxes
    bboxes = ""

    # how many faces (ground truths) there are in total
    total_faces = 0

    # number of found faces (bounding boxes) in the image
    face_nr = 0

    # number of found faces (bounding boxes) with confidence score is bigger than or equal to 0.99. these are the faces, which were unsuccessfully attacked
    over_face_nr = 0

    # number of images where faces were found
    total_image_nr = 0

    for image_nr in range(len(attacked_imgs)):
        # bounding boxes marking faces
        results = []
        print("Testing on:", image_nr, image_names[image_nr])
        try:  # handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them

            tmp_results = original_detector.detect_faces(attacked_imgs[image_nr])

            # only bounding boxes representing faces are relevant
            for bb in tmp_results:
                for ground_truth_bounding_box in ground_truths[image_names[image_nr]]:
                    # IoU is applied to guarantee that the bounding box marks a face
                    if IoU(bb['box'], ground_truth_bounding_box) > lambda_IoU:
                        results.append(bb)

        except Exception as e:  # if an exception is met store empty result and image without patch applied to it
            print("Image", image_nr, image_names[image_nr], " (skipping) has the following error:", e)
            continue

        if len(results):
            total_image_nr += 1

        # output face detection results for images with the adversarial patch applied
        tmp_bboxes, tmp_face_nr, tmp_over_face_nr = output_image(results, attacked_imgs[image_nr], image_names[image_nr]
                                                                 , patch_results_folder
                                                                 , ground_truths[image_names[image_nr]])

        bboxes += tmp_bboxes
        face_nr += tmp_face_nr
        over_face_nr += tmp_over_face_nr
        total_faces += len(ground_truths[image_names[image_nr]])

    # create a short descriptive .txt file in the folder, giving additional details for the test results
    top_text = "In " + str(total_image_nr) + " of " + str(len(attacked_imgs)) + " images " \
               + str(face_nr) + " of " + str(total_faces) + " faces were found. \n"
    bboxes = top_text + str(over_face_nr) + " of those faces had a confidence score >= 0.99. \n \n" + bboxes

    with open(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + str(face_nr) + "_faces_in_" + str(
            total_image_nr) + "_images.txt", "x") as f:
        f.write(bboxes)

    return


def run_test_unprepared(original_imgs, adversarial_patch, patch_results_folder):
    """
    Tests an adversarial patch on given unprepared images. In serial the patch is added to the

    :param original_imgs: list
        images on which the patch will be applied to
    :param  adversarial_patch: nd-array
        the adversarial patch
    :param patch_results_folder: str
        for the relative path where test results will be placed
    :return:
        None
    """

    # creates a new folder in which the test results will be placed (if it exists it will overwrite it)
    if os.path.exists(img_folder + "/" + results_folder + "/" + patch_results_folder):
        shutil.rmtree(img_folder + "/" + results_folder + "/" + patch_results_folder)
    os.makedirs(img_folder + "/" + results_folder + "/" + patch_results_folder)

    # string containing all found bounding boxes
    bboxes = ""

    # how many faces (ground truths) there are in total
    total_faces = 0

    # number of found faces (bounding boxes) in the image
    face_nr = 0

    # number of found faces (bounding boxes) with confidence score is bigger than or equal to 0.99. these are the faces, which were unsuccessfully attacked
    over_face_nr = 0

    # number of images where faces were found
    total_image_nr = 0

    for image_nr in range(len(original_imgs)):
        # bounding boxes marking faces
        results = []
        print("Testing on:", image_nr, image_names[image_nr])
        try:  # handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them

            attacked_image = apply_patch_one(original_imgs[image_nr], adversarial_patch
                                             , ground_truths[image_names[image_nr]])

            tmp_results = original_detector.detect_faces(attacked_image)

            # only bounding boxes representing faces are relevant
            for bb in tmp_results:
                for ground_truth_bounding_box in ground_truths[image_names[image_nr]]:
                    # IoU is applied to guarantee that the bounding box marks a face
                    if IoU(bb['box'], ground_truth_bounding_box) > lambda_IoU:
                        results.append(bb)

        except Exception as e:  # if an exception is met store empty result and image without patch applied to it
            print("Image", image_nr, image_names[image_nr], " (skipping) has the following error:", e)
            continue

        if len(results):
            total_image_nr += 1

        # output face detection results for images with the adversarial patch applied
        tmp_bboxes, tmp_face_nr, tmp_over_face_nr = output_image(results, attacked_image, image_names[image_nr]
                                                                 , patch_results_folder
                                                                 , ground_truths[image_names[image_nr]])

        bboxes += tmp_bboxes
        face_nr += tmp_face_nr
        over_face_nr += tmp_over_face_nr
        total_faces += len(ground_truths[image_names[image_nr]])

    # create a short descriptive .txt file in the folder, giving additional details for the test results
    top_text = "In " + str(total_image_nr) + " of " + str(len(original_imgs)) + " images " \
               + str(face_nr) + " of " + str(total_faces) + " faces were found. \n"
    bboxes = top_text + str(over_face_nr) + " of those faces had a confidence score >= 0.99. \n \n" + bboxes

    with open(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + str(face_nr) + "_faces_in_" + str(
            total_image_nr) + "_images.txt", "x") as f:
        f.write(bboxes)

    return


for patch_name in os.listdir(img_folder + "/" + patch_folder): # load and perform testing on each patch located in the local patch folder
    print("\n" + "Testing Patch " + patch_name)

    patch = cv2.cvtColor(cv2.imread((img_folder + "/" + patch_folder + "/" + patch_name)), cv2.COLOR_BGR2RGB)
    """
    # When applying the patch to all images at ones
    attacked_images = apply_patch_multiple(images, patch)
    run_test_prepared(attacked_images, patch_name)
    """
    # When applying the patch to one image at a time [Saving memory]
    run_test_unprepared(images, patch, patch_name)

    cv2.imwrite(img_folder + "/" + results_folder + "/" + patch_name + "/" + patch_name,
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
