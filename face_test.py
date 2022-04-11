import tensorflow as tf
import cv2
from mtcnn import MTCNN
import numpy as np
import copy
import os
import shutil

detector = MTCNN()

# Loads the images into an array
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
        confidence_score = round(bb['confidence'].numpy(), 3)  # extract confidence score of detected bounding box

        if confidence_score >= 0.99:
            over_face_nr += 1

        # Makes the bounding box visible
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


def run(patch_used, original_images, patch_results_folder):
    """
    Tests an adversarial patch on given images.

    :param patch_used: tensor of dtype=float32
        adversarial patch (a small image) to be tested by applying it onto faces for the purpose of reducing their detection rate
    :param original_images: list
        images on which the patch will be tested
    :param patch_results_folder: str
        for the relative path where test results will be placed
    :return:
        None
    """

    tmp_patch = patch_used

    # Creates a new folder in which the test results will be placed (if it exists it will overwrite it)
    if os.path.exists(img_folder + "/" + results_folder + "/" + patch_results_folder):
        shutil.rmtree(img_folder + "/" + results_folder + "/" + patch_results_folder)
    os.makedirs(img_folder + "/" + results_folder + "/" + patch_results_folder)

    # String containing all found bounding boxes
    bboxes = ""

    # How many faces (ground truths) there are in total
    total_faces = 0

    # Number of found faces (bounding boxes) in the image
    face_nr = 0

    # Number of found faces (bounding boxes) with confidence score is bigger than or equal to 0.99. these are the faces, which were unsuccessfully attacked
    over_face_nr = 0

    # Number of images where faces were found
    total_image_nr = 0

    for image_nr in range(len(original_images)):
        # Bounding boxes marking faces
        results = []
        print("Testing on:", image_nr, image_names[image_nr])
        try:  # handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them

            # amplification is 0, resulting in a gradient that's zero
            # therefore the patch won't be changed by our modified MTCNN instance for adversarial patch generation
            tmp_results, tmp_adv_img, new_patch = detector.new_detect_faces(original_images[image_nr], tmp_patch,
                                                                            ground_truths[image_names[image_nr]], 0)
            tmp_patch = new_patch

            # Only bounding boxes representing faces are relevant
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

        # Output face detection results for images with the adversarial patch applied
        tmp_bboxes, tmp_face_nr, tmp_over_face_nr = output_image(results, tmp_adv_img, image_names[image_nr]
                                                                 , patch_results_folder
                                                                 , ground_truths[image_names[image_nr]])

        bboxes += tmp_bboxes
        face_nr += tmp_face_nr
        over_face_nr += tmp_over_face_nr
        total_faces += len(ground_truths[image_names[image_nr]])

    # Create a short descriptive .txt file in the folder, giving additional details for the test results
    top_text = "In " + str(total_image_nr) + " of " + str(len(original_images)) + " images " \
               + str(face_nr) + " of " + str(total_faces) + " faces were found. \n"
    bboxes = top_text + str(over_face_nr) + " of those faces had a confidence score >= 0.99. \n \n" + bboxes

    with open(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + str(face_nr) + "_faces_in_" + str(
            total_image_nr) + "_images.txt", "x") as f:
        f.write(bboxes)

    return


for patch_name in os.listdir(img_folder + "/" + patch_folder): # Load and perform testing on each patch located in the local patch folder

    patch = cv2.cvtColor(cv2.imread((img_folder + "/" + patch_folder + "/" + patch_name)), cv2.COLOR_BGR2RGB)
    patch_tensor = tf.cast(patch, dtype=tf.float32)
    run(patch_tensor, images, patch_name)

    cv2.imwrite(img_folder + "/" + results_folder + "/" + patch_name + "/" + patch_name,
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
