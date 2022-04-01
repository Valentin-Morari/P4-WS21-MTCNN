import gc

import tensorflow as tf
import cv2
from mtcnn import MTCNN
import numpy as np
import os


# detector = MTCNN()

# Loads the images into an array

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

img_folder = "Face_Control"

# global AS
# AS = []
# Only saves ground_truth bounding boxes given by the detector since they already positive samples and therefore satisfy IoU(Ai,Bi) where Ai stands for anchor and Bi stands for ground-truth bounding box.
# Each entry represents one bounding box
images = []
image_names = []
ground_truths = {}

"""
ASes = [ # list of images with their ASes 
        {
            'ground_truth_image': [x][x][3], # image without a patch 
            'AS': [ # list of AS
                    {
                        'anchor': [4], # bounding box of the face found by MTCNN after the adversarial patch  was  added
                        'ground_truth_bounding_box': [4], # ground truth bounding box  stated by the WIDER_FACE dataset
                        'confidence_score': float, # confidence score of the found face
                        'patch': [128][128][3] # adversarial patch 
                     }
                   ]
         }
        ]
"""

labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")
img_count = 0
while labels:
    if img_count == 300:  # number of photos processed
        break

    img_count += 1
    img_name = labels.readline().rstrip("\n")
    print(img_name)
    if img_name == "":
        labels.close()
        break

    image_names.append(img_name)

    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
    ground_truth_count = int(labels.readline().rstrip("\n"))
    print(img_count, "HAS", ground_truth_count, "FACES")

    ground_truths[img_name] = []

    if ground_truth_count == 0:
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][
                                       0:4])  # There are no faces, so the ground_truth is all 0 [so the code can run]
        print(ground_truths[img_name])

    for i in range(ground_truth_count):
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][
                                       0:4])  # take only first 4 values for box size
x = 0

# alpha = 0.28 # - To match paper's examples
alpha = 0.5  # - To match paper's concrete parameters
lambda_IoU = 0.6  # To match paper's concrete parameters

patch = tf.Variable(np.random.randint(255, size=(128, 128, 3),
                                      dtype=np.uint8))  # Patch Initialization - Set w^P and h^P = 128 to match the paper


def IoU(boxA,
        boxB):  # Code taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    # print("BoxA: ")
    # print(boxA)
    # print("BoxB: ")
    # print(boxB)
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
    # print(iou)
    return iou


def new_run(patch_used, original_images, amplification_factor: int):
    # global AS  # AS is broken
    # select for AS based on IoU > 0.6
    # AS = []
    # Detects faces
    # result, adv_img = [detector.new_detect_faces(original_images[i], patch_used, ground_truths[image_names[i]]) for i in range(len(original_images))]
    result = []
    adv_img = []
    ASes = []
    tmp_patch = patch_used
    image_count = 0
    detector = MTCNN()
    for i in range(len(original_images)):

        print(i, "", image_names[i])

        try: #handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them
            _, _, new_patch = detector.new_detect_faces(original_images[i], tmp_patch, ground_truths[image_names[i]], amplification_factor) #detect>

            tmp_patch = new_patch #store new patch in the temporary variable

        except Exception as e: #if exception is met store empty result and image without patch applied to it
            print("Image", i, " (skipping) has the following error:", e)

        image_count += 1
        # Restarting MTCNN every 1 images, to save memory.
        if image_count % 1 == 0:
            detector = MTCNN()
            gc.collect()

    return ASes, adv_img, tmp_patch


init_patch = np.random.randint(255, size=(128, 128, 3),
                               dtype=np.uint8)  # Patch Initialization - Set w^P and h^P = 128 to match the paper


old_patch = tf.cast(init_patch, dtype=tf.float32)

amplification_factor = 1000

cv2.imwrite(img_folder + "/" + "_out_" + "INIT_" + "AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count)
            + "_Adversarial_Patch.jpg", cv2.cvtColor(init_patch, cv2.COLOR_RGB2BGR))

for epoch in range(121):

    _, _, new_patch = new_run(old_patch, images, amplification_factor)
    # print(tf.cast(new_patch, dtype=tf.float32)-old_patch)

    if epoch % 5 == 0:

        np_patch_out = new_patch.numpy()
        np_patch_out = np.fix(np_patch_out)
        cv2.imwrite(img_folder + "/" + "_out_" + str(epoch) + "_AmpF=" + str(amplification_factor) + "_IMG_COUNT="
                    + str(img_count) + "_Adversarial_Patch.jpg", cv2.cvtColor(np_patch_out, cv2.COLOR_RGB2BGR))

    if epoch == 60:
        amplification_factor *= 0.1
    if epoch == 80:
        amplification_factor *= 0.1

    old_patch = tf.cast(new_patch, dtype=tf.float32)

    print("Epoch", epoch)

    mu = 0

