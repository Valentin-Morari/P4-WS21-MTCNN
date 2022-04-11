import gc

import tensorflow as tf
import cv2
from mtcnn import MTCNN
import numpy as np
import os

# detector = MTCNN()

# Loads the images into an array

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

img_folder = "Face_Control"

images = []  # list of image data used for training patches
image_names = []  # the file names of images
ground_truths = {}  # the ground truths of faces in loaded images

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


def new_run(patch_used, original_images, amplification_factor: int):
    """
    Trains an adversarial patch on given images.

    :param patch_used: tensor of dtype=float32
        adversarial patch (a small image) to be trained by applying it onto faces for the purpose of reducing
        their detection rate
    :param original_images: list
        images on which the patch will be trained
    :param amplification_factor: int
        representing the amplification factor, which is used to amplify training of the patch
    :return: tensor of dtype=float32
        the trained adversarial patch
    """

    tmp_patch = patch_used
    image_count = 0
    detector = MTCNN()
    for i in range(len(original_images)):

        print(i, "", image_names[i])

        try:  # handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them
            _, _, new_patch = detector.new_detect_faces(original_images[i], tmp_patch, ground_truths[image_names[i]],
                                                        amplification_factor)  # detect>

            tmp_patch = new_patch  # store new patch in the temporary variable

        except Exception as e:  # if exception is met store empty result and image without patch applied to it
            print("Image", i, " (skipping) has the following error:", e)

        image_count += 1
        # Restarting MTCNN every 1 images, to save memory.
        if image_count % 1 == 0:
            detector = MTCNN()
            gc.collect()

    return tmp_patch


"""
#FOR OVERTAKING EXISTING PATCH
init_patch = cv2.cvtColor(cv2.imread((img_folder + "/" + "Patches_Training" + "/" + "_out_13_AmpF=100000_Adversarial_Patch.jpg")), cv2.COLOR_BGR2RGB)
"""

# FOR PATCH INITIALIZATION FROM 0
init_patch = np.random.randint(255, size=(128, 128, 3),
                               dtype=np.uint8)  # Patch Initialization - Set w^P and h^P = 128 to match the paper

old_patch = tf.cast(init_patch, dtype=tf.float32)

amplification_factor = 1000000
try_nr = 0

# (always) creates a new folder in which the train results will be placed
while os.path.exists(
        img_folder + "/AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count) + "_TryNR=" + str(try_nr)):
    try_nr += 1
target_folder = img_folder + "/AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count) + "_TryNR=" + str(
    try_nr)
os.makedirs(target_folder)

cv2.imwrite(
    target_folder + "/" + "_out_" + "INIT_" + "AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count)
    + "_Adversarial_Patch.jpg", cv2.cvtColor(init_patch, cv2.COLOR_RGB2BGR))

for epoch in range(101):

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
