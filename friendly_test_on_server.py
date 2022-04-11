import gc

import tensorflow as tf
import cv2
from mtcnn import MTCNN
from mtcnn import original_MTCNN
import numpy as np
import os

ground_truths_detector = original_MTCNN()

# loads the images into an array

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# loads the images into an array
img_folder = "User_Friendly/Train"

images = []  # list of image data used for training patches
image_names = os.listdir(img_folder + '/Images')  # the file names of images
img_count = len(image_names)  # how many images are used
ground_truths = {}  # the ground truths of faces in loaded images

for image_name in image_names:
    print(image_name)
    # loads the image into memory
    image = cv2.cvtColor(cv2.imread((img_folder + "/Images/" + image_name)), cv2.COLOR_BGR2RGB)
    # finds the ground truths [faces] of the image using the original output of MTCNN
    result_ground_truths = ground_truths_detector.detect_faces(image)
    # saves the image into images
    images.append(image)
    ground_truths[image_name] = []
    # saves the ground truths into the dictionary ground truths with the name of the image as the pointer
    for result_ground_truth in result_ground_truths:
        ground_truths[image_name].append(result_ground_truth['box'])

del (ground_truths_detector)
gc.collect()


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
        # restarting MTCNN every 1 images, to save memory.
        if image_count % 1 == 0:
            detector = MTCNN()
            gc.collect()

    return tmp_patch


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

    new_patch = new_run(old_patch, images, amplification_factor)

    if epoch % 5 == 0:
        np_patch_out = new_patch.numpy()
        np_patch_out = np.fix(np_patch_out)
        cv2.imwrite(target_folder + "/" + "_out_" + str(epoch) + "_AmpF=" + str(amplification_factor) + "_IMG_COUNT="
                    + str(img_count) + "_Adversarial_Patch.jpg", cv2.cvtColor(np_patch_out, cv2.COLOR_RGB2BGR))

    if epoch == 60:
        amplification_factor *= 0.1
    if epoch == 80:
        amplification_factor *= 0.1

    old_patch = tf.cast(new_patch, dtype=tf.float32)

    print("Epoch", epoch)

    mu = 0
