import gc

import tensorflow as tf
import cv2
from mtcnn import MTCNN
import numpy as np
import math
import copy
import os

#import psutil
#import memory_profiler
#from memory_profiler import profile

#detector = MTCNN()

# Loads the images into an array
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

#ASes = []

#labels = open(img_folder + "/" + "labels.txt", "r")
"""
while labels:
    img_name = labels.readline().rstrip("\n")
    if img_name == "":
        labels.close()
        break

    image_names.append(img_name)

    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
    ground_truth_count = int(labels.readline().rstrip("\n"))

    ground_truths[img_name] = []

    for i in range(ground_truth_count):
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()])
"""

  
labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")
n = 0
while labels:
    if n == 460: #number of photos processed
      break

    n += 1
    img_name = labels.readline().rstrip("\n")
    print(img_name)
    if img_name == "":
        labels.close()
        break

    image_names.append(img_name)
    

    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
    ground_truth_count = int(labels.readline().rstrip("\n"))
    print("HAS", ground_truth_count, "FACES")

    ground_truths[img_name] = []

    if ground_truth_count == 0:
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][0:4]) # There are no faces, so the ground_truth is all 0 [so the code can run]
        print(ground_truths[img_name])

    for i in range(ground_truth_count):
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][0:4]) # take only first 4 values for box size


  
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
    #print(iou)
    return iou


def output_images(ASes_output, originals, patch_output, prefix = ""):
    working_images = copy.deepcopy(originals)  # don't modify the original pictures

    for image_nr in range(len(working_images)):  # loop through all

        ground_truth_boxes = ground_truths[image_names[image_nr]]

        for ground_truth_bounding_box in ground_truth_boxes:  # ground truth loop
            cv2.rectangle(working_images[image_nr],
                          (ground_truth_bounding_box[0], ground_truth_bounding_box[1]),
                          (ground_truth_bounding_box[0] + ground_truth_bounding_box[2],
                           ground_truth_bounding_box[1] + ground_truth_bounding_box[3]),
                          (0, 255, 0),
                          2)

        for AS_of_one_image in ASes_output[image_nr]['AS']:
            bounding_box = AS_of_one_image['anchor']  # guess loop
            # print(bounding_box)
            # print(ASes_output[image_nr]['AS'])

            # Makes the bounding box visible
            cv2.rectangle(working_images[image_nr],
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)
        #print(working_images[image_nr])
        # Creates working_images in which the faces are marked through their bounding boxes
        cv2.imwrite(img_folder + "/" + "_out_" + prefix + image_names[image_nr].split("/")[1],
                    cv2.cvtColor(working_images[image_nr], cv2.COLOR_RGB2BGR))

        np_patch_out = patch_output.numpy()
        np_patch_out = np.fix(np_patch_out)
        cv2.imwrite(img_folder + "/" + "_out_" + prefix + "Adversarial_Patch.jpg",
                    cv2.cvtColor(np_patch_out, cv2.COLOR_RGB2BGR))
        

def apply_patch(originals, patch):
    working_images = copy.deepcopy(originals)  # don't modify the original pictures
    for image_nr in range(len(working_images)):

        i = 0  # used for pairing items in AS

        ground_truth_boxes = ground_truths[image_names[image_nr]]

        # draw detected face + plaster patch over source
        for bounding_box in ground_truth_boxes:  # ground truth loop

            if i >= len(ground_truths[image_names[image_nr]]):
                B = ground_truths[image_names[image_nr]][0]
            else:
                B = ground_truths[image_names[image_nr]][i]

            resize_value = alpha * math.sqrt(bounding_box[2] * bounding_box[3])
            resized_P = cv2.resize(patch, (round(resize_value), round(resize_value)))

            x_P = round(bounding_box[2] / 2)
            y_P = round(resize_value / 2)

            # draw patch over source image
            working_images[image_nr][
            y_P + bounding_box[1] - round(resized_P.shape[1] / 2):y_P + bounding_box[1] - round(
                resized_P.shape[1] / 2) + resized_P.shape[1],
            x_P + bounding_box[0] - round(resized_P.shape[0] / 2):x_P + bounding_box[0] - round(
                resized_P.shape[0] / 2) + resized_P.shape[0]] = resized_P

    return working_images

'''
def run(patch_used, edited_images):
    # global AS  # AS is broken
    # select for AS based on IoU > 0.6
    # AS = []
    # Detects faces
    result = [detector.detect_faces(edited_images[i]) for i in range(len(edited_images))]

    ASes = []  # Reset Adversarial Samples when re-running the algorithm

    # Extracts the bounding boxes from the detected faces
    for image_nr in range(len(edited_images)):

        AS = []

        i = 0  # used for pairing items in AS

        # ground_truth_boxes = ground_truths[image_names[image_nr]]

        faces = result[image_nr]
        # draw detected face + plaster patch over source
        for j in range(len(faces)):
            bounding_box = faces[j]['box']
            # print("BOUND: ")
            # print(bounding_box)
            confidence_score = faces[j]['confidence']

            for ground_truth_bounding_box in ground_truths[
                image_names[image_nr]]:  # check each possible ground truth box, instead of assuming they're ordered
                if (IoU(bounding_box, ground_truth_bounding_box) > lambda_IoU):
                    AS.append({'anchor': bounding_box, 'ground_truth_bounding_box': ground_truth_bounding_box,
                               'confidence_score': confidence_score, 'patch': patch_used})

            # print("GROUND: ")
            # print(ground_truth_bounding_box)

            # print("AS: ")
            # print(AS)

        ASes.append({'ground_truth_image': images[image_nr], 'AS': AS})
        # print(ASes)

    # AS = [ASi for ASi in AS if IoU(ASi[0], ASi[1]) > lambda_IoU]
    return ASes
'''

def new_run(patch_used, original_images, amplification_factor:int):
    # global AS  # AS is broken
    # select for AS based on IoU > 0.6
    # AS = []
    # Detects faces
    #result, adv_img = [detector.new_detect_faces(original_images[i], patch_used, ground_truths[image_names[i]]) for i in range(len(original_images))]
    result = []
    adv_img = []
    tmp_patch = patch_used
    image_count = 0
    detector = MTCNN()
    for i in range(len(original_images)):
        #process = psutil.Process(os.getpid())
        #print("IN TOTAL", process.memory_info().rss / 1000000, "MB")
        
        #print(tmp_patch,tf.math.scalar_mul(0.5, tmp_patch))

        print(image_count, "_", image_names[i])

        #Restarting MTCNN every 9 images, to save memory. 10 already rises up to 8GB sometimes.
        '''
        detector = MTCNN()
        gc.collect()
        '''

        tmp_result, tmp_adv_img, new_patch = detector.new_detect_faces(original_images[i], tmp_patch
                                                                       , ground_truths[image_names[i]], amplification_factor)
        #print(orig-patch_used)
        tmp_patch = new_patch
        result.append(tmp_result)
        adv_img.append(tmp_adv_img)

        image_count += 1

        if image_count % 9 == 0:
            detector = MTCNN()
            gc.collect()
    
    #print("RESULT:")
    #print(result)
    ASes = []  # Reset Adversarial Samples when re-running the algorithm

    # Extracts the bounding boxes from the detected faces
    for image_nr in range(len(original_images)):

        AS = []

        i = 0  # used for pairing items in AS

        # ground_truth_boxes = ground_truths[image_names[image_nr]]

        faces = result[image_nr]
        #print(image_nr)
        #print(faces)
        # draw detected face + plaster patch over source
        for j in range(len(faces)):
            bounding_box = faces[j]['box']
            # print("BOUND: ")
            # print(bounding_box)
            confidence_score = faces[j]['confidence']

            for ground_truth_bounding_box in ground_truths[
                image_names[image_nr]]:  # check each possible ground truth box, instead of assuming they're ordered
                if (IoU(bounding_box, ground_truth_bounding_box) > lambda_IoU):                                       # """ might be relevant? """
                    AS.append({'anchor': bounding_box, 'ground_truth_bounding_box': ground_truth_bounding_box,
                               'confidence_score': confidence_score, 'patch': patch_used})

            # print("GROUND: ")
            # print(ground_truth_bounding_box)

            # print("AS: ")
            # print(AS)

        ASes.append({'ground_truth_image': images[image_nr], 'AS': AS})
        # print(ASes)

    # AS = [ASi for ASi in AS if IoU(ASi[0], ASi[1]) > lambda_IoU]
    return ASes, adv_img, tmp_patch

init_patch = np.random.randint(255, size=(128, 128, 3),
                               dtype=np.uint8)  # Patch Initialization - Set w^P and h^P = 128 to match the paper

# s1 = run(init_patch, images)  # should be fineyy
# output_images(s1, images)
# input("Check images now...")

#a = apply_patch(images, init_patch)
#s2 = run(init_patch, a)  # should be fineyy
#output_images(s2, a)

old_patch = tf.cast(init_patch, dtype=tf.float32)

amplification_factor = 1000000

cv2.imwrite(img_folder + "/" + "_out_" + "INIT_"+ "AmpF=" + str(amplification_factor) +"_Adversarial_Patch.jpg",
            cv2.cvtColor(init_patch, cv2.COLOR_RGB2BGR))

for epoch in range(121):
    '''
    mu = 0
    for name in list(vars()):
        print(name, " ### ", sys.getsizeof(name))
        mu = mu + sys.getsizeof(name)
    print(mu)
    '''
    bbox, adv_img, new_patch = new_run(old_patch, images, amplification_factor)
    #print(tf.cast(new_patch, dtype=tf.float32)-old_patch)
    """
    if i ==0:
      output_images(bbox, adv_img, "first")
    elif i == 49:
      output_images(bbox, adv_img, "last")
    else:
      output_images(bbox, adv_img)
    """
    if epoch % 10 == 0:
      # output_images(bbox, adv_img, new_patch, str(epoch)+"_")

      np_patch_out = new_patch.numpy()
      np_patch_out = np.fix(np_patch_out)
      cv2.imwrite(img_folder + "/" + "_out_" + str(epoch) + "_AmpF=" + str(amplification_factor) + "_Adversarial_Patch.jpg",
                  cv2.cvtColor(np_patch_out, cv2.COLOR_RGB2BGR))

    if epoch == 60:
        amplification_factor *= 0.1
    if epoch == 80:
        amplification_factor *= 0.1

    old_patch = tf.cast(new_patch, dtype=tf.float32)

    print("Epoch", epoch)

    mu = 0
    '''
    for name in list(vars()):
        print(name, " ### ", sys.getsizeof(name))
        mu = mu + sys.getsizeof(name)
    print(mu)
    '''
    """
      if i == 1:
        patch = cv2.cvtColor(cv2.imread(('wizards.jpg')), cv2.COLOR_BGR2RGB)
    """
'''
def loss_object():
    #print(patch)
    p = patch.numpy()
    #print(p)
    a = apply_patch(images, p)
    s2 = run(p, a)

    confidence_list = []
    for ASes_of_one_image in s2:
        #print(ASes_of_one_image['AS'])
        #print("______________________________________________________________________________________________________________")
        for AS_of_one_image in ASes_of_one_image['AS']:
            #print(AS_of_one_image['confidence_score'])
            confidence_list.append(AS_of_one_image['confidence_score'])

    loss = tf.divide(tf.math.reduce_sum(tf.math.log(confidence_list)), len(confidence_list))

    return loss
'''

'''
opt = tf.keras.optimizers.SGD(learning_rate=0.01) #learning rate laut dem paper
for i in range(50):
    with tf.GradientTape() as tape:
        tape.watch(patch)
        loss = loss_object()
    grads = tape.gradient((loss, patch))
    processed_grads = [g for g in grads]
    grads_and_vars = zip(processed_grads, patch)
    opt.applygradients(grads_and_vars)
out_patch = patch.numpy
cv2.imwrite(img_folder + "/" + "_out_" + "Patch", cv2.cvtColor(out_patch, cv2.COLOR_RGB2BGR))
'''


'''
opt = tf.keras.optimizers.SGD(learning_rate=alpha)
for i in range(50):
    opt.minimize(loss_object(), patch)
out_patch = patch.numpy
cv2.imwrite(img_folder + "/" + "_out_" + "Patch",
            cv2.cvtColor(out_patch, cv2.COLOR_RGB2BGR))
'''

