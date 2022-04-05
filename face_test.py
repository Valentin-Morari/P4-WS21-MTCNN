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

n = 0
while labels:
    if n == 0:  # number of photos processed
      for _ in range(52762): #skip to the dataset relevant to us
          next(labels)
    if n == 994:
        break
          
    n += 1
    img_name = labels.readline().rstrip("\n") #reads image name from list of labels
    
    if img_name == "": #stop if we reach the end of the labels file
        labels.close()
        break

    image_names.append(img_name)
    print(img_folder+"/"+img_name)
    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
    ground_truth_count = int(labels.readline().rstrip("\n"))

    ground_truths[img_name] = []
    
    if not ground_truth_count:
      next(labels)
    else:
      for i in range(ground_truth_count):
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()])
    
# alpha = 0.28 # - To match paper's examples
alpha = 0.5  # - To match paper's concrete parameters
lambda_IoU = 0.6  # To match paper's concrete parameters

patch = tf.Variable(np.random.randint(255, size=(128, 128, 3),
                                      dtype=np.uint8))  # Patch Initialization - Set w^P and h^P = 128 to match the paper


def IoU(boxA,
        boxB):  # Code taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
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
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def output_images(ASes_output, originals, patch_results_folder, prefix = ""):
    working_images = copy.deepcopy(originals)  # don't modify the original pictures
    
    if os.path.exists(img_folder + "/" + results_folder + "/" + patch_results_folder):
          shutil.rmtree(img_folder + "/" + results_folder + "/" + patch_results_folder)
    os.makedirs(img_folder + "/" + results_folder + "/" + patch_results_folder)
        
    face_nr = 0
    total_image_nr = 0
    bboxes = ""
    for image_nr in range(len(working_images)):  # loop through all
        
        ground_truth_boxes = ground_truths[image_names[image_nr]]

        for ground_truth_bounding_box in ground_truth_boxes:  # ground truth loop
            
            cv2.rectangle(working_images[image_nr],
                          (ground_truth_bounding_box[0], ground_truth_bounding_box[1]),
                          (ground_truth_bounding_box[0] + ground_truth_bounding_box[2],
                           ground_truth_bounding_box[1] + ground_truth_bounding_box[3]),
                          (0, 255, 0),
                          2)

        if len(ASes_output[image_nr]['AS']):
          total_image_nr += 1
          
        for AS_of_one_image in ASes_output[image_nr]['AS']:
            face_nr += 1
            bounding_box = AS_of_one_image['anchor']  # guess loop
            confidence_score = AS_of_one_image['confidence_score'].numpy() # extract confidence score of detected bounding box

            # Makes the bounding box visible
            print(image_names[image_nr] + " has", bounding_box, " with confidence: ", confidence_score)
            bboxes += str(image_names[image_nr] + " has") + str(bounding_box) + " with confidence: " + str(confidence_score) + "\n"
            cv2.rectangle(working_images[image_nr],
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)
            cv2.putText(working_images[image_nr], str(confidence_score), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) #places confidence score over the detected face boxes
              
            prefix = str(AS_of_one_image['confidence_score'].numpy()) #added
        
        # Creates working_images in which the faces are marked through their bounding boxes
        try:
          cv2.imwrite(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + "_out_" + prefix + image_names[image_nr].split('/')[1],
                    cv2.cvtColor(working_images[image_nr], cv2.COLOR_RGB2BGR))
        except:
          cv2.imwrite(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + "_out_" + prefix + image_names[image_nr],
                    cv2.cvtColor(working_images[image_nr], cv2.COLOR_RGB2BGR))
        
        prefix = ""
    
    with open(img_folder + "/" + results_folder + "/" + patch_results_folder + "/" + str(face_nr) + "_faces_in_" + str(total_image_nr) + "_images.txt", "x") as f: #what
      f.write(bboxes)


def new_run(patch_used, original_images):
    result = []
    adv_img = []
    tmp_patch = patch_used
    for i in range(len(original_images)):
        try: #handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them

          tmp_result, tmp_adv_img, new_patch = detector.new_detect_faces(original_images[i], tmp_patch, ground_truths[image_names[i]], 0)
          tmp_patch = new_patch
          result.append(tmp_result)
          adv_img.append(tmp_adv_img)
        
        except Exception as e: #if exception is met store empty result and image without patch applied to it
          print("Image", i, " (skipping) has the following error:", e)
          result.append([])
          adv_img.append(original_images[i])

    ASes = []  # Reset Adversarial Samples when re-running the algorithm

    # Extracts the bounding boxes from the detected faces
    for image_nr in range(len(original_images)):

        AS = []

        faces = result[image_nr]
        # draw detected face + plaster patch over source
        for j in range(len(faces)):
            bounding_box = faces[j]['box']
            confidence_score = faces[j]['confidence']

            for ground_truth_bounding_box in ground_truths[
                image_names[image_nr]]:  # check each possible ground truth box, instead of assuming they're ordered
                if (IoU(bounding_box, ground_truth_bounding_box) > lambda_IoU):                                       # """ might be relevant? """
                    AS.append({'anchor': bounding_box, 'ground_truth_bounding_box': ground_truth_bounding_box,
                               'confidence_score': confidence_score, 'patch': patch_used})

        ASes.append({'ground_truth_image': images[image_nr], 'AS': AS})

    return ASes, adv_img, tmp_patch


for patch_name in os.listdir(img_folder + "/" + patch_folder):

    init_patch = cv2.cvtColor(cv2.imread((img_folder + "/" + patch_folder + "/" + patch_name)), cv2.COLOR_BGR2RGB)
    old_patch = tf.cast(init_patch, dtype=tf.float32)

    for i in range(1):
        bbox, adv_img, new_patch = new_run(old_patch, images)

        output_images(bbox, adv_img, patch_name, str(i)+"_pp_")
        old_patch = tf.cast(new_patch, dtype=tf.float32)
        print("Epoch", i)
    
        cv2.imwrite(img_folder + "/" + results_folder + "/" + patch_name + "/" + "P_test_patch.png",
                      cv2.cvtColor(new_patch.numpy(), cv2.COLOR_RGB2BGR))