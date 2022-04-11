import gc #used for garbage collection and reducing overall RAM usage

import tensorflow as tf
import cv2 #image processing library
from mtcnn import MTCNN #face detection library (local -> check mtcnn folder)
import numpy as np
import math
import copy #used for creating deep copies in order to not modify the originals
import os

# Loads the images into an array
img_folder = "Face_Control"
images = []
image_names = []
ground_truths = {}


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


"""
Adversarial Sample Set structure:

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
  
labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r") #read WIDERFACE ground truth labels (taken from the training set)
    
img_count = 0 #number of photos processed

while labels:
    if img_count == 460: #stop when number of photos processed reaches desired value
      labels = open(img_folder + "/" + "wider_face_train_bbx_gt.txt", "r")
      for _ in range(40077):
          next(labels)
    if img_count == 1029:
        break

    img_count += 1
    img_name = labels.readline().rstrip("\n") #reads image name from list of labels
    if img_name == "": #stop if we reach the end of the labels file
        labels.close()
        break

    image_names.append(img_name) #appends the image's name to a list of image NAMES
    images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB)) #appends the image data to a list of all image DATAS (loads into memory)
    ground_truth_count = int(labels.readline().rstrip("\n")) #extract number of faces on image [according to label file, WIDERFACE format]
    ground_truths[img_name] = [] #initialize dictionary of ground_truths for the image, where ground_truth boxes will be stored

    if ground_truth_count == 0:
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][0:4]) # There are no faces, so the ground_truth is all 0 [so the code can run]
        
    for i in range(ground_truth_count):
        ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()][0:4]) # take only first 4 values for box size, as WIDERFACE maps out more values by default (such as luminosity)

# alpha = 0.28 # - To match paper's given examples
alpha = 0.5  # - To match paper's concrete parameters
lambda_IoU = 0.6  # To match paper's concrete parameters

patch = tf.Variable(np.random.randint(255, size=(128, 128, 3),
                                      dtype=np.uint8))  # Patch Initialization - Set w^P and h^P = 128 to match the paper

def IoU(boxA,
        boxB):  # Code taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    # MTCNN doesn't give the end points in [2] and [3] but the distance from the start point to the end point (width and height)
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def output_images(ASes_output, imgs, patch_output, prefix = ""):
    """
    handles storing the images outside of the program's memory

    :param ASes_output: Adversarial Sample Set structure
        adversarial samples
    :param imgs:
        images on which the patch was applied to
    :param patch_output:
        the adversarial patch
    :param prefix: str
        stating the epoch in which the images are returned. Default is ""
    :return:
        None
    """
    
    working_images = copy.deepcopy(imgs)  # don't modify the original pictures

    for image_nr in range(len(working_images)):  # loop through all

        ground_truth_boxes = ground_truths[image_names[image_nr]] #pull up the ground_truths appropriate to the image

        for ground_truth_bounding_box in ground_truth_boxes:  # ground truth loop
            cv2.rectangle(working_images[image_nr],
                          (ground_truth_bounding_box[0], ground_truth_bounding_box[1]),
                          (ground_truth_bounding_box[0] + ground_truth_bounding_box[2],
                           ground_truth_bounding_box[1] + ground_truth_bounding_box[3]),
                          (0, 255, 0),
                          2) #imprint a green rectangle over the ground truth coordinates (ground_truth_bounding_box[2] and [3] are width and height, respectively)

        for AS_of_one_image in ASes_output[image_nr]['AS']:
            bounding_box = AS_of_one_image['anchor']  # extract detected bounding boxes
            confidence_score = AS_of_one_image['confidence_score'].numpy() # extract confidence score of detected bounding box
            
            # Makes the bounding box visible
            cv2.rectangle(working_images[image_nr],
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2) #imprint a blue rectangle over the detected bounding box of the face
            
            cv2.putText(working_images[image_nr], str(confidence_score), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) #places confidence score over the detected face boxes
                          
        # Creates working_images in which the faces are marked through their bounding boxes
        cv2.imwrite(target_folder + "/" + "_out_" + prefix + image_names[image_nr].split("/")[1], cv2.cvtColor(working_images[image_nr], cv2.COLOR_RGB2BGR))

        np_patch_out = patch_output.numpy() #convert from Tensorflow variable to numpy, numerical-only representation
        np_patch_out = np.fix(np_patch_out)
        cv2.imwrite(target_folder + "/" + "_out_" + prefix + "Adversarial_Patch.jpg",
                    cv2.cvtColor(np_patch_out, cv2.COLOR_RGB2BGR)) # Stores and overwrites latest Adversarial_Patch image in local folder (used for visual convenience)


def new_run(patch_used, original_images, amplification_factor:int):
    """
    Trains an adversarial patch on given images. (Represents one epoch from the paper)

    :param patch_used: tensor of dtype=float32
        adversarial patch (a small image) to be trained by applying it onto faces for the purpose of reducing
        their detection rate
    :param original_images: list
        images on which the patch will be trained
    :param amplification_factor: int
        representing the amplification factor, which is used to amplify training of the patch
    :return:
        ASes: Adversarial Sample Set structure
            adversarial samples
        adv_img: list containing nd-arrays
            image on which the patch was applied to
        tmp_patch: TensorFlow variable
            the adversarial patch after one epoch
    """
    
    result = [] #stores MTCNN face detection result
    adv_imgs = [] #stores images with patch applied to them
    tmp_patch = patch_used #initialize temporary patch value to input patch
    detector = MTCNN() #initialize the MTCNN detector (repeated for alleviating RAM)
    
    for i in range(len(original_images)): #process every image stored 
        
        print(i, "_", image_names[i])

        try: #handles exotic errors (often related to extreme bounding box sizes) by skipping over images that produce them
          tmp_result, tmp_adv_img, new_patch = detector.new_detect_faces(original_images[i], tmp_patch, ground_truths[image_names[i]], amplification_factor) #detect faces on given images
          
          tmp_patch = new_patch #store new patch in the temporary variable
          result.append(tmp_result) #store face detection results
          adv_imgs.append(tmp_adv_img) #store image with patch applied to it
          
        except Exception as e: #if exception is met store empty result and image without patch applied to it
          print("Image", i, " (skipping) has the following error:", e)
          result.append([])
          adv_imgs.append(original_images[i])
        
        #Restarting MTCNN every 3 images, to save memory. It may rise up to 8GB sometimes. Reseting also resets RAM usage, at least under Windows.
        if i % 3 == 0:
            detector = MTCNN()
            gc.collect()
        
    ASes = []  # Reset Adversarial Sample Set each epoch 

    # Extracts the bounding boxes from the detected faces
    for image_nr in range(len(original_images)):
        AS = [] # List of Adversarial Samples for this image (multiple faces -> multiple samples)
        faces = result[image_nr] # access face detection results for current image
        
        # draw detected face + plaster patch over source
        for j in range(len(faces)):
            bounding_box = faces[j]['box']
            confidence_score = faces[j]['confidence']

            for ground_truth_bounding_box in ground_truths[image_names[image_nr]]:  # check each possible ground truth box, instead of assuming they're ordered
                if (IoU(bounding_box, ground_truth_bounding_box) > lambda_IoU):                                       # """ might be relevant? """
                    AS.append({'anchor': bounding_box, 'ground_truth_bounding_box': ground_truth_bounding_box,
                               'confidence_score': confidence_score, 'patch': patch_used})

        ASes.append({'ground_truth_image': images[image_nr], 'AS': AS})

    # AS = [ASi for ASi in AS if IoU(ASi[0], ASi[1]) > lambda_IoU]
    return ASes, adv_imgs, tmp_patch

"""
#FOR PATCH INITIALIZATION FROM 0
init_patch = np.random.randint(255, size=(128, 128, 3),
                               dtype=np.uint8)  # Patch Initialization - Set w^P and h^P = 128 to match the paper
"""

#FOR OVERTAKING EXISTING PATCH
init_patch = cv2.cvtColor(cv2.imread((img_folder + "/" + "Patches_Training" + "/" + "_out_13_AmpF=100000_Adversarial_Patch.jpg")), cv2.COLOR_BGR2RGB)

old_patch = tf.cast(init_patch, dtype=tf.float32)

amplification_factor = 1000000
try_nr = 0

# (always) creates a new folder in which the train results will be placed
while os.path.exists(img_folder + "/AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count) + "_TryNR=" + str(try_nr)):
    try_nr += 1
target_folder = img_folder + "/AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count) + "_TryNR=" + str(try_nr)
os.makedirs(target_folder)

cv2.imwrite(target_folder + "/" + "_out_" + "INIT_" + "AmpF=" + str(amplification_factor) + "_IMG_COUNT=" + str(img_count)
            + "_Adversarial_Patch.jpg", cv2.cvtColor(init_patch, cv2.COLOR_RGB2BGR))

for epoch in range(101):

    bbox, adv_img, new_patch = new_run(old_patch, images, amplification_factor)

    cv2.imwrite(target_folder + "/" + "_out_" + str(epoch) + "_AmpF=" + str(amplification_factor) + "_Adversarial_Patch.jpg", cv2.cvtColor(new_patch.numpy(), cv2.COLOR_RGB2BGR))

    if epoch % 5 == 0:
      output_images(bbox, adv_img, new_patch, str(epoch)+"_")

    old_patch = tf.cast(new_patch, dtype=tf.float32)

    print("Epoch", epoch)

    mu = 0


