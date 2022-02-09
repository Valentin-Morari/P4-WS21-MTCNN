import cv2
from mtcnn import MTCNN
import numpy as np
import math

detector = MTCNN()

# Loads the images into an array
img_folder = "Face_Control"

ground_truths = {}
images = []
image_names = []

global AS
AS = []

labels = open(img_folder + "/" + "labels.txt", "r")

while labels:
  img_name = labels.readline().rstrip("\n")
  if img_name == "":
    labels.close()
    break
  
  image_names.append(img_name)
  
  images.append(cv2.cvtColor(cv2.imread((img_folder + "/" + img_name)), cv2.COLOR_BGR2RGB))
  ground_truth_count = int(labels.readline().rstrip("\n"))

  ground_truths[img_name] = []
  
  for i in range (ground_truth_count):
    ground_truths[img_name].append([int(value) for value in labels.readline().rstrip("\n").split()])
  
# Detects faces
result = [detector.detect_faces(images[i]) for i in range(len(images))]

#alfa = 0.28 # - To match paper's examples
alfa = 0.5 # - To match paper's concrete parameters
lambda_IoU = 0.6 # To match paper's concrete parameters 

P = np.random.randint(255, size=(128,128,3), dtype=np.uint8) # Patch Initialization - Set w^P and h^P = 128 to match the paper 

def IoU(boxA, boxB):    #Code taken directly from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def run(AS, lambda_IoU):
  global AS
  #select for AS based on IoU > 0.6
  AS = [ASi for ASi in AS if IoU(ASi[0], ASi[1]) > lambda_IoU]
      

# Extracts the bounding boxes from the detected faces
for image_nr in range(len(images)):
    
    i = 0 #used for pairing items in AS
    
    # draw ground truth
    
    ground_truth_boxes = ground_truths[image_names[image_nr]]
    
    for bounding_box in ground_truth_boxes:
      cv2.rectangle(images[image_nr],
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0, 255, 0),
                        2)
      
    # draw detected face + plaster patch over source
    for face in result[image_nr]:
        bounding_box = face['box']
        
        # Makes the bounding box visible
        cv2.rectangle(images[image_nr],
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),
                      2)
        
        if i >= len(ground_truths[image_names[image_nr]]):
          B = ground_truths[image_names[image_nr]][0]
        else:
          B = ground_truths[image_names[image_nr]][i]
                      
        AS.append((bounding_box, B, face['confidence'], P))
        i+=1
        
        resize_value = alfa*math.sqrt(bounding_box[2]*bounding_box[3])
        resized_P = cv2.resize(P, (round(resize_value), round(resize_value)))

        x_P = round(bounding_box[2]/2)
        y_P = round(resize_value/2)

        # draw patch over source image 
        images[image_nr][y_P+bounding_box[1]-round(resized_P.shape[1]/2):y_P+bounding_box[1]-round(resized_P.shape[1]/2)+resized_P.shape[1], x_P+bounding_box[0]-round(resized_P.shape[0]/2):x_P+bounding_box[0]-round(resized_P.shape[0]/2)+resized_P.shape[0]] = resized_P
        
    
    # Creates Images in which the faces are marked through their bounding boxes
    cv2.imwrite(img_folder+"/" + "_out_" + image_names[image_nr], cv2.cvtColor(images[image_nr], cv2.COLOR_RGB2BGR))
