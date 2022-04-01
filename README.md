# P4-WS21-MTCNN

## Attacking Face Detection with Adversarial Patches

![_out_0 890586359_peopledrivingcar_peopledrivingcar_59_144](https://user-images.githubusercontent.com/35852035/161250277-c0aa3d4a-c017-4d22-ab0b-801693956343.jpg)

_Legend: Blue box - detected face, with a confidence score above it, and green - ground truth bounding box._

Clone the environment using conda by running: conda env create -f environment.yml in the base folder containing environment.yml

Run it by calling:
  1. conda activate MTCNN-ONET
  2. python3 test.py
  
For the O-MTCNN attack, the main file to be modified is mtcnn.py in the folder mtcnn. 

Alternatively, a pre-made venv environment is provided. Activate using source bin/activate and then run using python3 test.py

### Getting started

To run our project from scratch (no conda):

    1. git clone https://github.com/Valentin-Morari/P4-WS21-MTCNN
    2. python3.8 -m venv <name your new virtual environment>
    3. Windows: a. cd <your virtual environment's name>/Scripts
                b. start activate.bat
                c. cd ../.. (to go back to the original folder)
       Bash:    source <your virtual environment's name>/bin/activate
    4. cd P4-WS21-MTCNN (our project's name)
    5. pip3.8 install -r reqs.txt
  
### How to train 
  
PATCH TRAINING:
  python3.8 test.py
  or
  python3.8 test_on_server.py (memory optimized for execution on a remote server)

  What happens in test.py:
  
    - img_folder (Face_Control) -> stores images to use for training the adversarial patch generation
    - labels (wider_face_train_bbx_gt.txt) -> retrieve ground truths of the bounding boxes for the faces useed during training
    - init_patch -> which image to initialize from (random, or specify a starting image)
    - amplification factor -> gradient amplification power to offset smaller training image set or smaller number of epochs
    - various cv2.imwrite functions -> used for saving the images with the patches applied and the patches themselves to the hard drive

  Check the Face_Control/1kImgResults folder for execution results. 
  Legend: Blue box - detected face, with a confidence score above it, and green - ground truth bounding box.

### How to test 
  
PATCH TESTING:

  1. place the 0--Parade, 2--Demonstration and 13--Interview (or whatever image testing datasets you want) into the folder Test_Faces.
  2. place the patch(es) you want to test into the Test_Faces/to_test folder
  2. make sure there's an appropriate ground truth label file in the Test_Faces directory (loading images for testing is done using this file)
  3. python3.8 face_test.py
  4. test results will be placed in the Test_Faces/test_results folder, into a newly created folder based on the name of your patch(es)

  
    
