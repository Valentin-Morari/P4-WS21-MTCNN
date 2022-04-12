# P4-WS21-MTCNN
## Attacking Face Detection with Adversarial Patches

This project generates adversarial patches against MTCNN's ONET and converts MTCNN's ONET implementation to tensorflow (from numpy), to enable automatic gradient differentiation. 

[MTCNN implementation](https://github.com/ipazc/mtcnn) by Iván de Paz Centeno. Patch generation based on "[Design and 
Interpretation of Universal Adversarial Patches in Face Detection](https://arxiv.org/abs/1912.05021)” released in 2020 by Yang, Xiao, Wei, Fangyun, Zhang, Hongyang, and Zhu, Jun.

Uses Python3.8 and Tensorflow2.7. 

![_out_0 890586359_peopledrivingcar_peopledrivingcar_59_144](https://user-images.githubusercontent.com/35852035/161250277-c0aa3d4a-c017-4d22-ab0b-801693956343.jpg)

_Legend: Blue box - detected face, with a confidence score above it, and green - ground truth bounding box for the face, ideal result. Image courtesy of WIDERFACE, with our project's results applied._

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
  
  How to train the normal version:
  
    python3.8 test.py (additionally returns images with the applied adversarial patches)
    or
    python3.8 test_on_server.py (memory optimized for execution on a remote server | only returns the adversarial patch)

    What happens in test.py and in test_on_server:
  
      - img_folder (Face_Control) -> stores images to use for training the adversarial patch generation
      - labels (wider_face_train_bbx_gt.txt) -> retrieve ground truths of the bounding boxes for the faces useed during training
      - init_patch -> which image (the adversarial patch) to initialize from (random, or specify a starting image)
      - amplification factor -> gradient amplification power to offset smaller training image set or smaller number of epochs 
                              or a bigger patch
      - various cv2.imwrite functions -> used for saving the images with the patches applied and the patches themselves to the hard drive

    Check the latest TryNR folder for your used amplification factor and image count in the Face_Control folder for execution 
    results. (e.g. AmpF=1000000_IMG_COUNT=1029_TryNR=0)
  
  How to train the user friendly version:
  
    add images, with which you want to train a adversarial patch, into User_Friendly/Train/Images
    
    execute python3.8 friendly_test_on_server.py
    
    What happens in friendly_test_on_server.py
    
      - img_folder (User_Friendly/Train) -> stores images to use for training the adversarial patch generation
      - ground_truths -> uses the results of a image in MTCNN as the ground truths for the image
      - init_patch -> which image (the adversarial patch) to initialize from (random)
      - amplification factor -> gradient amplification power to offset smaller training image set or smaller number of epochs 
                              or a bigger patch
      - various cv2.imwrite functions -> used for saving the patches to the hard drive
    
    Check the latest TryNR folder for your used amplification factor and image count in the User_Friendly/Train folder for execution 
    results. (e.g. AmpF=1000000_IMG_COUNT=8_TryNR=0)
  
  Legend: Blue box - detected face, with its confidence score above it, and green - ground truth bounding box.

### How to test 
  
PATCH TESTING:

How to test with the normal version:

  1. place the 0--Parade, 2--Demonstration and 13--Interview (or whatever image testing datasets you want) into the folder Test_Faces.
  2. place the patch(es) you want to test into the Test_Faces/to_test folder
  2. make sure there's an appropriate ground truth label file in the Test_Faces directory (loading images for testing is done using this file)
  3. python3.8 face_test.py
  4. test results will be placed in the Test_Faces/test_results folder, into a newly created folder based on the name of your patch(es)

How to test with the user friendly version:

  1. place images you want your patches to test on into the folder User_Friendly/Test/Images.
  2. place the patch(es) you want to test into the User_Friendly/Test/to_test folder
  3. python3.8 friendly_face_test.py
  4. test results will be placed in the User_Friendly/Test/test_results folder, into a newly created folder based on the name of your patch(es)

  
    
