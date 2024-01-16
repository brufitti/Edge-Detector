# Edge-Detector Project

Files and usability:

## ---generator.py---
>
>Detects edges in a gray-scale image and saves the percieved results in a new image called 'LAPLACE_' + original_name.
>
>The main function reads a single image, applies the sobel derivatives and stores the result, then runs the image through an untrained network. Soon after it display the three images (gray, edge and network).
>
>>To use main function set the arguments image_dir and image_name to the folder and name of the desired image respectively.
>
>>To generate a dataset from folders of images, create three directories as subdirectories to the main workspace (train, test and validation).
>>train is where train images must be stored.
>>validation is where validation images must be stored.
>>test is for files you wish to use as examples outside of validation and test loop, such as new images with new HxW ratio that don't fit the train and validation file format.


## ---edge_net.py---
>
>File containing the network model class
>
>While the number of intermediary channels was chosen at random (1->5->1), the minimum that has yielded significant results is 1->2->1

## ---data.py---
>
>File containing the custom dataset class
>
>Can also be used to generate a dataset with the generate_dataset function

## ---train.py---
>
>Train structure that uses edge_net.py and data.py
>
>Capable of resuming previous traning by setting --resume_dir='path_to_last_saved_model.pt'
>Stores the model weights and optimizer properties every 100 epochs
>Checks for cuda availability, but does not report on the availability, if training is slow, check that cuda is enabled in your device.
>
>read all parser.add_argument() options to understand valid argumets when calling the train.py file, a default is set for all args, but the train loop requires a 'models/' directory.

## ---eval.py---
>
>For evaluation of a single image, set weights path to your desired state_dict, image_path to the original image to be run throught the network and laplace path to the corresponding 'LAPLACE_' edge image
>
















