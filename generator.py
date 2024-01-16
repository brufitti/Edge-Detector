"""
Based on OPENCV Tutorials
https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
"""
import os
import cv2

import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

from edge_net import EdgeDetector


def generate_laplace(image_path, image_names, is_colored = False):
    colorspace = (cv2.IMREAD_COLOR if is_colored else cv2.IMREAD_GRAYSCALE)
    # delete all previous edge images
    for image_name in image_names:
        if 'LAPLACE_' in image_name:
            os.remove(os.path.join(image_path, image_name))
    
    for image_name in image_names:
        if 'LAPLACE_' in image_name:
            continue
        original_img = cv2.imread(os.path.join(image_path, image_name), colorspace)
        
        blurred_img = cv2.GaussianBlur(original_img, (3, 3), 0)
    
        laplacian_x = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        laplacian_y = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)

        laplacian_x = cv2.convertScaleAbs(laplacian_x)
        laplacian_y = cv2.convertScaleAbs(laplacian_y)

        laplacian = cv2.addWeighted(laplacian_x, 0.5, laplacian_y, 0.5, 0)
        
        cv2.imwrite(os.path.join(image_path, ('LAPLACE_' + image_name)), laplacian)

def call_laplace_on_folder(path, is_colored=False):
    
    image_names = os.listdir(path)
    generate_laplace(path, image_names, is_colored)

def call_laplace_on_folders(paths, is_colored=False):
    for path in paths:
        call_laplace_on_folder(path, is_colored)
    
def main(image_dir='test/', image_name='IMG_1044.jpg'):
    image_path = image_dir
    original_img = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
    assert original_img is not None, "file not found"
    
    tensor_img = transforms.ToTensor()(original_img)
    blurred_img = cv2.GaussianBlur(original_img, (3, 3), 0)
    
    laplacian_x = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    laplacian_y = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    
    laplacian_x = cv2.convertScaleAbs(laplacian_x)
    laplacian_y = cv2.convertScaleAbs(laplacian_y)
    
    laplacian = cv2.addWeighted(laplacian_x, 0.5, laplacian_y, 0.5, 0)

    inShape = (original_img.shape[0], original_img.shape[1], 1)
    outShape = (inShape[0]-2, inShape[1]-2, inShape[2])

    EDN = EdgeDetector(inShape=inShape, outShape=outShape, bias= False)
    edn_tensor = EDN.forward(tensor_img)
    edn_image = edn_tensor.data.numpy().transpose((1,2,0))


    print(original_img.shape)
    print(tensor_img.shape)
    print(edn_tensor.shape)
    print(edn_image.shape)


    plt.imshow(original_img, cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.figure()
    plt.imshow(edn_image, cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.figure()
    plt.imshow(laplacian, cmap = 'gray'), plt.xticks([])

    plt.show()

if __name__ == '__main__':
    # Test generator
    # main()    
    
    # Generate data
    call_laplace_on_folders(('train/', 'validation/', 'test/'))