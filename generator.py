"""
Based on OPENCV Tutorials
"""
import os
import cv2

import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

from edge_net import EdgeDetector


def generate_laplace(image_path, image_names, is_colored = False):
    colorspace = (cv2.IMREAD_COLOR if is_colored else cv2.IMREAD_GRAYSCALE)
    for image_name in image_names:
        original_img = cv2.imread(os.path.join(image_path, image_name), colorspace)
        # tensor_img = transforms.ToTensor()(original_img)
        laplacian1 = cv2.Sobel(src=original_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        laplacian1 = cv2.Sobel(src=laplacian1, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        laplacian2 = cv2.Sobel(src=original_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        laplacian2 = cv2.Sobel(src=laplacian2, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
        laplacian = laplacian1 + laplacian2
        cv2.imwrite(os.path.join(image_path, ('LAPLACE_' + image_name)), laplacian)

def call_laplace_on_folder(path, is_colored = False):
    
    image_names = os.listdir(path)
    generate_laplace(path, image_names, is_colored)


def main():
    # original_img = cv2.imread('../headshots/1.png', cv2.IMREAD_GRAYSCALE)
    image_name = 'IMG_1044.jpg'
    image_path = 'validation/'
    original_img = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)
    tensor_img = transforms.ToTensor()(original_img)
    assert original_img is not None, "file not found"

    laplacian1 = cv2.Sobel(src=original_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    laplacian1 = cv2.Sobel(src=laplacian1, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    laplacian2 = cv2.Sobel(src=original_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    laplacian2 = cv2.Sobel(src=laplacian2, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    laplacian = laplacian1 + laplacian2
    laplacial_original = cv2.Laplacian(src=original_img, )
    # cv2.imwrite('validation/IMG_1044_LAPLACE.jpg', laplacian)

    inShape = (original_img.shape[0], original_img.shape[1], original_img.shape[2])
    outShape = (inShape[0]-2, inShape[1]-2, inShape[2])

    EDN = EdgeDetector(inShape=inShape, outShape=outShape, bias= False)
    edn_tensor = EDN.forward(tensor_img)
    edn_image = edn_tensor.data.numpy().transpose((1,2,0))


    print(original_img.shape)
    print(tensor_img.shape)
    print(edn_tensor.shape)
    print(edn_image.shape)


    plt.imshow(original_img, cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.figure()
    plt.imshow(edn_image, cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.yticks([]), plt.figure()
    plt.imshow(laplacian, cmap = 'gray'), plt.xticks([])

    plt.show()

if __name__ == '__main__':
    main()
    # call_laplace_on_folder('train/')
    # call_laplace_on_folder('validation/')