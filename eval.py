import os
import cv2
import torch

import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from edge_net import EdgeDetector

def main(load_path, image_path, laplace_path):
    model = EdgeDetector()
    
    checkpoint = torch.load(load_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplace_image = cv2.imread(laplace_path, cv2.IMREAD_GRAYSCALE)
    model_image = model((transforms.ToTensor()(original_image)).cuda())
    plt.imshow(original_image, cmap = 'gray') ,plt.xticks([]) ,plt.yticks([]), plt.title("original_image") ,plt.figure()
    plt.imshow(laplace_image, cmap = 'gray')  ,plt.xticks([]) ,plt.yticks([]), plt.title("edge_image") ,plt.figure()
    plt.imshow(model_image.cpu().data.numpy().transpose((1,2,0)), cmap = 'gray'), plt.xticks([]), plt.yticks([]), plt.title("network_image"), plt.show()


if __name__ == '__main__':
    weights_path = 'models/model-5000.pt'
    image_path = 'train/1.png'
    laplace_path = 'train/LAPLACE_1.png'
    main(weights_path, image_path, laplace_path)