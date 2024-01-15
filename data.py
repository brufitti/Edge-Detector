import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from generator import call_laplace_on_folder

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.images_paths = []
        self.labels_paths = []
        for file in sorted(os.listdir(dataset_path)):
                if os.path.basename(file).startswith('LAPLACE_'):
                    self.labels_paths.append(os.path.join(dataset_path, file))
                else:
                    self.images_paths.append(os.path.join(dataset_path, file))
    
    def __getitem__(self, idx):
        image = transforms.ToTensor()(cv2.imread(self.images_paths[idx], cv2.IMREAD_GRAYSCALE))
        label = transforms.ToTensor()(cv2.imread(self.labels_paths[idx], cv2.IMREAD_GRAYSCALE))
        return image, label
    
    def __len__(self):
        return len(self.images_paths)
    
def get_dataset_from(path, in_colors=False):
    colorspace = (cv2.IMREAD_COLOR if in_colors else cv2.IMREAD_GRAYSCALE)
    
    number_of_pictures = 0
    dataset_x = torch.Tensor()
    dataset_y = torch.Tensor()
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        laplace_path = os.path.join(path, ('LAPLACE_' + file)) 
        if 'LAPLACE_' in image_path:
            continue
        assert laplace_path is not None,  ('Laplace not found for file: ' + file)
        if os.path.isfile(image_path):
            # count number of files in folder
            number_of_pictures += 1
            img = cv2.imread(image_path, colorspace)
            tensor_img = transforms.ToTensor()(img)
            laplace = cv2.imread(laplace_path, colorspace)
            tensor_laplace = transforms.ToTensor()(laplace)
            dataset_x.append(tensor_img)
            dataset_y.append(tensor_laplace)
    print('Pictures located for dataset: ', number_of_pictures)
    dataset = TensorDataset(dataset_x,dataset_y)
    return dataset

def generate_dataset(path=None):
    if path == None:
        _ = call_laplace_on_folder('train/')
        _ = call_laplace_on_folder('validation/')
        return None
    else:
        call_laplace_on_folder(path)

if __name__ == '__main__':
    # generate_dataset()
    # _ = get_dataset_from('train/')
    # _ = get_dataset_from('validation/')
    
    teste = MyDataset('train')
    print(teste.__len__())
    teste.__getitem__(2)
    