import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from edge_net import EdgeDetector
from data import MyDataset


def train(args, model, train_dataset, validation_dataset):
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr= args.learning_rate)
    
    criterion = torch.nn.MSELoss()
    
    train_dataLoader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=0)
    
    validation_dataLoader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=args.validation_batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)
    
    start_epoch = 0
    
    # Resume training from checkpoint
    if args.resume_path is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            checkpoint = torch.load(args.resume_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        
    
        
    for epoch in range(start_epoch, args.epochs + start_epoch):
        
        # Training
        # ------------------------
        model.train()
        train_loss = 0
        for image,laplace in train_dataLoader:
            
            if torch.cuda.is_available():
                image   = image.cuda()
                laplace = laplace.cuda()
            
            output_image = model(image)
            loss = criterion(output_image, laplace)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss
            
        train_loss = train_loss/len(train_dataset)
        
        # Validation 
        # ------------------------
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for image,laplace in validation_dataLoader:
            
                if torch.cuda.is_available():
                    image   = image.cuda()
                    laplace = laplace.cuda()
                
                output_image = model(image)
                loss = criterion(output_image, laplace)
                optimizer.zero_grad()
                
                validation_loss += loss
                
            validation_loss = validation_loss/len(validation_dataset)
        
        # Print current loss and save dict ever 100th epoch
        # ------------------------
        if (epoch != 0 and epoch % 100 == 0) or (epoch == args.epochs + start_epoch - 1):
            print("epoch: ", epoch)
            print("training Loss: ", train_loss.item())
            print("Validation Loss: ", validation_loss.item())
            
            state = {
                'epoch'    : epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            
            torch.save(state, os.path.join(args.model_path, 'model-{}.pt'.format(state['epoch'])))


def main(args):
    train_set = MyDataset(args.train_path)
    validation_set = MyDataset(args.validation_path)
    example_image = train_set.__getitem__(0)[0]
    in_shape = (100,100,1)
    out_shape = (in_shape[0]-2, in_shape[1]-2, in_shape[2])
    model = EdgeDetector(inShape=in_shape, outShape=out_shape)
    train(args, model, train_set, validation_set)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir'    ,help='train data directory'               ,dest='train_path'            ,type=str   ,default='train/')
    parser.add_argument('--val_dir'      ,help='validation data directory'          ,dest='validation_path'       ,type=str   ,default='validation/')
    parser.add_argument('--model_dir'    ,help='model directory'                    ,dest='model_path'            ,type=str   ,default='models/')
    parser.add_argument('--resume_dir'   ,help='if resuming, set path to checkpoint',dest='resume_path'           ,type=str   ,default=None)
    parser.add_argument('--train_batch'  ,help='train batch size'                   ,dest='train_batch_size'      ,type=int   ,default=20)
    parser.add_argument('--val_batch'    ,help='validation batch size'              ,dest='validation_batch_size' ,type=int   ,default=6)
    parser.add_argument('--epochs'       ,help='number of epochs to train for'      ,dest='epochs'                ,type=int   ,default=10001)
    parser.add_argument('--learning_rate',help='learning rate'                      ,dest='learning_rate'         ,type=float ,default=1e-3)
    args = parser.parse_args()
    
    main(args)