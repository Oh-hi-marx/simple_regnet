

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import cv2
import wandb
import os
import glob
import re
import numpy as np
import random
from PIL import Image

def visulise(tensor,step):
    os.makedirs('visualise', exist_ok=True)
    print(tensor[0].shape)
    for i in range(tensor.shape[0]):
        img = tensor[i].permute(1, 2, 0)  .cpu().detach().numpy()[:, :, ::-1]
        cv2.imwrite("visualise/"+str(step) +"_" + str(i)+ ".jpg", img*255)

class qualityFace(Dataset):
    def __init__(self, imgs, transforms):
        self.imgs= imgs
        self.transforms = transforms
    def __len__(self):
        return (len(self.imgs))

    def __getitem__(self, i):
       img = self.imgs[i]
       img = Image.open(img)

       img = self.transforms(img)
       label = [0,0,0,0,0]
       return img, torch.FloatTensor(label)

if __name__ == "__main__":
    wandb.init(project="utkface-regnet-combined")
    imgs = glob.glob("ffhq/*")
    epochs = 100
    device = 'cuda'
    numClass = 5
    learning_rate = 0.0001

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=(0.6,1.2),contrast=(0.3),saturation=(0.8,1.2),hue=(-0.05,0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 2)),
        transforms.RandomAdjustSharpness(1.4, p=0.2),
        transforms.RandomAutocontrast(p=0.1),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.05),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),

    ])

    dataset = qualityFace(imgs[:-10], transform)
    dataloader = DataLoader(dataset, batch_size = 100, num_workers = 5)
    testdataset = qualityFace(imgs[-10:], transform)
    testloader = DataLoader(testdataset, batch_size = 100, num_workers = 5)

    model = models.regnet_y_400mf(weights = "IMAGENET1K_V2")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numClass)
    model = nn.Sequential(model, nn.Sigmoid())
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('checkpoints', exist_ok=True)

    step = 0
    for epoch in range(epochs):

        for batchi, (inputs, labels) in enumerate(dataloader):
            #visulise(inputs, batchi)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients

            optimizer.step()

            if(step%100==0):
                wandb.log({ 'train loss': loss}, step = step)
            step+= 1

        ############ TEST ###############
        with torch.no_grad():
            loss =0;correct = 0;total = 0

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                total += labels.size(0)
                loss += criterion(outputs, labels)
                correct += torch.sum(torch.abs(outputs.data - labels.data) <0.5)
            accuracy = (   100 * correct / (total*numClass))
            print('Accuracy of the network on test images: %0.3f %%' % accuracy)
            wandb.log({'test accuracy': accuracy, 'epoch': epoch, 'test loss': loss})


        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
            'checkpoints' + os.sep + str(epoch)+'_'+ str(round(accuracy.item(),2)) + '.pth')
