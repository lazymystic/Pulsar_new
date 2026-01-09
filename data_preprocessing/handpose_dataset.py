from config import CFG
import sklearn.preprocessing
import numpy as np
from ast import literal_eval as make_tuple
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from utils import adj_mat

def df_to_numpy(df):
  print("DF TO NUMPY")
  x_elems = df.drop("LABEL", axis=1).values
  print("X ELEMS SHAPE: ", x_elems.shape)
  x = [make_tuple(elem) for elem in x_elems for elem in elem]
  x = np.array(x)
  print("X SHAPE: ", x.shape)
  x = x.reshape(len(x_elems), 3*21)
  print("X SHAPE: ", x.shape)
  lb = sklearn.preprocessing.LabelBinarizer().fit(CFG.classes)
  y = np.array(df["LABEL"]).reshape(-1,1)
  print("Y SHAPE: ", y.shape)
  y_ohe = lb.transform(y)
  y_ohe = np.hstack((y_ohe,1-y_ohe)) ### ADJUSTMENT FOR CLASS = 2
  print("Y OHE SHAPE: ", y_ohe.shape)

  train_data_numpy = (x, y_ohe)
  return train_data_numpy

class HandPoseDatasetNumpy(Dataset):
    def __init__(self, data, bone_stream=CFG.bone_stream, vel_stream=CFG.vel_stream, acc_stream=CFG.acc_stream):
        self.data = data
        self.bone_stream = bone_stream
        self.vel_stream = vel_stream
        self.acc_stream = acc_stream
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        x = self.data[0]
        y = self.data[1]
        seq_x = x[idx:idx+CFG.sequence_length]
        seq_y = y[idx:idx+CFG.sequence_length]

        #padding
        to_pad = CFG.sequence_length - seq_x.shape[0]
        x_pad = np.pad(seq_x, ((0, to_pad), (0, 0)), mode='mean')
        y_pad = np.pad(seq_y, ((0, to_pad), (0, 0)), mode='mean')
        
        x = x_pad.reshape(CFG.sequence_length, 21, 3)
        
        if CFG.add_feats:
            #additional features: https://link.springer.com/content/pdf/10.1186/s13640-019-0476-x.pdf
            #r = np.sqrt(x**2 + y**2 + z**2)
            #theta = np.arccos(z/r)
            #phi = np.arctan(y/x)
            theta = phi = r = np.zeros_like(x[:, :, 0])

            if x.all() != 0:
              r = np.sqrt(x[:, :, 0]**2 + x[:, :, 1]**2 + x[:, :, 2]**2)
              theta = np.arccos(x[:, :, 2]/r)
              phi = np.arctan(x[:, :, 1]/x[:, :, 0])
            add_feats = np.stack([r, theta, phi], axis=-1)
            
            x = np.concatenate((x, add_feats),axis=2)

        if self.bone_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            x = dist

        if self.vel_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            vel = np.gradient(dist, axis=0)
            x = vel
            
        if self.acc_stream:
            dist = np.zeros_like(x)
            for v1, v2 in adj_mat.inward:
                dist[:, v1, :] = x[:, v2, :] - x[:, v1, :]
            vel = np.gradient(dist, axis=0)
            acc = np.gradient(vel, axis=0)
            x = acc

        if CFG.add_phi:
            phi = np.arctan(x[:, :, 1]/x[:, :, 0])
            x = np.concatenate((x, phi),axis=2)

        return x, y_pad


class HandImageDataset(Dataset):
    def __init__(self, df, root_dir,  seq_len=16, transforms=False):
        self.data_frames = df
        self.labels = self.data_frames["label"]
        self.img_paths = self.data_frames["file"]
        self.classes = ["Grasp",   "Move",    "Negative",    "Position",    "Reach",   "Release"]
        if CFG.no_release:
            self.classes = ["Grasp",   "Move",    "Negative",    "Position",    "Reach"]
        self.lb = sklearn.preprocessing.LabelBinarizer().fit(self.classes)
        self.seq_len = seq_len
        self.root_dir = root_dir
        self.transforms = transforms

        self.rand_rot_range = (-15,15)
        self.resize = (568,568)
        self.rand_crop_size = (512,512)
    
    def set_transform_params(self):
        self.rrc = transforms.RandomResizedCrop(self.rand_crop_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))
        print(self.rrc)
        self.crop_indices = self.rrc.get_params(torch.rand(3,*self.rand_crop_size), self.rrc.scale, self.rrc.ratio)

        self.rot_angle = np.random.randint(*self.rand_rot_range)
        self.p_hflip = random.random()
        self.p_vflip = random.random()

    def do_transform(self, image):
        resize = transforms.Resize(size=self.resize)
        image = resize(image)
        #center_crop = transforms.CenterCrop(size=(850,850))
        #image = center_crop(image)
        
        i, j, h, w = self.crop_indices
        image = transforms.functional.resized_crop(image, i, j, h, w, self.rrc.size)

        image = transforms.functional.rotate(image, self.rot_angle)

        if self.p_hflip > 0.5:
            image = transforms.functional.hflip(image)
        if self.p_vflip > 0.5:
            image = transforms.functional.vflip(image)

        image = transforms.functional.to_tensor(image)
        return image
    
    def __len__(self):
        return len(self.data_frames)
    
    def __getitem__(self, idx):
        start = idx
        if start > len(self.data_frames) - self.seq_len:
            start = len(self.data_frames) - self.seq_len
        end = start + self.seq_len    
        indices = list(range(start, end))
        images = []
        labels = []
        self.set_transform_params()

        for i in indices:
            image_path = self.root_dir + self.img_paths[i]
            image = Image.open(image_path)
            if self.transforms:
                image = self.do_transform(image)

            images.append(image)

        y = torch.tensor(self.lb.transform(self.labels[start:end]), dtype=torch.long)
        img_seq = torch.stack(images)

        return img_seq, y
