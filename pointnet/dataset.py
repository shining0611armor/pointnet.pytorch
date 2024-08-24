from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
from torch.utils.data import Dataset
import glob
import h5py
import torch.utils.data
import requests
import zipfile


#----------------------------shapenet dataset-------------------------

def download_shapenet(root):
    # Ensure the root directory exists
    if not os.path.exists(root):
        os.mkdir(root)
    
    #Check if the dataset directory already exists
    dataset_dir = os.path.join(root, 'shapenetcorev2_hdf5_2048')
    if not os.path.exists(dataset_dir):
        url = 'https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/?dl=1'
        local_zip_file = os.path.join(root, 'shapenetcorev2_hdf5_2048.zip')
        print("Downloading ZIP file...")
        # Download the ZIP file
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful
        #  Save the ZIP file to the specified directory
        with open(local_zip_file, 'wb') as f:
            f.write(response.content)
        print("ZIP file downloaded successfully!")
        #  Extract the ZIP file
        print("Extracting ZIP file...")
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(root)  # Extract contents into the root directory
        print("ZIP file extracted successfully!")
        #  Move the extracted files to the dataset directory
        extracted_folder = os.path.join(root, 'shapenetcorev2_hdf5_2048')
        if not os.path.exists(dataset_dir):
            os.rename(extracted_folder, dataset_dir)
        else:
            print("Directory already exists. Skipping move.")
        #  Clean up by removing the downloaded ZIP file
        os.remove(local_zip_file)
        print("Downloaded ZIP file removed.")
    else:
        print("Dataset already exists. No download needed.")


def load_data_shapenet(root, partition):
    root = os.path.join(root, 'data')
    download_shapenet(root)
    all_data = []
    all_label = []
    g = sorted(glob.glob(os.path.join(root, 'shapenetcorev2_hdf5_2048', '%s*.h5' % partition)))
    for h5_name in g:
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        #print(all_data)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


#-------------------------------ModelNet Dataset-----------------------

def download_modelnet(root):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], root))
        os.system('rm %s' % (zipfile))

# download('/content/pointnet.pytorch')

def load_data_modelnet(root, partition):
    root = os.path.join(root, 'data')
    download_modelnet(root)
    all_data = []
    all_label = []
    g = sorted(glob.glob(os.path.join(root, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)))
    for h5_name in g:
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        # print(all_data)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
'''
# Use the load_data function
root_dir = '/content/pointnet.pytorch'
partition = 'train'  # or 'test'

data, labels = load_data_shapenet(root_dir, partition)

# Verify the data
print(f'Data shape: {data.shape}')
print(f'Labels shape: {labels.shape}')
'''
#-----------------------------------Dataset class for all datasets---------------------- 

class DatasetClass(Dataset):
    def __init__(self, root, dataset_name='modelnet' , npoints=2500, split='train', data_augmentation=True):
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.split = split
        self.dataset_name = dataset_name
        self.data, self.labels = None, None
        
        if self.dataset_name.lower() == 'modelnet':
            self.data, self.labels = load_data_modelnet(root, split)
            self.cat = {i: i for i in range(40)}  # ModelNet40 has 40 classes
        elif self.dataset_name.lower() == 'shapenet':
            self.data, self.labels = load_data_shapenet(root, split)
            self.cat = {i: i for i in range(55)}  # ShapeNet has 55 classes
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Choose 'modelnet' or 'shapenet'.")

        # List of class indices
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        point_set = self.data[index]
        cls = self.labels[index]

        # Randomly select points (if npoints < 2048)
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Center and normalize the point set
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        # Apply data augmentation if specified
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.data)






if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = DatasetClass(root=datapath, dataset_name = dataset)
        print(len(d))
        print(d[0])

    if dataset == 'modelnet':
        d = DatasetClass(root=datapath, dataset_name = dataset)
        print(len(d))
        print(d[0])

