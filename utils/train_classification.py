from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import DatasetClass
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = DatasetClass(
        root=opt.dataset,
        dataset_name = 'shapenet',
        npoints=opt.num_points,
        split='train')

    test_dataset = DatasetClass(
        root=opt.dataset,
        dataset_name = 'shapenet',
        npoints=opt.num_points,
        split='test',
        data_augmentation=False)
        
elif opt.dataset_type == 'modelnet40':
    dataset = DatasetClass(
        root=opt.dataset,
        dataset_name = 'modelnet',
        npoints=opt.num_points,
        split='train')

    test_dataset = DatasetClass(
        root=opt.dataset,
        dataset_name = 'modelnet',
        npoints=opt.num_points,
        split='test',
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('number of classes : ', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

for epoch in range(opt.nepoch):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training loop
    classifier.train()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target.squeeze()  # Ensure target is 1D
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()

        # Forward pass
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)  # Target should be 1D after squeezing

        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct_train += pred_choice.eq(target).cpu().sum().item()
        total_train += target.size(0)

    # Calculate and print training metrics
    train_loss = running_loss / len(dataloader)
    train_accuracy = (correct_train / total_train) * 100  # Convert to percentage
    print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    
    # Validation loop
    classifier.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            target = target.squeeze()  # Ensure target is 1D
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            # Forward pass
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)  # Ensure target is 1D after squeezing

            # Accumulate loss and accuracy
            running_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct_test += pred_choice.eq(target).cpu().sum().item()
            total_test += target.size(0)

    # Calculate and print validation metrics
    test_loss = running_loss / len(testdataloader)
    test_accuracy = (correct_test / total_test) * 100  # Convert to percentage
    print(f'Epoch {epoch+1}, Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%')

    # Save the model after each epoch
    torch.save(classifier.state_dict(), f'{opt.outf}/cls_model_{epoch+1}.pth')
