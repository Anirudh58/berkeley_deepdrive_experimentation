import os
import pickle
from PIL import Image
import random

import torch


# config vars
root = '.'

dataset_path = os.path.join(root, 'dataset')
train_dataset_path = os.path.join(dataset_path, 'train')
val_dataset_path = os.path.join(dataset_path, 'val')

val_split = 0.2

target_labels = ['car', 'traffic sign', 'pedestrian']
#target_labels = ['car']

# create a map for label->id
label_id_map = {}
id_label_map = {}
id_color_map = {}
colors = ['r', 'b', 'g']
for i in range(1, len(target_labels)+1):
    label_id_map[target_labels[i-1]] = i
    id_label_map[i] = target_labels[i-1]
    color = colors[i-1]
    id_color_map[i] = color

class BDDDataset(object):
    
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # load the train set calculated from prev notebook
        with open('train_set.pkl', 'rb') as f:
            self.train_set = pickle.load(f)
            
        self.imgs = list(os.listdir(train_dataset_path))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        # load image
        img_name = self.imgs[idx]
        img_path = os.path.join(train_dataset_path, img_name)
        img = Image.open(img_path).convert('RGB')
        
        labels = []
        boxes = []
        
        for instance in self.train_set[img_name]['labels']:
            # making sure we dont have visually unclear instances
            if instance['category'] in label_id_map and not instance['attributes']['occluded'] and not instance['attributes']['truncated']:
                labels.append(label_id_map[instance['category']])
                boxes.append([instance['box2d']['x1'], instance['box2d']['y1'], instance['box2d']['x2'], instance['box2d']['y2']])
                
        num_instances = len(labels)
        
        # convert all variables to tensors
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_instances,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # attach all info into a dict target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
                
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target