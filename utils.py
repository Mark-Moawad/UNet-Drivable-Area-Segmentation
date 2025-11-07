# basic imports
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple

# DL library imports
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# For dice loss function
import segmentation_models_pytorch as smp

###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

# Inverse transformation to convert back to normal RGB format
inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


# Constants for Standard color mapping for 3-class drivable area segmentation
# Reference: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Label = namedtuple("Label", ["name", "train_id", "color"])

# BDD100K Drivable Area has 3 classes
# Direct lane (ego lane), Alternative lane (adjacent lanes), Background
drivables = [
    Label("direct", 0, (171, 44, 236)),
    Label("alternative", 1, (86, 211, 19)),
    Label("background", 2, (0, 0, 0)),
]

# Create color mapping array for visualization
train_id_to_color = np.array([c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)])

# Number of classes
NUM_CLASSES = 3


###################################
# DATASET CLASS
###################################

class BDD100k_dataset(Dataset):
    def __init__(self, images, labels, tf=None):
        self.images = images
        self.labels = labels
        self.tf = tf
    
    def __len__(self):
        return len(self.images)
  
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.tf is not None:
            image = self.tf(image)
        
        label = torch.from_numpy(label).long()
        
        return image, label


###################################
# DATASET UTILITIES
###################################

def get_datasets(images, labels):
    n_samples = len(images)
    train_end = int(0.7 * n_samples)
    val_end = int(0.9 * n_samples)
    
    train_images = images[:train_end]
    train_labels = labels[:train_end]
    
    val_images = images[train_end:val_end]
    val_labels = labels[train_end:val_end]
    
    test_images = images[val_end:]
    test_labels = labels[val_end:]
    
    train_set = BDD100k_dataset(train_images, train_labels, tf=preprocess)
    val_set = BDD100k_dataset(val_images, val_labels, tf=preprocess)
    test_set = BDD100k_dataset(test_images, test_labels, tf=preprocess)
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")
    
    return train_dataloader, val_dataloader, test_dataloader


###################################
# METRIC CLASS
###################################

class meanIoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.iou_metric = 0.0
        
    def update(self, y_preds, labels):
        predicted_labels = torch.argmax(y_preds, dim=1)
        
        batch_ious = []
        for i in range(self.num_classes):
            tp = torch.sum((predicted_labels == i) & (labels == i)).float()
            fp = torch.sum((predicted_labels == i) & (labels != i)).float()
            fn = torch.sum((predicted_labels != i) & (labels == i)).float()
            
            denom = tp + fp + fn
            if denom == 0:
                batch_ious.append(float('nan'))
            else:
                batch_ious.append((tp / denom).item())
        
        valid_ious = [iou for iou in batch_ious if not np.isnan(iou)]
        if valid_ious:
            self.iou_metric = np.mean(valid_ious)
        else:
            self.iou_metric = 0.0
    
    def compute(self):
        return self.iou_metric


###################################
# EVALUATION FUNCTION
###################################

def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    model.eval()
    total_loss = 0.0
    metric = metric_class(num_classes)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            total_loss += loss.item()
            
            metric.update(y_preds, labels)
    
    avg_loss = total_loss / len(dataloader)
    avg_metric = metric.compute()
    
    return avg_loss, avg_metric


###################################
# VISUALIZATION FUNCTIONS
###################################

def plot_training_results(results, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(results['epoch'], results['trainLoss'], label='Train Loss')
    axes[0].plot(results['epoch'], results['validationLoss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    metric_col = [col for col in results.columns if col not in ['epoch', 'trainLoss', 'validationLoss']][0]
    axes[1].plot(results['epoch'], results[metric_col], label=metric_col, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_col)
    axes[1].set_title(f'{model_name} - {metric_col}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {model_name}_training_curves.png")
    plt.close()


def visualize_predictions(model, dataset, axes, device, numTestSamples=5, id_to_color=None):
    if id_to_color is None:
        id_to_color = train_id_to_color
    
    model.eval()
    indices = np.random.choice(len(dataset), numTestSamples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            
            image_input = image.unsqueeze(0).to(device)
            prediction = model(image_input)
            predicted_label = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
            
            image_rgb = inverse_transform(image).permute(1, 2, 0).cpu().numpy()
            image_rgb = np.clip(image_rgb, 0, 1)
            
            label_rgb = id_to_color[label.cpu().numpy()] / 255.0
            pred_rgb = id_to_color[predicted_label] / 255.0
            
            if numTestSamples == 1:
                axes[0].imshow(image_rgb)
                axes[0].set_title('Image')
                axes[0].axis('off')
                
                axes[1].imshow(label_rgb)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_rgb)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
            else:
                axes[i, 0].imshow(image_rgb)
                axes[i, 0].set_title('Image' if i == 0 else '')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(label_rgb)
                axes[i, 1].set_title('Ground Truth' if i == 0 else '')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(pred_rgb)
                axes[i, 2].set_title('Prediction' if i == 0 else '')
                axes[i, 2].axis('off')


def predict_video(model, model_name, video_path, output_dir, 
                  target_width=320, target_height=180, device='cpu', 
                  train_id_to_color=None):
    if train_id_to_color is None:
        train_id_to_color = globals()['train_id_to_color']
    
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f'{video_name}_{model_name}_segmented.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width * 2, target_height))
    
    print(f"Processing {frame_count} frames...")
    
    with torch.no_grad():
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (target_width, target_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            frame_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
            
            prediction = model(frame_tensor)
            predicted_label = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
            
            pred_rgb = train_id_to_color[predicted_label].astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
            
            combined = np.hstack([frame_resized, pred_bgr])
            
            out.write(combined)
    
    cap.release()
    out.release()
    
    return output_path
