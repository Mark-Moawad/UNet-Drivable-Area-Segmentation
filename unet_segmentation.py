"""
This module implements a drivable area segmentation pipeline using a UNet architecture on image and video data
based on the original paper.

It provides utilities to load pre-processed BDD100K drivable area dataset, define a custom PyTorch dataset,
create data loaders, build and train a UNet model, evaluate performance using the mean Intersection
over Union (IoU) metric, visualize sample predictions, and process video inputs for segmentation.

Dataset:
    Pre-processed BDD100K Drivable Area Segmentation dataset (3 classes):
    Automatically downloaded from Google Drive on first run.
    
    The dataset contains 3 classes for drivable area understanding:
    0: direct (ego lane - current lane)
    1: alternative (adjacent lanes - other drivable lanes)
    2: background (non-drivable areas)

Key Components:
    - Model Architecture:
        • double_conv(): Defines a double convolution block used in the UNet architecture.
        • UNetEncoder: Implements the encoder part of UNet with skip connections.
        • UNetDecoder: Implements the decoder part that up-samples and combines skip connections.
        • UNet: Assembles the complete UNet model with a bottleneck and final 1x1 convolution layer.
    - Dataset and DataLoader:
        • BDD100k_dataset: Custom Dataset class for loading images and labels.
        • get_datasets(): Splits the dataset into training, validation, and test subsets.
        • get_dataloaders(): Creates corresponding DataLoader objects for batch processing.
    - Training and Evaluation:
        • train_validate_model(): Handles training loop, validation, and model checkpointing 
          based on minimum validation loss with early stopping.
        • evaluate_model(): Evaluates model performance on validation or test data sets.
        • meanIoU: A class that computes mean Intersection over Union (IoU) for segmentation.
    - Visualization and Post-Processing:
        • inverse_transform(): Reverses normalization for visualizing images.
        • plot_training_results(): Plots training and validation loss curves and metric progress.
        • visualize_predictions(): Displays predictions alongside the original images and ground-truth labels.
        • predict_video(): Processes an input video frame-by-frame and writes segmentation predictions to a new video file.
    - Main Routine:
        • main(): Orchestrates data preparation, model training, evaluation on test set, visualization
          of predictions, and optional video processing.

Dependencies:
    Python libraries: os, numpy, torch, torchvision, cv2 (OpenCV), matplotlib, segmentation_models_pytorch.

Usage:
    Run the module as a script to perform the complete pipeline:
        $ python unet_segmentation.py
    
    The script loads data, trains the UNet model, evaluates it, and can process
    videos for segmentation visualization.

Original paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox
can be found at: https://arxiv.org/abs/1505.04597
"""

import os
import glob
import zipfile
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Import utility functions from utils.py
from utils import (
    inverse_transform, meanIoU, plot_training_results, 
    evaluate_model, visualize_predictions, predict_video,
    get_datasets, get_dataloaders, train_id_to_color
)

# Optional imports (will be imported dynamically if needed)
try:
    import gdown
except ImportError:
    gdown = None

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Number of classes for drivable area segmentation
NUM_CLASSES = 3

# Training configuration
MAX_EPOCHS = 100  # Maximum epochs - training will stop early if conditions are met
TARGET_METRIC = 0.85  # Target meanIoU for early stopping
TARGET_LOSS = 0.15    # Target validation loss for early stopping

# Define paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_PATH = os.path.join(DATA_DIR, "outputs")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


###################################
# UNET MODEL ARCHITECTURE
###################################

def double_conv(in_channels, out_channels):
    """Double convolution block used in UNet architecture"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNetEncoder(nn.Module):
    """Encoder part of UNet with skip connections"""
    def __init__(self, in_channels, layer_channels):
        super(UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        # Double Convolution blocks
        for num_channels in layer_channels:
            self.encoder.append(double_conv(in_channels, num_channels))
            in_channels = num_channels

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Pass input image through Encoder blocks
        # and return outputs at each stage
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections


class UNetDecoder(nn.Module):
    """Decoder part of UNet"""
    def __init__(self, layer_channels):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.ModuleList()

        # Decoder layer Double Convolution blocks
        # and upsampling blocks
        self.decoder = nn.ModuleList()
        for num_channels in reversed(layer_channels):
            self.decoder.append(nn.ConvTranspose2d(num_channels*2, num_channels,
                                                   kernel_size=2, stride=2))
            self.decoder.append(double_conv(num_channels*2, num_channels))

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.decoder), 2):
            # Upsample output and reduce channels by 2
            x = self.decoder[idx](x)

            # If skip connection shape doesn't match, resize
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatenate and pass through double_conv block
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        return x


class UNet(nn.Module):
    """Complete UNet architecture"""
    def __init__(self, in_channels, out_channels, layer_channels):
        super(UNet, self).__init__()

        # Encoder and decoder modules
        self.encoder = UNetEncoder(in_channels, layer_channels)
        self.decoder = UNetDecoder(layer_channels)

        # conv layer to transition from encoder to decoder and
        # 1x1 convolution to reduce num channels to out_channels
        self.bottleneck = double_conv(layer_channels[-1], layer_channels[-1]*2)
        self.final_conv = nn.Conv2d(layer_channels[0], out_channels, kernel_size=1)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, x):
        # Encoder blocks
        encoder_output, skip_connections = self.encoder(x)

        # transition between encoder and decoder
        x = self.bottleneck(encoder_output)

        # we need the last skip connection first
        # so reversing the list
        skip_connections = skip_connections[::-1]

        # Decoder blocks
        x = self.decoder(x, skip_connections)

        # final 1x1 conv to match input size
        return self.final_conv(x)


###################################
# TRAINING FUNCTION WITH EARLY STOPPING
###################################

def train_validate_model(model, num_epochs, model_name, criterion, optimizer,
                         device, dataloader_train, dataloader_valid,
                         metric_class, metric_name, num_classes, 
                         lr_scheduler=None, output_path='.'):
    """
    Train and validate the model with early stopping.
    
    Args:
        model: PyTorch model to train
        num_epochs: Maximum number of training epochs
        model_name: Name for saving the model
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cpu/cuda)
        dataloader_train: Training dataloader
        dataloader_valid: Validation dataloader
        metric_class: Metric class for evaluation
        metric_name: Name of the metric
        num_classes: Number of output classes
        lr_scheduler: Optional learning rate scheduler
        output_path: Path to save model and results
        
    Returns:
        results: DataFrame with training history
    """
    # Initialize placeholders for running values
    results = []
    min_val_loss = np.inf
    len_train_loader = len(dataloader_train)

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        # Compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)
        validation_loss, validation_metric = evaluate_model(
            model, dataloader_valid, criterion, metric_class, 
            num_classes, device
        )

        print(f'Epoch: {epoch+1}, trainLoss: {train_loss:.5f}, '
              f'validationLoss: {validation_loss:.5f}, '
              f'{metric_name}: {validation_metric:.2f}')

        # Store results
        results.append({
            'epoch': epoch,
            'trainLoss': train_loss,
            'validationLoss': validation_loss,
            f'{metric_name}': validation_metric
        })

        # If validation loss has decreased, save model
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            model_path = os.path.join(output_path, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        # Early stopping conditions
        if validation_metric >= TARGET_METRIC:
            print(f"\n{'='*80}")
            print(f"Early stopping: {metric_name} reached target threshold!")
            print(f"Target: {TARGET_METRIC:.2f}, Achieved: {validation_metric:.2f}")
            print(f"{'='*80}\n")
            break
        
        if validation_loss <= TARGET_LOSS:
            print(f"\n{'='*80}")
            print(f"Early stopping: Validation loss reached target threshold!")
            print(f"Target: {TARGET_LOSS:.4f}, Achieved: {validation_loss:.4f}")
            print(f"{'='*80}\n")
            break

    # Plot results
    results = pd.DataFrame(results)
    plot_training_results(results, model_name)
    
    # Save training statistics
    stats_file = os.path.join(output_path, f"{model_name}_training_stats.csv")
    results.to_csv(stats_file, index=False)
    print(f"\nTraining statistics saved to: {stats_file}")
    
    return results


###################################
# MAIN EXECUTION
###################################

def create_model(num_classes=3):
    """Create and return UNet model for 3-class drivable area segmentation"""
    model = UNet(in_channels=3, out_channels=num_classes, 
                 layer_channels=[64, 128, 256, 512]).to(device)
    return model


def download_and_prepare_data():
    """Download dataset from Google Drive if not present"""
    images_path = os.path.join(DATASET_DIR, "image_180_320.npy")
    labels_path = os.path.join(DATASET_DIR, "label_180_320.npy")
    
    # Check if dataset exists, if not download from Google Drive
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"\nDataset not found. Downloading from Google Drive...")
        
        # Install gdown if not available
        global gdown
        if gdown is None:
            print("Installing gdown...")
            subprocess.check_call(['pip', 'install', 'gdown'])
            import gdown
        
        # Download and extract dataset
        print("Downloading dataset archive...")
        gdown.download("https://drive.google.com/uc?id=1sX6kHxpYoEICMTfjxxhK9lTW3B7OUxql", 
                      "segmentation.zip", quiet=False)
        
        print("Extracting dataset...")
        with zipfile.ZipFile("segmentation.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Move files from segmentation/dataset to data/dataset
        # The zip extracts to segmentation/dataset/*.npy
        source_dir = os.path.join("segmentation", "dataset")
        if os.path.exists(source_dir):
            print(f"Moving files from {source_dir} to {DATASET_DIR}...")
            for file in os.listdir(source_dir):
                if file.endswith('.npy'):
                    src = os.path.join(source_dir, file)
                    dst = os.path.join(DATASET_DIR, file)
                    shutil.move(src, dst)
                    print(f"  Moved: {file}")
            # Clean up temporary directories
            shutil.rmtree("segmentation")
            print("Cleaned up temporary segmentation directory")
        
        # Clean up zip file
        if os.path.exists("segmentation.zip"):
            os.remove("segmentation.zip")
            print("Removed segmentation.zip")
        
        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset found locally.")


def load_data():
    """Load and prepare pre-processed dataset"""
    print("Loading pre-processed dataset...")
    
    # Load images and labels
    images_path = os.path.join(DATASET_DIR, "image_180_320.npy")
    labels_path = os.path.join(DATASET_DIR, "label_180_320.npy")
    
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print("\nCreating train/val/test splits...")
    train_set, val_set, test_set = get_datasets(images, labels)
    
    print("\nCreating data loaders...")
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_set, val_set, test_set, batch_size=8)
    
    return train_dataloader, val_dataloader, test_dataloader, test_set


def train_model(model, train_dataloader, val_dataloader, num_epochs=MAX_EPOCHS, model_name='UNet_baseline'):
    """Train the UNet model"""
    # Setup loss function
    criterion = smp.losses.DiceLoss('multiclass', classes=[0, 1, 2], 
                                   log_loss=True, smooth=1.0)
    
    # Setup optimizer and learning rate scheduler
    MAX_LR = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=num_epochs,
                          steps_per_epoch=len(train_dataloader),
                          pct_start=0.3, div_factor=10, anneal_strategy='cos')
    
    # Train model
    print(f"\nTraining {model_name} for up to {num_epochs} epochs...")
    results = train_validate_model(
        model, num_epochs, model_name, criterion, optimizer,
        device, train_dataloader, val_dataloader,
        meanIoU, 'meanIoU', num_classes=NUM_CLASSES, lr_scheduler=scheduler,
        output_path=MODELS_DIR
    )
    
    return results


def evaluate_trained_model(model, test_dataloader, model_name='UNet_baseline'):
    """Evaluate trained model on test set"""
    # Load best model
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate
    criterion = smp.losses.DiceLoss('multiclass', classes=[0, 1, 2], 
                                   log_loss=True, smooth=1.0)
    _, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
    print(f"\nModel has {test_metric:.4f} mean IoU on test set")
    
    return test_metric


def visualize_test_predictions(model, test_set, num_samples=5):
    """Visualize model predictions on test set"""
    print(f"\nVisualizing {num_samples} predictions...")
    
    _, axes = plt.subplots(num_samples, 3, figsize=(3*6, num_samples * 4))
    visualize_predictions(model, test_set, axes, device, 
                         numTestSamples=num_samples, id_to_color=train_id_to_color)
    
    # Save figure
    output_file = os.path.join(OUTPUT_PATH, 'UNet_baseline_predictions.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to: {output_file}")
    # Don't show in non-interactive environments
    # plt.show()
    plt.close()


def process_videos(model, model_name='UNet_baseline'):
    """Process videos if available"""
    # Look for videos in common locations
    video_patterns = [
        os.path.join(DATASET_DIR, '*.mp4'),
        os.path.join(DATASET_DIR, '*.avi'),
        os.path.join(ROOT_DIR, '*.mp4'),
        os.path.join(ROOT_DIR, '*.avi'),
    ]
    
    video_files = []
    for pattern in video_patterns:
        video_files.extend(glob.glob(pattern))
    
    if not video_files:
        print(f"\nNo video files found for processing")
        return
    
    print(f"\nFound {len(video_files)} videos to process")
    
    for video_path in video_files:
        print(f"\nProcessing: {os.path.basename(video_path)}")
        try:
            output_path = predict_video(
                model, model_name, video_path, PROCESSED_DIR,
                target_width=320, target_height=180, device=device,
                train_id_to_color=train_id_to_color
            )
            print(f"Saved to: {output_path}")
        except Exception as e:
            print(f"Error processing {os.path.basename(video_path)}: {e}")


def main():
    """Main execution function"""
    print("="*60)
    print("UNet Drivable Area Segmentation Pipeline")
    print("Dataset: BDD100K Drivable Area (3 classes)")
    print("="*60)
    
    # Configuration
    MODEL_NAME = 'UNet_baseline'
    N_EPOCHS = 25
    NUM_TEST_SAMPLES = 5
    
    # Flags to control execution
    train_model_flag = True  # Set to True to train model
    evaluate_model_flag = True
    visualize_flag = True
    process_videos_flag = False  # Set to True to process videos
    
    # Step 1: Download and prepare data if needed
    download_and_prepare_data()
    
    # Step 2: Load data
    train_dataloader, val_dataloader, test_dataloader, test_set = load_data()
    
    # Step 3: Create model
    print("\nCreating UNet model...")
    model = create_model(num_classes=NUM_CLASSES)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Output classes: {NUM_CLASSES}")
    
    # Step 4: Train model (if enabled)
    if train_model_flag:
        train_model(model, train_dataloader, val_dataloader, 
                   num_epochs=N_EPOCHS, model_name=MODEL_NAME)
    else:
        print("\nSkipping training (train_model_flag=False)")
    
    # Step 5: Evaluate model (if enabled)
    if evaluate_model_flag:
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        evaluate_trained_model(model, test_dataloader, model_name=MODEL_NAME)
    
    # Step 6: Visualize predictions (if enabled)
    if visualize_flag:
        print("\n" + "="*60)
        print("Visualizing predictions...")
        print("="*60)
        visualize_test_predictions(model, test_set, num_samples=NUM_TEST_SAMPLES)
    
    # Step 7: Process videos (if enabled)
    if process_videos_flag:
        print("\n" + "="*60)
        print("Processing videos...")
        print("="*60)
        process_videos(model, model_name=MODEL_NAME)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
