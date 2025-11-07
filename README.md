# UNet Drivable Area Segmentation

This project implements drivable area segmentation using the UNet architecture on the BDD100K dataset. The model identifies three key areas in driving scenes: **ego lane** (direct drivable area), **adjacent lanes** (alternative drivable areas), and **background** (non-drivable areas).

![UNet Architecture](unet_architecture.png)

*Image reference: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)*

## Dataset

**BDD100K Drivable Area Segmentation (3 Classes)**

The dataset is automatically downloaded from Google Drive when you run the training script for the first time.

- **Dataset Source**: Pre-processed BDD100K drivable area data (180x320 resolution)
- **Google Drive ID**: `1sX6kHxpYoEICMTfjxxhK9lTW3B7OUxql`
- **Size**: ~100MB (compressed)

### Class Definitions:

| Class ID | Category | Color (RGB) | Description |
|----------|----------|-------------|-------------|
| 0 | direct | (171, 44, 236) | Current/ego lane - the lane the vehicle is driving in |
| 1 | alternative | (86, 211, 19) | Adjacent/alternative lanes - other drivable lanes |
| 2 | background | (0, 0, 0) | Non-drivable areas - sidewalks, buildings, etc. |

This 3-class approach is essential for:
- Lane keeping assistance systems
- Autonomous navigation and path planning
- Drivable area detection for ADAS
- Real-time decision making in autonomous vehicles

## Quick Start

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional, but recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mark-Moawad/UNet-Drivable-Area-Segmentation.git
   cd UNet-Drivable-Area-Segmentation
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

That's it! The dataset will be automatically downloaded when you run the training script for the first time.

## Usage

### Training

The training script is designed to work out of the box. Simply run:

```bash
python unet_segmentation.py
```

The script will:
1. Automatically download the dataset from Google Drive (first run only)
2. Extract and organize the data
3. Load pre-processed images and labels (180x320 resolution)
4. Train the UNet model
5. Save the trained model and training curves

**Training Configuration:**
- **Epochs**: 25 (default)
- **Batch Size**: 8
- **Learning Rate**: 3e-4 (OneCycleLR scheduler)
- **Loss Function**: Dice Loss
- **Dataset Split**: 70% train, 20% validation, 10% test
- **Input Resolution**: 180x320x3
- **Output Classes**: 3

The trained model will be saved as `UNet_baseline.pt`.

### Evaluation & Visualization

After training completes, the script automatically:
- Evaluates the model on the test set
- Computes mean IoU metric
- Generates prediction visualizations comparing:
  - Original RGB images
  - Ground truth labels
  - Model predictions

### Video Inference

To test the model on driving videos:

1. **Download sample videos** (KITTI dataset examples):
   ```bash
   # Highway scene
   gdown "1fs2Sc0OK1g9Epnl9PhoQyQEfBF3YYC7A"
   
   # Residential scene
   gdown "1uosW46RhVD7ysFni1qY_XQI4mSlPzSHz"
   
   # Campus scene
   gdown "1n5QxX6LIImRvrhrcnm9kA8HKAEb8MN0x"
   ```

2. **Run inference** by modifying `unet_segmentation.py`:
   ```python
   from utils import predict_video
   
   predict_video(model, "UNet_drivable", "highway_1241_376.avi", 
                 "segmentation", 1241, 376, "cuda", train_id_to_color)
   ```

## Model Architecture

The UNet model consists of:
- **Encoder**: 4 downsampling blocks with max pooling
  - Layer channels: [64, 128, 256, 512]
  - Each block: 2 convolutional layers with BatchNorm and ReLU
- **Bottleneck**: Double convolution at the lowest resolution (1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
  - Transposed convolutions for upsampling
  - Skip connections from encoder for feature fusion
- **Output**: 1x1 convolution to produce 3-class predictions

**Total Parameters**: ~31M

## Key Features

### Automatic Dataset Download
- No manual dataset preparation needed
- Automatically downloads from Google Drive using `gdown`
- Extracts and organizes data structure
- Uses pre-processed 180x320 resolution images for fast training

### Clean Python Script
- Professional, well-documented code structure
- No Jupyter notebook dependencies
- Direct execution from command line
- Comprehensive error handling and logging

### Efficient Training
- **Pre-processed Data**: Images resized to 180x320 for fast training
- **ImageNet Normalization**: Better convergence and performance
- **OneCycleLR Scheduler**: Optimal learning rate scheduling
- **Dice Loss**: Better handling of class imbalance
- **Model Checkpointing**: Saves best model based on validation loss

### Quality Visualization
- Color-coded predictions for easy interpretation:
  - Magenta: Ego lane (where you're driving)
  - Green: Adjacent lanes (where you can merge)
  - Black: Non-drivable areas
- Training curves automatically plotted and saved
- Side-by-side comparison of ground truth vs predictions

### Video Processing
- Quality-preserving inference pipeline
- Processes videos at original resolution
- Fast inference on downsampled frames
- Upscales predictions to match original video quality

## Results

With 25 epochs of training, the model achieves good performance on the BDD100K drivable area segmentation task:
- Clear distinction between ego lane and adjacent lanes
- Accurate identification of non-drivable areas
- Real-time capable inference (~30 FPS on GPU)

Sample predictions show:
- **Magenta areas**: Current lane (safe to drive)
- **Green areas**: Adjacent lanes (safe for lane changes)
- **Black areas**: Non-drivable regions (avoid)

## Project Structure

```
UNet-Drivable-Area-Segmentation/
├── unet_segmentation.py       # Main script with training & inference
├── utils.py                    # Utility functions (metrics, visualization, video processing)
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── venv/                      # Virtual environment (created after setup)
└── dataset/                   # Auto-downloaded dataset (created on first run)
    ├── image_180_320.npy      # Pre-processed images (7,000 samples)
    └── label_180_320.npy      # Pre-processed labels (7,000 samples)
```

## Dependencies

Key libraries:
- `torch>=1.10.0`: PyTorch deep learning framework
- `torchvision>=0.11.0`: Computer vision utilities
- `opencv-python>=4.5.0`: Video processing and image manipulation
- `segmentation-models-pytorch>=0.3.0`: Dice loss implementation
- `numpy>=1.19.0`: Numerical operations
- `matplotlib>=3.3.0`: Visualization
- `tqdm>=4.62.0`: Progress bars
- `pandas>=1.3.0`: Data handling
- `gdown>=4.7.1`: Google Drive file downloader

See `requirements.txt` for complete list.

## Training Tips

- **GPU Recommended**: Training on CPU will be significantly slower
- **Adjust Epochs**: You can modify `N_EPOCHS` in the script for longer/shorter training
- **Batch Size**: Reduce if you encounter GPU memory issues
- **Learning Rate**: The OneCycleLR scheduler automatically handles learning rate

## Why This Project?

This project demonstrates:
- ✅ **Deep Learning Fundamentals**: U-Net architecture, encoder-decoder design
- ✅ **Computer Vision**: Image segmentation, semantic understanding
- ✅ **PyTorch Expertise**: Custom models, datasets, training loops
- ✅ **Autonomous Driving**: Lane detection, drivable area segmentation
- ✅ **Production Skills**: Quality-preserving inference, video processing
- ✅ **Clean Code**: Well-documented, modular, easy to understand

Perfect for showcasing computer vision and autonomous driving skills to recruiters!

## References

- **Original UNet Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **BDD100K Dataset**: [Berkeley DeepDrive](https://bdd-data.berkeley.edu/)
- **BDD100K Paper**: [BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling](https://arxiv.org/abs/1805.04687)

## License

This project is licensed under the MIT License.

## Author

**Mark Moawad**  
Perception Engineer | Self-Driving Car Specialist

This project showcases practical computer vision skills for autonomous driving applications, demonstrating the ability to implement, train, and deploy deep learning models for real-world driving scenarios.

## Acknowledgments

- Original U-Net paper by Ronneberger et al.
- BDD100K dataset team at Berkeley
- PyTorch and segmentation_models_pytorch communities

## Contact

For questions or collaboration opportunities, please reach out through GitHub.


