

# Fine-Grained Image Classification using Semi-Supervised Learning

This project implements a semi-supervised learning approach for fine-grained image classification using Vision Transformers (specifically Swin Transformer) and CNNs. The implementation achieves state-of-the-art performance of 94.82% accuracy using Swin Transformer Large.

## Authors
- Sriraja Vignesh Senthil Murugan
- Devanathan Nallur Gandamani
- Mohnish Sai Prasad

## Dataset Structure
```
dataset/
├── train/
│   ├── labeled/          # 9,854 labeled training images
│   │   ├── 00000.jpg
│   │   └── ...
│   └── unlabeled/        # 22,995 unlabeled training images
│       ├── 09854.jpg
│       └── ...
├── test/                 # 8,213 test images
│   ├── 32849.jpg
│   └── ...
├── categories.csv        # Category metadata
├── train_labeled.csv     # Training labels
└── sample_submission.csv # Submission format
```

## Requirements
```
torch
torchvision
timm
tqdm
pandas
Pillow
scikit-learn
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture
We implement multiple architectures:
- Swin Transformer Large (94.82% accuracy)
- Swin Transformer Base (92.99% accuracy)
- InceptionV3 (~89% accuracy)
- EfficientNetV2 (~89% accuracy)

## Implementation Details

### 1. Base Model Training
```python
# Initialize model
model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)

# Update classification head
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)

# Configure optimizer
optimizer = torch.optim.AdamW([
    {'params': model.head.fc.parameters(), 'lr': 1e-3},
    {'params': model.layers[3].blocks[0].parameters(), 'lr': 5e-5},
    {'params': model.layers[3].blocks[1].parameters(), 'lr': 1e-4},
], weight_decay=1e-4)
```

### 2. Semi-Supervised Learning
The implementation uses a pseudo-labeling approach with a confidence threshold of 0.95:

1. Train initial model on labeled data
2. Generate predictions for unlabeled data
3. Select high-confidence predictions (>95%)
4. Combine labeled and pseudo-labeled data
5. Retrain model on combined dataset

## Running the Code

### 1. Data Preparation
```python
# Extract dataset
!unzip /content/ucsc-cse-244-a_data-set.zip -d /content/data-set/

# Setup data paths
labeled_image_folder = "/content/data-set/train/labeled"
labeled_image_true_values = "/content/data-set/train_labeled.csv"
unlabeled_image_folder = '/content/data-set/train/unlabeled'
```

### 2. Training Base Model
```python
# Run training loop
python train_base.py --batch_size 32 --epochs 15 --model swin_large
```

### 3. Generating Pseudo-Labels
```python
# Generate pseudo-labels for unlabeled data
python generate_pseudo_labels.py --confidence_threshold 0.95
```

### 4. Training Combined Model
```python
# Train on combined dataset
python train_combined.py --batch_size 32 --epochs 5
```

## Results
We achieved the following accuracies:
- Swin Large (Base Model): 94.82%
- Swin Base (Combined): 92.99%
- InceptionV3: ~89%
- EfficientNetV2: ~89%

## Model Weights
The trained model weights can be found at:
```
./saved-models/swin_large_32.pth
```

## Reproducibility
To ensure reproducible results:
1. Use fixed random seeds
2. Follow the provided training configuration
3. Use the same data splits
4. Maintain consistent evaluation protocols

## Citation
If you use this code in your research, please cite:
```bibtex
@article{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
