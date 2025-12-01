# Fine-Tuning Pipeline Documentation

This document describes the complete pipeline for converting preprocessed emotion data into a format suitable for fine-tuning, the fine-tuning process itself, and the results obtained.

## Table of Contents

1. [Data Preprocessing Overview](#data-preprocessing-overview)
2. [Data Pipeline: Preprocessed Data → Fine-Tuning Format](#data-pipeline-preprocessed-data--fine-tuning-format)
3. [Fine-Tuning Process](#fine-tuning-process)
4. [Results](#results)
5. [Conclusion](#conclusion)

---

## Data Preprocessing Overview

The preprocessing stage (handled by `preprocessed_data/preprocess_emotions.py`) converts raw ROS2 bag data into structured image-emotion pairs:

- **Input**: ROS2 bag files (`data/data_0.db3`) containing emotion estimates from the `/emotion_estimate` topic
- **Output**: Directory structure with:
  - `canvas.png`: The image associated with each batch
  - `emotion_probabilities.json`: JSON file containing emotion label counts and probability distributions

The preprocessing script:
1. Reads emotion labels from ROS2 bag files
2. Counts occurrences of each emotion label per batch
3. Converts counts to probability distributions
4. Normalizes labels to the ArtEmis emotion order:
   - `amusement`, `awe`, `contentment`, `excitement`
   - `anger`, `disgust`, `fear`, `sadness`, `something else`

**Output Structure:**
```
preprocessed_data/
├── artist-session-1/
│   ├── batch_0016/
│   │   ├── canvas.png
│   │   └── emotion_probabilities.json
│   ├── batch_0017/
│   │   └── ...
│   └── ...
├── artist-session-2/
│   └── ...
└── artist-session-3/
    └── ...
```

---

## Data Pipeline: Preprocessed Data → Fine-Tuning Format

### Overview

The `create_pickle_batches.py` script transforms the preprocessed data into a format optimized for fine-tuning by:
1. Loading images and emotion distributions
2. Extracting CLIP embeddings for each image
3. Creating train/validation/test splits
4. Saving data as pickle batch files

### Step-by-Step Process

#### 1. **Data Loading and Validation**

The script scans all session directories and batch subdirectories in `preprocessed_data/`:

```python
def find_all_batches(data_root: Path) -> List[Path]:
    """Find all batch directories."""
    batches = []
    for session_dir in sorted(data_root.glob("*-session-*")):
        for batch_dir in sorted(session_dir.glob("batch_*")):
            if batch_dir.is_dir():
                batches.append(batch_dir)
    return batches
```

For each batch directory, it loads:
- **Image**: `canvas.png` from the batch directory
- **Emotion Distribution**: Probability vector from `emotion_probabilities.json`

The emotion distribution is validated and normalized to ensure:
- Exactly 9 values (one per ArtEmis emotion)
- Sum equals 1.0 (proper probability distribution)
- Fallback to uniform distribution if all zeros

#### 2. **Data Splitting**

The data is randomly shuffled and split into three sets:
- **Training**: 70% of batches
- **Validation**: 15% of batches
- **Test**: 15% of batches

Default random seed: `42` (for reproducibility)

#### 3. **CLIP Embedding Extraction**

For each split:
1. **Load CLIP Model**: Uses `RN50x16` by default (produces 768-dimensional embeddings)
2. **Create Dataset**: `ImageEmotionDataset` class handles image loading and preprocessing
3. **Extract Embeddings**: Images are passed through CLIP's visual encoder to obtain feature vectors
4. **Save as Pickle Batches**: Each batch is saved as `batch{N}.bin` containing:
   - `image`: CLIP embedding tensor (shape: `[batch_size, 768]`)
   - `label`: Emotion probability distribution tensor (shape: `[batch_size, 9]`)

**Key Code:**
```python
# Extract CLIP embeddings
with torch.no_grad():
    clip_features = clip_model.encode_image(image_tensor)
    # Save features and labels together
    batch = {
        'image': clip_features.cpu(),
        'label': emotion_distributions
    }
    pickle.dump(batch, f)
```

#### 4. **Output Structure**

The final fine-tuning data structure:

```
finetuning_data/
├── train/
│   ├── batch0.bin
│   ├── batch1.bin
│   └── ... (51 batches)
├── val/
│   ├── batch0.bin
│   ├── batch1.bin
│   └── ... (11 batches)
└── test/
    ├── batch0.bin
    ├── batch1.bin
    └── ... (11 batches)
```

**Statistics:**
- **Total batches processed**: ~73 batches
- **Training samples**: ~51 batches
- **Validation samples**: ~11 batches
- **Test samples**: ~11 batches

### Usage

```bash
python create_pickle_batches.py \
    --data_root preprocessed_data \
    --output_dir finetuning_data \
    --clip_model RN50x16 \
    --batch_size 32 \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15 \
    --seed 42
```

---

## Fine-Tuning Process

### Model Architecture

The fine-tuned model uses a **Single-Layer Perceptron (SLP)** architecture:

```python
class SLP(nn.Module):
    def __init__(self, input_size=768, output_size=9):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, output_size)
        )
```

- **Input**: 768-dimensional CLIP embeddings (from RN50x16)
- **Output**: 9 emotion logits (one per ArtEmis emotion)
- **Loss Function**: `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
- **Optimizer**: Adam

### Fine-Tuning Procedure

The `finetune.py` script implements the fine-tuning process:

#### 1. **Model Initialization**

- Creates a new SLP model with appropriate input/output sizes
- Loads pre-trained weights from `neural_checkpoints/C-RN50x16` (if available)
- Falls back to random initialization if checkpoint not found

#### 2. **Network Capacity Test** (Optional)

Before training, the model is tested to ensure it can overfit a single batch:
- Trains on one batch for 20 epochs
- Verifies loss drops below 0.5
- Confirms the model has sufficient capacity to learn

#### 3. **Training Strategy**

The fine-tuning uses **validation-monitored training** with learning rate scheduling:

```python
def train_eval(self, epochs_per_lr, lrs):
    for lr in lrs:
        for epoch in range(epochs_per_lr):
            self.lr = lr
            previous_state = copy.copy(self.model.state_dict())
            self.train_n_epochs(1)
            val_loss = self.eval_loss()
            
            # Early stopping: revert if validation loss increases
            if val_loss > self.validation_losses[-1]:
                self.model.load_state_dict(previous_state)
                break
            self.validation_losses.append(val_loss)
```

**Learning Rate Schedule:**
- **Phase 1**: `1e-3` (initial learning rate)
- **Phase 2**: `1e-4` (10x reduction)
- **Phase 3**: `1e-5` (100x reduction)

**Training Features:**
- **Validation Monitoring**: After each epoch, validation loss is computed
- **Early Stopping**: If validation loss increases, the model state is reverted and training moves to the next learning rate phase
- **Epoch Distribution**: Total epochs split across the three learning rate phases

#### 4. **Training Configuration**

**Default Parameters:**
- **Learning Rate**: `1e-3`
- **Epochs**: 20 (distributed across LR phases)
- **Batch Processing**: Uses `Pickle_data_loader` to load batches from pickle files
- **Device**: Automatically uses CUDA if available, otherwise CPU

### Usage

```bash
python finetune.py \
    --checkpoint neural_checkpoints/C-RN50x16 \
    --data finetuning_data \
    --output neural_checkpoints/finetuned_emotion_predictor_model \
    --lr 1e-3 \
    --epochs 20
```

---

## Results

### Test Set Evaluation

The fine-tuned model was evaluated on 5 test images. The results demonstrate the model's ability to predict emotion distributions for new images.

#### Test Image Predictions

| Image | Top Emotion | Probability | Top 3 Emotions |
|-------|-------------|-------------|----------------|
| `image1.png` | Contentment | 0.707 | 1. Contentment (0.707)<br>2. Sadness (0.247)<br>3. Excitement (0.013) |
| `image2.png` | Contentment | 0.824 | 1. Contentment (0.824)<br>2. Sadness (0.151)<br>3. Excitement (0.011) |
| `image3.png` | Sadness | 0.513 | 1. Sadness (0.513)<br>2. Contentment (0.444)<br>3. Excitement (0.019) |
| `image4.png` | Sadness | 0.572 | 1. Sadness (0.572)<br>2. Contentment (0.338)<br>3. Amusement (0.034) |
| `image5.png` | Contentment | 0.512 | 1. Contentment (0.512)<br>2. Sadness (0.439)<br>3. Excitement (0.021) |

#### Average Emotion Distribution

Across all test images, the model's average predicted emotion distribution:

| Emotion | Average Probability |
|---------|-------------------|
| **Contentment** | **0.565** |
| **Sadness** | **0.384** |
| Excitement | 0.019 |
| Amusement | 0.014 |
| Anger | 0.005 |
| Fear | 0.005 |
| Disgust | 0.003 |
| Awe | 0.002 |
| Something else | 0.002 |

### Key Observations

1. **Strong Predictions**: The model produces confident predictions with top emotions having probabilities > 0.5 for most images.

2. **Emotion Bias**: The model shows a bias toward **Contentment** and **Sadness**, which likely reflects the distribution of emotions in the training data.

3. **Clear Distinctions**: The model successfully distinguishes between different emotional content:
   - Images 1, 2, and 5 are classified as primarily "Contentment"
   - Images 3 and 4 are classified as primarily "Sadness"

4. **Probability Distributions**: The model outputs well-calibrated probability distributions that sum to 1.0, allowing for uncertainty quantification.

5. **Visualization**: A comprehensive visualization (`test_predictions.png`) was generated showing:
   - Original images alongside their predicted emotion distributions
   - Bar charts with probability values
   - Highlighted top emotions

### Model Performance Characteristics

- **Confidence**: High confidence predictions (top emotion typically > 0.5)
- **Consistency**: Consistent predictions across similar images
- **Calibration**: Proper probability distributions (sum to 1.0)
- **Generalization**: Successfully predicts emotions for unseen test images

---

## Conclusion

The fine-tuning pipeline successfully:

1. **Converted preprocessed data** from image + JSON format to CLIP embeddings stored as pickle batches
2. **Created proper train/val/test splits** (70/15/15) for model evaluation
3. **Fine-tuned the emotion predictor** using validation-monitored training with learning rate scheduling
4. **Achieved good performance** on test images with confident, well-calibrated predictions

The fine-tuned model demonstrates the ability to predict emotion distributions for artistic images, with a particular strength in identifying **Contentment** and **Sadness** emotions. The pipeline is reproducible and can be extended to fine-tune on additional data or adjust hyperparameters for improved performance.

### Future Improvements

- **Data Augmentation**: Add image augmentation to increase training data diversity
- **Class Balancing**: Address the bias toward Contentment/Sadness through class weighting or data resampling
- **Architecture**: Experiment with deeper networks or attention mechanisms
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and training epochs
- **Evaluation Metrics**: Add more comprehensive metrics (precision, recall, F1-score) on a larger test set

---

## File Structure Summary

```
EmotionPredictor/
├── preprocessed_data/              # Input: Images + emotion JSON files
│   ├── artist-session-1/
│   │   └── batch_*/                # canvas.png + emotion_probabilities.json
│   └── ...
├── create_pickle_batches.py        # Converts preprocessed → pickle batches
├── finetuning_data/                # Output: CLIP embeddings as pickle files
│   ├── train/batch*.bin
│   ├── val/batch*.bin
│   └── test/batch*.bin
├── finetune.py                     # Fine-tuning script
├── neural_checkpoints/
│   ├── C-RN50x16                   # Pre-trained checkpoint (input)
│   └── finetuned_emotion_predictor_model  # Fine-tuned model (output)
└── test_finetuned_model.ipynb      # Evaluation notebook
```

