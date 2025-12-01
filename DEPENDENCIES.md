# EmotionPredictor Dependencies

## Poetry Package Setup

This repository has been configured with Poetry for dependency management.

### Installation

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install CLIP (from git, as it's not on PyPI)
poetry run pip install git+https://github.com/openai/CLIP.git
```

### Core Dependencies

The following dependencies are managed by Poetry (see `pyproject.toml`):

- **torch** (^2.0.0) - PyTorch deep learning framework
- **torchvision** (^0.15.0) - Computer vision utilities for PyTorch
- **scikit-learn** (^1.3.0) - Machine learning utilities
- **pandas** (^2.0.0) - Data manipulation and analysis
- **matplotlib** (^3.7.0) - Plotting library
- **plotly** (^5.17.0) - Interactive plotting
- **seaborn** (^0.12.0) - Statistical data visualization
- **Pillow** (^10.0.0) - Image processing
- **jupyter** (^1.0.0) - Jupyter notebook support
- **tqdm** (^4.66.0) - Progress bars
- **termcolor** (^2.3.0) - Terminal colors
- **dash** (^2.14.0) - Web dashboard framework
- **requests** (^2.31.0) - HTTP library
- **numpy** (^1.24.0) - Numerical computing
- **ipython** (^8.0.0) - Enhanced Python shell

### External Dependencies (Not on PyPI)

- **CLIP** - Must be installed from GitHub:
  ```bash
  poetry run pip install git+https://github.com/openai/CLIP.git
  ```

- **artemis** (optional) - Only needed if using original ArtEmis dataset loading:
  ```bash
  # Install from: https://github.com/optas/artemis
  ```

### Python Version

Requires Python >= 3.9

### Usage

After installation, activate the Poetry environment:

```bash
poetry shell
```

Or run commands with:

```bash
poetry run python emotionpredictor/demo_images.py
```

### Running Demos

1. **Image Demo**:
   ```bash
   poetry run python emotionpredictor/demo_images.py
   ```
   Then open http://localhost:8050 in your browser

2. **Text Demo**:
   ```bash
   poetry run python emotionpredictor/demo_text.py
   ```
   Then open http://localhost:8048 in your browser

### Model Checkpoints

The pre-trained model checkpoints should be in `neural_checkpoints/` directory:
- `C-RN50x16` - CLIP ResNet50x16 model (recommended)
- Other variants available: `C-RN50`, `C-RN101`, `C-ViT-B16`, `C-ViT-B32`, etc.

