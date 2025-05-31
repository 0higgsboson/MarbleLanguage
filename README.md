# Marble Language v2

A machine learning project that implements a transformer-based language model for a constrained "Marble Language" - a simple artificial language designed for movement commands and collision detection.

## Overview

The Marble Language consists of 9 core tokens that describe movement and state:
- `I` - pronoun
- `static`, `move` - states  
- `East`, `West`, `North`, `South` - directions
- `bump` - collision indicator
- `then` - connector for chaining actions

Example sentences:
- "I move East"
- "Move North bump then static"
- "I static then move South"

## Project Structure

```
v2/
├── MarbleLanguageDataset.py      # Dataset generation and management
├── MarbleSentenceGenerator.py    # Random sentence generator
├── marble_transformer_pretraining.py  # Main training script with transformer model
├── training_plotter.py           # Training visualization and analysis
├── marble_model/                 # Trained model artifacts
│   ├── best_model.pt            # Best trained model weights
│   └── training_results.json    # Training metrics and results
└── marble_sentences_*.txt        # Generated training data files
```

## Features

### Sentence Generation
- **Random sentence generation** with grammatical constraints
- **Vocabulary compliance** validation
- **Configurable probabilities** for sentence patterns
- **Unique sentence generation** to avoid duplicates

### Transformer Model
- **Custom transformer architecture** optimized for marble language
- **Causal language modeling** with autoregressive generation
- **Attention mechanisms** with position embeddings
- **Early stopping** and learning rate scheduling

### Training Pipeline
- **Automatic data parsing** from text files
- **Train/validation/test splits** with shuffling
- **Real-time loss monitoring** and sample generation
- **Model checkpointing** with best model saving
- **Comprehensive logging** with iteration-level tracking

### Visualization
- **Real-time training plots** (if matplotlib available)
- **Loss curves**, accuracy progression, and perplexity tracking
- **Learning rate scheduling** visualization
- **Training summary** statistics and analysis

## Quick Start

### 1. Generate Training Data

```bash
python3 MarbleSentenceGenerator.py 1000
```

This creates a file with 1000 unique marble language sentences.

### 2. Train the Model

```bash
python3 marble_transformer_pretraining.py marble_sentences_*.txt --epochs 50 --batch_size 32
```

### 3. Analyze Training Results

```bash
python3 training_plotter.py marble_model/training_results.json
```

## Usage Examples

### Generate Custom Dataset
```bash
# Generate 500 sentences with custom filename
python3 MarbleSentenceGenerator.py 500 my_dataset.txt
```

### Train with Custom Parameters
```bash
# Train with larger model and more epochs
python3 marble_transformer_pretraining.py data.txt \
    --epochs 100 \
    --batch_size 64 \
    --output_dir my_model \
    --device cuda
```

### Multiple Data Sources
```bash
# Train on multiple data files
python3 marble_transformer_pretraining.py file1.txt file2.txt file3.txt
```

## Model Architecture

- **Vocabulary Size**: 13 tokens (9 marble + 4 special tokens)
- **Embedding Dimension**: 64
- **Transformer Layers**: 2
- **Attention Heads**: 2
- **Feed-forward Dimension**: 128
- **Max Sequence Length**: 16 tokens
- **Parameters**: ~17,000 total

## Training Results

The current best model achieves:
- **Test Loss**: 0.77
- **Test Accuracy**: 60.1%
- **Test Perplexity**: 2.16
- **Training Time**: ~164 sentences dataset

## Dependencies

**Required:**
- Python 3.7+
- PyTorch
- NumPy

**Optional:**
- matplotlib (for training visualization)

Install dependencies:
```bash
pip3 install torch numpy matplotlib
```

## File Formats

### Sentence Files
Generated sentence files contain:
```
1. "I move East"
2. "Move North bump"
3. "I static then move South"
...
```

### Model Files
- `best_model.pt` - PyTorch model checkpoint with weights and metadata
- `training_results.json` - Training metrics and final test results

## Grammar Rules

The marble language follows these constraints:
1. Sentences must start with 'I' or 'Move'
2. Length between 2-8 tokens
3. Directions must follow 'move' or 'Move'
4. 'then' connects action sequences
5. 'bump' indicates collision events
6. All tokens must be from the defined vocabulary

## Advanced Usage

### Custom Vocabulary
Modify `create_vocabulary()` in the training script to add new tokens.

### Model Architecture
Adjust transformer parameters in `MarbleTransformer` class:
- `embed_dim` - embedding size
- `num_heads` - attention heads
- `num_layers` - transformer layers
- `ff_dim` - feed-forward dimension

### Training Configuration
Configure training in the argument parser:
- Batch size, learning rate, epochs
- Early stopping patience
- Device selection (CPU/CUDA)

## Contributing

This is a research/educational project exploring:
- Constrained language modeling
- Transformer architectures on small vocabularies  
- Synthetic dataset generation
- Training pipeline development

## License

Open source - feel free to use and modify for research and educational purposes.