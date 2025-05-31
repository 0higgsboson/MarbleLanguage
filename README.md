# Marble Language v2

A machine learning project that implements a transformer-based language model for a constrained "Marble Language" - a simple artificial language designed for movement commands and collision detection.

## Overview

The Marble Language is an enhanced artificial language system that simulates marble movement and collisions in a bounded box. The language includes:

### Core Vocabulary (16+ tokens):
- `I` - pronoun
- `move` - movement state
- `east`, `west`, `north`, `south` - directions
- `bump`, `into` - collision indicators
- `then` - connector for chaining actions
- `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, `black`, `white`, `gray`, `brown`, `cyan` - marble colors
- `top`, `bottom`, `left`, `right` - named walls
- `marble` - object identifier

### Enhanced Rules:
1. **Unique Colors**: Each marble has a unique color (no duplicates)
2. **Self-Collision Prevention**: Marbles cannot bump into themselves
3. **Named Walls**: Four walls (top, bottom, left, right) that marbles can hit
4. **Collision Awareness**: Marbles know what they bumped into
5. **Wall Collisions**: Random marble-to-wall collisions

Example sentences:
- "I red marble move east"
- "I blue marble move north bump into green marble"
- "I yellow marble move west bump into top"
- "I purple marble move south bump into left then orange marble move north"

## Project Structure

```
v2/
├── MarbleLanguageDataset.py      # Legacy dataset generation script
├── MarbleSentenceGenerator.py    # Legacy sentence generator
├── marble_transformer_pretraining.py  # Main training script with transformer model
├── training_plotter.py           # Training visualization and analysis
├── marble_language/              # Main package (organized code)
│   ├── __init__.py              # Package exports
│   ├── core/                    # Core language modules
│   │   ├── config.py           # Enhanced rules & configuration
│   │   └── generator.py        # Enhanced sentence generator
│   ├── training/                # Model training modules
│   │   ├── model.py            # Enhanced transformer model
│   │   └── trainer.py          # Training utilities
│   └── utils/                   # Utility modules
│       └── validation.py       # Comprehensive validation system
├── datasets/                     # Generated training data files
│   ├── dataset-1000_*.txt      # Large datasets
│   ├── dataset-200.txt         # Medium datasets
│   └── enhanced_dataset-*.txt  # Enhanced format datasets
└── marble_model/                # Trained model artifacts
    ├── best_model.pt           # Best trained model weights
    └── training_results.json   # Training metrics and results
```

## Features

### Enhanced Sentence Generation
- **Random sentence generation** with enhanced grammatical constraints
- **Unique marble colors** per scene (no color duplicates)
- **Wall collision support** with named walls (top, bottom, left, right)
- **Self-collision prevention** (marbles can't bump into themselves)
- **Collision target awareness** (marbles know what they hit)
- **Configurable probabilities** for sentence patterns
- **Validation system** for all enhanced rules

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

### Enhanced Visualization & Analytics
- **Real-time loss plotting** with iteration-level and epoch-level metrics
- **Interactive training plots** showing loss curves, accuracy, and learning rate
- **Historical training database** tracking all runs and language evolution
- **Language evolution monitoring** capturing vocabulary and rule changes over time
- **Training comparison tools** for analyzing performance across different configurations

## Quick Start

### 1. Generate Training Data

```bash
# Legacy generator
python3 MarbleSentenceGenerator.py 1000

# Enhanced generator with collision rules
python3 -m marble_language.core.generator 1000
```

Both create files in the `datasets/` directory with unique marble language sentences.

### 2. Train the Model

```bash
# Auto-select latest dataset (recommended)
python3 marble_transformer_pretraining.py --epochs 50 --batch_size 32

# Or specify specific dataset files
python3 marble_transformer_pretraining.py datasets/dataset-1000_*.txt --epochs 50

# Or use filenames directly (will look in datasets/)
python3 marble_transformer_pretraining.py dataset-1000_*.txt --epochs 50
```

### 3. Analyze Training Results

```bash
python3 training_plotter.py marble_model/training_results.json
```

## Usage Examples

### Generate Custom Dataset
```bash
# Legacy generator - saves to datasets/
python3 MarbleSentenceGenerator.py 500 my_dataset.txt

# Enhanced generator - saves to datasets/
python3 -m marble_language.core.generator 500 my_enhanced_dataset.txt
```

### Train with Custom Parameters
```bash
# Train with larger model and more epochs
python3 marble_transformer_pretraining.py datasets/data.txt \
    --epochs 100 \
    --batch_size 64 \
    --output_dir my_model \
    --device cuda
```

### Multiple Data Sources
```bash
# Train on multiple data files from datasets directory
python3 marble_transformer_pretraining.py \
    dataset-1000_*.txt dataset-500_*.txt \
    --datasets_dir ./datasets
```

## Model Architecture

- **Vocabulary Size**: 20+ tokens (enhanced marble language + 4 special tokens)
- **Colors Available**: 12 unique marble colors
- **Wall Names**: 4 named walls (top, bottom, left, right)
- **Collision Types**: Marble-to-marble and marble-to-wall
- **Embedding Dimension**: 64
- **Transformer Layers**: 2
- **Attention Heads**: 2
- **Feed-forward Dimension**: 128
- **Max Sequence Length**: 16 tokens
- **Parameters**: ~20,000+ total (expanded vocabulary)

## Training Results

The current best model achieves:
- **Test Loss**: 0.77
- **Test Accuracy**: 60.1%
- **Test Perplexity**: 2.16
- **Training Time**: ~164 sentences dataset

## Dependencies

**Required:**
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.21+

**Optional:**
- matplotlib 3.3+ (for training visualization and plotting)
- pytest 6.0+ (for testing)
- black 21.0+ (for code formatting)
- flake8 3.8+ (for linting)
- scikit-learn 1.0+ (for additional ML utilities)
- tqdm 4.60+ (for progress bars)

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
# Core dependencies for training
pip install torch>=2.0.0 numpy>=1.21.0

# Additional dependencies for full functionality
pip install matplotlib>=3.3.0 tqdm>=4.60.0

# All at once
pip install torch numpy matplotlib tqdm
```

### Dependency Status Check
When running training, the script will show which features are available:
```
✓ matplotlib available - real-time plotting enabled
✓ Enhanced real-time plotter available  
✓ Training database available
✓ Progress bars available (tqdm)
```

If missing dependencies:
```
❌ matplotlib not installed - no real-time plotting available
   Install with: pip3 install matplotlib numpy
   For full functionality: pip3 install torch matplotlib numpy tqdm
```

## File Formats

### Dataset Directory Structure
All generated datasets are organized in the `datasets/` directory:

```
datasets/
├── dataset-1000_20250530_210522.txt    # Large legacy dataset
├── dataset-200.txt                     # Medium legacy dataset  
├── dataset-500_20250530_210213.txt     # Medium legacy dataset
├── enhanced_dataset-100_*.txt          # Enhanced format datasets
└── custom_dataset.txt                  # User-generated datasets
```

### Sentence Files
Generated sentence files contain:
```
Enhanced Marble Language Dataset - Generated on 2025-05-30 21:05:22
================================================================================

Dataset Statistics:
  Total sentences: 100
  Unique colors used: 8
  Wall collisions: 23
  Marble collisions: 31
  Average sentence length: 12.4 tokens

Sentences:
----------------------------------------
1. "I red marble move east bump into blue marble"
2. "I green marble move north bump into top"
3. "I yellow marble move west bump into purple marble move south"
...
```

### Model Files
- `best_model.pt` - PyTorch model checkpoint with weights and metadata
- `training_results.json` - Training metrics and final test results

## Enhanced Grammar Rules

The marble language follows these enhanced constraints:

### Core Rules:
1. **Sentence Structure**: Must start with 'I'
2. **Length**: Between 3-20 tokens per sentence
3. **Marble Identification**: "[color] marble" format required
4. **Movement**: Directions must follow 'move'
5. **Collision Format**: "bump into [target]" where target is marble or wall

### Enhanced Collision Rules:
6. **Unique Colors**: No two marbles can share the same color in a scene
7. **Self-Collision Prevention**: A marble cannot bump into itself
8. **Wall Collisions**: Marbles can bump into named walls (top, bottom, left, right)
9. **Collision Awareness**: Must specify what was bumped into
10. **Valid Targets**: Either "[color] marble" or wall name

### Example Valid Sentences:
- `I red marble move east` (basic movement)
- `I blue marble move north bump into green marble` (marble collision)
- `I yellow marble move west bump into top` (wall collision)
- `I purple marble move south bump into left then orange marble move north` (sequence)

### Example Invalid Sentences:
- `I red marble move east bump into red marble` (self-collision)
- `I red marble move east then red marble move west` (duplicate colors)
- `I marble move east` (missing color)
- `I red marble bump into something` (invalid target)

## Advanced Usage

### Configuration System
Use the organized package structure to modify language rules:
```python
from marble_language.core.config import MARBLE_CONFIG
from marble_language.utils.validation import MarbleLanguageValidator

# Access vocabulary
colors = MARBLE_CONFIG.vocabulary['colors']
walls = MARBLE_CONFIG.vocabulary['walls']

# Modify rules
MARBLE_CONFIG.marble_rules['unique_colors'] = True
MARBLE_CONFIG.wall_rules['wall_collision_probability'] = 0.4

# Validate sentences
validator = MarbleLanguageValidator()
result = validator.validate_sentence("I red marble move east bump into blue marble")
```

### Custom Vocabulary
Add new colors or modify existing ones in the config:
```python
from marble_language.core.config import MARBLE_CONFIG

# Add new color
MARBLE_CONFIG.vocabulary['colors'].append('silver')

# Generate with enhanced generator
from marble_language.core.generator import EnhancedMarbleSentenceGenerator
generator = EnhancedMarbleSentenceGenerator()
sentences = generator.generate_sentences(100)
```

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

## Real-Time Training Visualization

The enhanced training system provides comprehensive real-time visualization:

### Training Plots
The system generates **6 real-time plots** during training:

#### **Iteration-Level Plots** (Updated every 5 iterations):
1. **Training Loss per Iteration** - Shows loss at each training step with moving average
2. **Learning Rate per Iteration** - Displays learning rate schedule (log scale)
3. **Batch Accuracy per Iteration** - Per-batch accuracy during training

#### **Epoch-Level Plots** (Updated after each epoch):
4. **Training & Validation Loss** - Compares train vs validation loss over epochs
5. **Validation Accuracy** - Tracks model accuracy improvement over epochs  
6. **Validation Perplexity** - Shows language model perplexity trends

### Plot Evolution Example

**Before Enhancement:**
- Basic epoch-level plotting only
- No real-time feedback
- Limited metrics visualization

**After Enhancement:**
```
Real-Time Training Progress: MarbleTransformer
Epoch 15 | Train Loss: 1.234 | Val Loss: 1.456 | Val Acc: 0.78

┌─────────────────┬─────────────────┬─────────────────┐
│ Loss/Iteration  │ Learning Rate   │ Batch Accuracy  │
│ (with moving    │ (log scale)     │ (real-time)     │
│ average)        │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────┬─────────────────┬─────────────────┐
│ Train/Val Loss  │ Val Accuracy    │ Val Perplexity  │
│ (epoch level)   │ (trending up)   │ (trending down) │
└─────────────────┴─────────────────┴─────────────────┘
```

## Historical Training Database

All training runs are automatically logged to a comprehensive database:

### Training Run Tracking
```python
# Each run captures:
- Model configuration (vocab size, parameters, architecture)
- Dataset information (files used, sentence count)
- Training hyperparameters (batch size, learning rate, epochs)
- Performance metrics (loss, accuracy, perplexity over time)
- Final results and model paths
```

### Language Evolution Monitoring
The system tracks vocabulary and rule changes over time:

```python
# Evolution tracking captures:
- Vocabulary additions/removals (new colors, walls, etc.)
- Rule modifications (collision rules, wall probabilities)
- Version snapshots of language configuration
- Performance impact of language changes
```

### Query Training History
```bash
# View recent training runs
python3 -c "
from marble_language.utils.training_database import TrainingRunDatabase
db = TrainingRunDatabase()
runs = db.get_training_runs(10)
for run in runs:
    print(f'{run[\"run_id\"]}: Acc={run[\"best_val_accuracy\"]:.3f}')
"

# Get best performing models
python3 -c "
db = TrainingRunDatabase()
best = db.get_best_runs('best_val_accuracy', 5)
print('Top 5 models by accuracy:')
for run in best:
    print(f'  {run[\"timestamp\"]}: {run[\"best_val_accuracy\"]:.4f}')
"
```

### Database Schema
- **training_runs**: Complete run metadata and final results
- **epoch_stats**: Per-epoch metrics (loss, accuracy, perplexity)
- **iteration_stats**: Per-iteration training loss and batch accuracy  
- **language_evolution**: Vocabulary and rule change tracking

## Troubleshooting

### "I don't see plots during training"

**Solution 1: Terminal Plotting (Works without installing anything!)**
```bash
# Demo the terminal plotting system
python3 demo_terminal_plotting.py

# Run actual training - plots will show in terminal
python3 marble_transformer_pretraining.py
```

**What you'll see:**
```
✓ Terminal ASCII plotting enabled
  Loss plots will be displayed in terminal every 25 iterations
  (More frequent for shorter training runs)

================================================================================
TRAINING PROGRESS
================================================================================
Training Statistics:
  Current Loss: 1.234
  Min Loss: 0.856
  Trend: ↓ decreasing

Loss Plot (iterations 0-500)
┌────────────────────────────────────────────────────────────────────────────────┐
│*                                                                               │ 2.000
│  *                                                                             │ 1.800
│    **                                                                          │ 1.600
│       ***                                                                      │ 1.400
│          ****                                                                  │ 1.200
│              ******                                                            │ 1.000
│                    ********                                                    │ 0.800
└────────────────────────────────────────────────────────────────────────────────┘
================================================================================
```

### **Plot Frequency (Adaptive):**
- **First 50 iterations**: Plots every 10 iterations
- **Iterations 50-100**: Plots every 15 iterations  
- **Longer training**: Plots every 25 iterations
- **End of each epoch**: Always shows final progress for short training

**Solution 2: Full GUI Plotting (Requires matplotlib)**
```bash
# Install plotting dependencies
python3 install_dependencies.py

# Or manually:
pip3 install matplotlib numpy
```

**Common Issues:**
- **macOS Homebrew**: Use `--break-system-packages` or create virtual environment
- **Linux**: May need `python3-tk` package: `sudo apt-get install python3-tk`  
- **Windows**: Ensure using Python 3.8+ and pip is up to date

### "No datasets found"

**Generate a dataset first:**
```bash
# Create enhanced dataset with wall collisions
python3 -m marble_language.core.generator 1000

# Or use legacy generator
python3 MarbleSentenceGenerator.py 1000
```

### "Module not found errors"

**Install core dependencies:**
```bash
python3 install_dependencies.py
```

**Or check what's missing:**
```bash
python3 -c "
import sys
missing = []
for pkg in ['torch', 'numpy', 'matplotlib']:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)
if missing:
    print(f'Install: pip3 install {\" \".join(missing)}')
"
```

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.