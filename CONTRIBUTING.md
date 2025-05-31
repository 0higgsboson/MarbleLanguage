# Contributing to MarbleLLM

Thank you for your interest in contributing to MarbleLLM! This project explores transformer-based language modeling on constrained artificial languages.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/MarbleLLM.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Development Setup

### Running Tests
```bash
# Generate sample data
python MarbleSentenceGenerator.py 100

# Test training pipeline
python marble_transformer_pretraining.py marble_sentences_*.txt --epochs 5
```

### Code Style
- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Areas for Contribution

### 🚀 Features
- **Model improvements**: Different architectures, attention mechanisms
- **Training enhancements**: Better data augmentation, curriculum learning
- **Evaluation metrics**: BLEU scores, semantic similarity measures
- **Visualization**: Interactive training dashboards, model interpretability
- **Grammar expansion**: More complex marble language rules

### 🐛 Bug Fixes
- Training stability issues
- Memory optimization
- Cross-platform compatibility
- Error handling improvements

### 📚 Documentation
- API documentation
- Tutorials and examples
- Performance benchmarks
- Architecture explanations

### 🧪 Research
- Comparative studies with other small language models
- Analysis of emergent behaviors
- Scaling laws for constrained languages
- Transfer learning experiments

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear commit messages
3. Test your changes thoroughly
4. Update documentation if needed
5. Submit a pull request with:
   - Clear description of changes
   - Test results or examples
   - Any breaking changes noted

## Code Organization

```
MarbleLLM/
├── MarbleSentenceGenerator.py    # Data generation
├── marble_transformer_pretraining.py  # Main training logic
├── training_plotter.py          # Visualization utilities
├── tests/                       # Unit tests (to be added)
├── examples/                    # Usage examples (to be added)
└── docs/                       # Documentation (to be added)
```

## Research Guidelines

This project is educational and research-focused. When contributing:

- **Cite relevant papers** in comments or documentation
- **Include experimental results** when proposing changes
- **Document hyperparameter choices** and their rationale
- **Consider computational efficiency** - keep experiments runnable on modest hardware

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for research ideas or architectural questions
- Check existing issues before creating new ones

## Recognition

Contributors will be acknowledged in the repository and any future publications or presentations based on this work.

Happy coding! 🎯