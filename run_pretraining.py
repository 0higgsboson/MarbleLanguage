#!/usr/bin/env python3
"""
Easy wrapper for running marble language pretraining
"""

import sys
from marble_pretraining_simple import SimpleMarbleTrainer

def main():
    print("🎯 Marble Language Pretraining (Simplified)")
    print("=" * 50)
    
    # Parse command line arguments or use defaults
    dataset_size = 500  # Default size
    epochs = 5  # Default epochs
    
    if len(sys.argv) > 1:
        try:
            dataset_size = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid dataset size. Using default: 500")
    
    if len(sys.argv) > 2:
        try:
            epochs = int(sys.argv[2])
        except ValueError:
            print("❌ Invalid epochs. Using default: 5")
    
    print(f"📝 Dataset size: {dataset_size} sentences")
    print(f"🔄 Training epochs: {epochs}")
    print()
    
    # Run training
    trainer = SimpleMarbleTrainer(dataset_size=dataset_size)
    trainer.run_training(epochs=epochs)

if __name__ == "__main__":
    main()