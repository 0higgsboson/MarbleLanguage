#!/usr/bin/env python3
"""
Training Results Plotter and Analyzer
Analyzes and plots training results from saved JSON files
"""

import json
import argparse
import os
from typing import Dict, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Error: matplotlib not available. Install with: pip3 install matplotlib")
    exit(1)


def load_training_results(results_file: str) -> Dict:
    """Load training results from JSON file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def plot_training_history(training_history: List[Dict], model_name: str, output_dir: str = None):
    """Create comprehensive training plots"""
    
    if not training_history:
        print("No training history found in results file")
        return
    
    # Extract data
    epochs = [epoch['epoch'] for epoch in training_history]
    train_losses = [epoch['train_loss'] for epoch in training_history]
    val_losses = [epoch['val_loss'] for epoch in training_history]
    val_accuracies = [epoch['val_accuracy'] for epoch in training_history]
    val_perplexities = [epoch['val_perplexity'] for epoch in training_history]
    learning_rates = [epoch['learning_rate'] for epoch in training_history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Training Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]
    axes[0, 0].plot(best_epoch, best_val_loss, 'go', markersize=10, label=f'Best (Epoch {best_epoch})')
    axes[0, 0].legend()
    
    # Plot 2: Accuracy progression
    axes[0, 1].plot(epochs, val_accuracies, 'g-', linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mark best accuracy
    best_acc_idx = np.argmax(val_accuracies)
    best_acc_epoch = epochs[best_acc_idx]
    best_accuracy = val_accuracies[best_acc_idx]
    axes[0, 1].plot(best_acc_epoch, best_accuracy, 'ro', markersize=8)
    axes[0, 1].text(best_acc_epoch, best_accuracy + 0.02, f'Best: {best_accuracy:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Perplexity
    axes[0, 2].plot(epochs, val_perplexities, 'purple', linewidth=2, marker='o', markersize=3)
    axes[0, 2].set_title('Validation Perplexity')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Perplexity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Mark best perplexity
    best_perp_idx = np.argmin(val_perplexities)
    best_perp_epoch = epochs[best_perp_idx]
    best_perplexity = val_perplexities[best_perp_idx]
    axes[0, 2].plot(best_perp_epoch, best_perplexity, 'ro', markersize=8)
    
    # Plot 4: Learning Rate Schedule
    axes[1, 0].plot(epochs, learning_rates, 'orange', linewidth=2, marker='o', markersize=3)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # Log scale for learning rate
    
    # Plot 5: Loss improvement over time
    loss_improvements = []
    best_so_far = float('inf')
    for val_loss in val_losses:
        if val_loss < best_so_far:
            improvement = best_so_far - val_loss
            best_so_far = val_loss
        else:
            improvement = 0
        loss_improvements.append(improvement)
    
    axes[1, 1].bar(epochs, loss_improvements, alpha=0.7, color='skyblue')
    axes[1, 1].set_title('Validation Loss Improvements')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Improvement')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training statistics summary
    axes[1, 2].text(0.05, 0.95, 'Training Summary', fontsize=14, fontweight='bold', 
                    transform=axes[1, 2].transAxes, va='top')
    
    stats_text = f"""
Total Epochs: {len(epochs)}
Best Validation Loss: {min(val_losses):.4f} (Epoch {best_epoch})
Best Accuracy: {max(val_accuracies):.4f} (Epoch {best_acc_epoch})
Best Perplexity: {min(val_perplexities):.4f} (Epoch {best_perp_epoch})

Final Results:
• Final Val Loss: {val_losses[-1]:.4f}
• Final Accuracy: {val_accuracies[-1]:.4f}
• Final Perplexity: {val_perplexities[-1]:.4f}

Training Progress:
• Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%
• Val Loss Reduction: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%
• Accuracy Gain: {((val_accuracies[-1] - val_accuracies[0]) * 100):.1f}%
    """
    
    axes[1, 2].text(0.05, 0.85, stats_text, fontsize=10, transform=axes[1, 2].transAxes, 
                    va='top', ha='left', family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        plot_filename = os.path.join(output_dir, f'{model_name}_analysis.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to: {plot_filename}")
    
    plt.show()


def plot_loss_curves_only(training_history: List[Dict], model_name: str, output_dir: str = None):
    """Create a focused plot showing only loss curves"""
    
    epochs = [epoch['epoch'] for epoch in training_history]
    train_losses = [epoch['train_loss'] for epoch in training_history]
    val_losses = [epoch['val_loss'] for epoch in training_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Mark best epoch
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_losses[best_epoch_idx]
    plt.plot(best_epoch, best_val_loss, 'go', markersize=12, label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    
    plt.title(f'Training Progress: {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with key stats
    stats_text = f'Final Training Loss: {train_losses[-1]:.4f}\nFinal Validation Loss: {val_losses[-1]:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        plot_filename = os.path.join(output_dir, f'{model_name}_loss_curves.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Loss curves plot saved to: {plot_filename}")
    
    plt.show()


def compare_multiple_runs(results_files: List[str]):
    """Compare training results from multiple runs"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, results_file in enumerate(results_files):
        try:
            results = load_training_results(results_file)
            training_history = results.get('training_history', [])
            
            if training_history:
                epochs = [epoch['epoch'] for epoch in training_history]
                val_losses = [epoch['val_loss'] for epoch in training_history]
                
                model_name = results.get('model_name', f'Model {i+1}')
                color = colors[i % len(colors)]
                
                best_loss = min(val_losses) if val_losses else None
                if best_loss is not None:
                    label = f'{model_name} (Best: {best_loss:.4f})'
                else:
                    label = f'{model_name} (No data)'
                plt.plot(epochs, val_losses, color=color, linewidth=2, label=label)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Training Comparison - Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()