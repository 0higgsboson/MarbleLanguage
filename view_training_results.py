#!/usr/bin/env python3
"""
Simple training results viewer that doesn't require matplotlib/numpy
Shows training statistics and creates basic text-based plots
"""

import json
import os
import sys
from datetime import datetime


def load_training_results(results_file: str):
    """Load training results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {results_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {results_file}")
        return None


def print_training_summary(results: dict):
    """Print a summary of training results"""
    print("=" * 80)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"Model: {results.get('model_name', 'Unknown')}")
    print(f"Training completed: {results.get('training_time', 'Unknown')}")
    print(f"Vocabulary size: {results.get('vocab_size', 'Unknown')}")
    print(f"Number of sentences: {results.get('num_sentences', 'Unknown')}")
    print(f"Training time: {results.get('total_training_time_seconds', 0):.2f} seconds")
    
    print("\n" + "-" * 40)
    print("FINAL METRICS")
    print("-" * 40)
    
    # Final test results
    test_loss = results.get('test_loss', 'N/A')
    test_accuracy = results.get('test_accuracy', 'N/A')
    test_perplexity = results.get('test_perplexity', 'N/A')
    
    if test_loss != 'N/A':
        print(f"Test Loss: {test_loss:.4f}")
    else:
        print(f"Test Loss: {test_loss}")
        
    if test_accuracy != 'N/A':
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    else:
        print(f"Test Accuracy: {test_accuracy}")
        
    if test_perplexity != 'N/A':
        print(f"Test Perplexity: {test_perplexity:.4f}")
    else:
        print(f"Test Perplexity: {test_perplexity}")


def create_simple_text_plot(values: list, title: str, width: int = 60):
    """Create a simple text-based plot"""
    if not values or len(values) < 2:
        return f"{title}: Not enough data to plot"
    
    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val if max_val > min_val else 1.0
    
    lines = []
    lines.append(f"{title}")
    lines.append("-" * len(title))
    lines.append(f"Range: {min_val:.4f} to {max_val:.4f}")
    lines.append("")
    
    # Create plot
    plot_lines = []
    for i, val in enumerate(values):
        # Normalize value to plot width
        if value_range > 0:
            pos = int((val - min_val) / value_range * (width - 1))
        else:
            pos = width // 2
        
        line = [' '] * width
        line[pos] = '*'
        
        # Add value label
        plot_line = ''.join(line) + f" {val:.4f}"
        plot_lines.append(plot_line)
    
    # Show every nth line to avoid too much output
    step = max(1, len(plot_lines) // 20)  # Show max 20 lines
    for i in range(0, len(plot_lines), step):
        lines.append(f"{i:3d}: {plot_lines[i]}")
    
    if len(plot_lines) > 20:
        lines.append(f"... (showing every {step} points)")
    
    return "\n".join(lines)


def analyze_training_file(results_file: str):
    """Analyze a training results file"""
    print(f"\nAnalyzing: {results_file}")
    
    results = load_training_results(results_file)
    if not results:
        return
    
    # Print summary
    print_training_summary(results)
    
    # Check for training history data (multiple formats)
    training_history = results.get('training_history', [])
    terminal_data = results.get('data', {})
    
    if training_history:
        print("\n" + "=" * 80)
        print("TRAINING PROGRESSION")
        print("=" * 80)
        
        # Extract epoch data
        epochs = []
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch_data in training_history:
            epochs.append(epoch_data.get('epoch', 0))
            train_losses.append(epoch_data.get('train_loss', 0))
            val_losses.append(epoch_data.get('val_loss', 0))
            val_accuracies.append(epoch_data.get('val_accuracy', 0))
        
        # Show progression
        print(f"Total epochs: {len(epochs)}")
        if train_losses:
            print(f"Initial training loss: {train_losses[0]:.4f}")
            print(f"Final training loss: {train_losses[-1]:.4f}")
            print(f"Best training loss: {min(train_losses):.4f}")
        
        if val_losses:
            print(f"Initial validation loss: {val_losses[0]:.4f}")
            print(f"Final validation loss: {val_losses[-1]:.4f}")
            print(f"Best validation loss: {min(val_losses):.4f}")
        
        if val_accuracies:
            print(f"Initial validation accuracy: {val_accuracies[0]:.4f} ({val_accuracies[0]*100:.1f}%)")
            print(f"Final validation accuracy: {val_accuracies[-1]:.4f} ({val_accuracies[-1]*100:.1f}%)")
            print(f"Best validation accuracy: {max(val_accuracies):.4f} ({max(val_accuracies)*100:.1f}%)")
        
        # Create text plots
        if len(train_losses) > 1:
            print("\n" + create_simple_text_plot(train_losses, "Training Loss Progression"))
        
        if len(val_losses) > 1:
            print("\n" + create_simple_text_plot(val_losses, "Validation Loss Progression"))
        
        if len(val_accuracies) > 1:
            print("\n" + create_simple_text_plot(val_accuracies, "Validation Accuracy Progression"))
    
    elif terminal_data:
        print("\n" + "=" * 80)
        print("TRAINING PROGRESSION (Terminal Plotter Data)")
        print("=" * 80)
        
        # Extract terminal plotter data
        iterations = terminal_data.get('iterations', [])
        losses = terminal_data.get('losses', [])
        accuracies = terminal_data.get('accuracies', [])
        learning_rates = terminal_data.get('learning_rates', [])
        
        print(f"Total iterations: {len(iterations)}")
        if losses:
            print(f"Initial loss: {losses[0]:.4f}")
            print(f"Final loss: {losses[-1]:.4f}")
            print(f"Best loss: {min(losses):.4f}")
            print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
        
        if accuracies:
            non_zero_acc = [a for a in accuracies if a > 0]
            if non_zero_acc:
                print(f"Initial accuracy: {non_zero_acc[0]:.4f} ({non_zero_acc[0]*100:.1f}%)")
                print(f"Final accuracy: {accuracies[-1]:.4f} ({accuracies[-1]*100:.1f}%)")
                print(f"Best accuracy: {max(accuracies):.4f} ({max(accuracies)*100:.1f}%)")
        
        if learning_rates:
            print(f"Initial learning rate: {learning_rates[0]:.6f}")
            print(f"Final learning rate: {learning_rates[-1]:.6f}")
        
        # Create text plots for terminal data
        if len(losses) > 1:
            print("\n" + create_simple_text_plot(losses, "Training Loss Progression (Per Iteration)"))
        
        if len(accuracies) > 1:
            print("\n" + create_simple_text_plot(accuracies, "Batch Accuracy Progression"))
        
        # Show recent trend
        if len(losses) >= 10:
            recent_losses = losses[-10:]
            trend = "↓ decreasing" if recent_losses[-1] < recent_losses[0] else "↑ increasing"
            print(f"\nRecent trend (last 10 points): {trend}")
    
    else:
        print("\n⚠️  No training history found in results file")
        print("   This file contains only final results, not epoch-by-epoch progression")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 view_training_results.py <training_results.json> [additional_files...]")
        print("\nExample:")
        print("  python3 view_training_results.py marble_model/training_results.json")
        print("  python3 view_training_results.py marble_model/training_results.json other_model/training_results.json")
        return
    
    print("Marble Language v2 - Training Results Viewer")
    print("(No dependencies required - works without matplotlib/numpy)")
    
    results_files = sys.argv[1:]
    
    for results_file in results_files:
        if not os.path.exists(results_file):
            print(f"\n❌ File not found: {results_file}")
            continue
        
        analyze_training_file(results_file)
        
        if len(results_files) > 1:
            print("\n" + "=" * 100)
    
    print(f"\nAnalysis complete! Processed {len([f for f in results_files if os.path.exists(f)])} file(s)")
    
    # Suggest matplotlib installation for better plots
    print("\n💡 For better visualizations, install matplotlib:")
    print("   pip3 install matplotlib numpy")
    print("   python3 training_plotter.py marble_model/training_results.json")


if __name__ == "__main__":
    main()