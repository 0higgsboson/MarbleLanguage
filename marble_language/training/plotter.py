#!/usr/bin/env python3
"""
Enhanced Real-time Training Plotter
Plots loss and metrics at both iteration and epoch level during training
"""

import os
from typing import Dict, List, Optional
import logging

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    from matplotlib.animation import FuncAnimation
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealTimeLossPlotter:
    """Real-time loss plotter that updates during training iterations"""
    
    def __init__(self, output_dir: str, model_name: str, update_frequency: int = 10):
        self.output_dir = output_dir
        self.model_name = model_name
        self.update_frequency = update_frequency  # Update plot every N iterations
        self.plot_available = PLOTTING_AVAILABLE
        
        # Data storage
        self.iteration_data = {
            'iterations': [],
            'train_losses': [],
            'learning_rates': [],
            'batch_accuracies': []
        }
        
        self.epoch_data = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_perplexities': [],
            'learning_rates': []
        }
        
        if self.plot_available:
            self._setup_plots()
    
    def _setup_plots(self):
        """Initialize the plotting interface"""
        # Set up matplotlib style
        style.use('default')
        plt.ion()  # Interactive mode for real-time plotting
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle(f'Real-Time Training Progress: {self.model_name}', fontsize=16, fontweight='bold')
        
        # Iteration-level plots (top row)
        self.axes[0, 0].set_title('Training Loss (Per Iteration)')
        self.axes[0, 0].set_xlabel('Iteration')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Learning Rate (Per Iteration)')
        self.axes[0, 1].set_xlabel('Iteration')
        self.axes[0, 1].set_ylabel('Learning Rate')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[0, 2].set_title('Batch Accuracy (Per Iteration)')
        self.axes[0, 2].set_xlabel('Iteration')
        self.axes[0, 2].set_ylabel('Accuracy')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Epoch-level plots (bottom row)
        self.axes[1, 0].set_title('Training & Validation Loss (Per Epoch)')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Validation Accuracy (Per Epoch)')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Accuracy')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        self.axes[1, 2].set_title('Validation Perplexity (Per Epoch)')
        self.axes[1, 2].set_xlabel('Epoch')
        self.axes[1, 2].set_ylabel('Perplexity')
        self.axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_iteration(self, iteration: int, train_loss: float, learning_rate: float, 
                        batch_accuracy: Optional[float] = None):
        """Update plots with iteration-level data"""
        if not self.plot_available:
            return
        
        # Store data
        self.iteration_data['iterations'].append(iteration)
        self.iteration_data['train_losses'].append(train_loss)
        self.iteration_data['learning_rates'].append(learning_rate)
        self.iteration_data['batch_accuracies'].append(batch_accuracy or 0.0)
        
        # Update plots every N iterations
        if iteration % self.update_frequency == 0:
            self._update_iteration_plots()
    
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                    val_accuracy: float, val_perplexity: float, learning_rate: float):
        """Update plots with epoch-level data"""
        if not self.plot_available:
            return
        
        # Store data
        self.epoch_data['epochs'].append(epoch)
        self.epoch_data['train_losses'].append(train_loss)
        self.epoch_data['val_losses'].append(val_loss)
        self.epoch_data['val_accuracies'].append(val_accuracy)
        self.epoch_data['val_perplexities'].append(val_perplexity)
        self.epoch_data['learning_rates'].append(learning_rate)
        
        self._update_epoch_plots()
    
    def _update_iteration_plots(self):
        """Update the iteration-level plots"""
        try:
            # Clear and redraw iteration plots
            for i in range(3):
                self.axes[0, i].clear()
            
            iterations = self.iteration_data['iterations']
            
            # Plot 1: Training Loss
            if self.iteration_data['train_losses']:
                self.axes[0, 0].plot(iterations, self.iteration_data['train_losses'], 
                                   'b-', linewidth=1, alpha=0.7)
                self.axes[0, 0].set_title('Training Loss (Per Iteration)')
                self.axes[0, 0].set_xlabel('Iteration')
                self.axes[0, 0].set_ylabel('Loss')
                self.axes[0, 0].grid(True, alpha=0.3)
                
                # Add moving average
                if len(self.iteration_data['train_losses']) > 50:
                    window = min(50, len(self.iteration_data['train_losses']) // 10)
                    moving_avg = self._moving_average(self.iteration_data['train_losses'], window)
                    avg_iterations = iterations[-len(moving_avg):]
                    self.axes[0, 0].plot(avg_iterations, moving_avg, 'r-', linewidth=2, 
                                       label=f'Moving Avg ({window})')
                    self.axes[0, 0].legend()
            
            # Plot 2: Learning Rate
            if self.iteration_data['learning_rates']:
                self.axes[0, 1].plot(iterations, self.iteration_data['learning_rates'], 
                                   'orange', linewidth=1)
                self.axes[0, 1].set_title('Learning Rate (Per Iteration)')
                self.axes[0, 1].set_xlabel('Iteration')
                self.axes[0, 1].set_ylabel('Learning Rate')
                self.axes[0, 1].grid(True, alpha=0.3)
                self.axes[0, 1].set_yscale('log')  # Log scale for learning rate
            
            # Plot 3: Batch Accuracy
            if self.iteration_data['batch_accuracies']:
                self.axes[0, 2].plot(iterations, self.iteration_data['batch_accuracies'], 
                                   'g-', linewidth=1, alpha=0.7)
                self.axes[0, 2].set_title('Batch Accuracy (Per Iteration)')
                self.axes[0, 2].set_xlabel('Iteration')
                self.axes[0, 2].set_ylabel('Accuracy')
                self.axes[0, 2].set_ylim(0, 1)
                self.axes[0, 2].grid(True, alpha=0.3)
            
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.warning(f"Error updating iteration plots: {e}")
    
    def _update_epoch_plots(self):
        """Update the epoch-level plots"""
        try:
            # Clear and redraw epoch plots
            for i in range(3):
                self.axes[1, i].clear()
            
            epochs = self.epoch_data['epochs']
            
            # Plot 1: Training & Validation Loss
            if self.epoch_data['train_losses']:
                self.axes[1, 0].plot(epochs, self.epoch_data['train_losses'], 
                                   'b-', label='Training Loss', linewidth=2)
            if self.epoch_data['val_losses']:
                self.axes[1, 0].plot(epochs, self.epoch_data['val_losses'], 
                                   'r-', label='Validation Loss', linewidth=2)
            self.axes[1, 0].set_title('Training & Validation Loss (Per Epoch)')
            self.axes[1, 0].set_xlabel('Epoch')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 2: Validation Accuracy
            if self.epoch_data['val_accuracies']:
                self.axes[1, 1].plot(epochs, self.epoch_data['val_accuracies'], 
                                   'g-', linewidth=2)
                self.axes[1, 1].set_title('Validation Accuracy (Per Epoch)')
                self.axes[1, 1].set_xlabel('Epoch')
                self.axes[1, 1].set_ylabel('Accuracy')
                self.axes[1, 1].set_ylim(0, 1)
                self.axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 3: Validation Perplexity
            if self.epoch_data['val_perplexities']:
                self.axes[1, 2].plot(epochs, self.epoch_data['val_perplexities'], 
                                   'purple', linewidth=2)
                self.axes[1, 2].set_title('Validation Perplexity (Per Epoch)')
                self.axes[1, 2].set_xlabel('Epoch')
                self.axes[1, 2].set_ylabel('Perplexity')
                self.axes[1, 2].grid(True, alpha=0.3)
            
            # Update main title with current stats
            if epochs:
                current_epoch = epochs[-1]
                current_train_loss = self.epoch_data['train_losses'][-1] if self.epoch_data['train_losses'] else 0
                current_val_loss = self.epoch_data['val_losses'][-1] if self.epoch_data['val_losses'] else 0
                current_val_acc = self.epoch_data['val_accuracies'][-1] if self.epoch_data['val_accuracies'] else 0
                
                self.fig.suptitle(f'Real-Time Training Progress: {self.model_name}\n'
                                f'Epoch {current_epoch} | Train Loss: {current_train_loss:.4f} | '
                                f'Val Loss: {current_val_loss:.4f} | Val Acc: {current_val_acc:.3f}', 
                                fontsize=14, fontweight='bold')
            
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            logger.warning(f"Error updating epoch plots: {e}")
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
        
        averages = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            averages.append(avg)
        return averages
    
    def save_plots(self, filename_prefix: str = None):
        """Save current plots to files"""
        if not self.plot_available:
            return
        
        try:
            if filename_prefix is None:
                filename_prefix = f"{self.model_name}_training_progress"
            
            # Save the main plot
            plot_filename = os.path.join(self.output_dir, f'{filename_prefix}.png')
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to: {plot_filename}")
            
            # Save iteration data as well
            self._save_loss_curve_data(filename_prefix)
            
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")
    
    def _save_loss_curve_data(self, filename_prefix: str):
        """Save raw plotting data for later analysis"""
        try:
            import json
            
            data_filename = os.path.join(self.output_dir, f'{filename_prefix}_data.json')
            
            plot_data = {
                'iteration_data': self.iteration_data,
                'epoch_data': self.epoch_data,
                'model_name': self.model_name,
                'update_frequency': self.update_frequency
            }
            
            with open(data_filename, 'w') as f:
                json.dump(plot_data, f, indent=2)
            
            logger.info(f"Plot data saved to: {data_filename}")
            
        except Exception as e:
            logger.warning(f"Could not save plot data: {e}")
    
    def close(self):
        """Close the plotting interface"""
        if self.plot_available:
            plt.ioff()  # Turn off interactive mode
            plt.close(self.fig)


class IterationLossTracker:
    """Simple loss tracker for iteration-level statistics"""
    
    def __init__(self, smoothing_factor: float = 0.1):
        self.smoothing_factor = smoothing_factor
        self.losses = []
        self.smoothed_loss = None
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
    
    def update(self, loss: float) -> Dict[str, float]:
        """Update loss statistics"""
        self.losses.append(loss)
        
        # Update smoothed loss (exponential moving average)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = (1 - self.smoothing_factor) * self.smoothed_loss + self.smoothing_factor * loss
        
        # Update min/max
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)
        
        return {
            'current_loss': loss,
            'smoothed_loss': self.smoothed_loss,
            'min_loss': self.min_loss,
            'max_loss': self.max_loss,
            'total_iterations': len(self.losses)
        }


if __name__ == "__main__":
    # Test the plotter
    import time
    import math
    
    if PLOTTING_AVAILABLE:
        plotter = RealTimeLossPlotter("./test_output", "Test Model", update_frequency=5)
        
        print("Testing real-time loss plotting...")
        
        # Simulate training iterations
        for iteration in range(100):
            # Simulate decreasing loss with noise
            train_loss = 2.0 * math.exp(-iteration/50) + 0.1 * (0.5 - abs(0.5 - (iteration % 20) / 20))
            learning_rate = 0.001 * (0.95 ** (iteration // 10))
            batch_accuracy = min(0.95, 0.3 + 0.7 * (1 - math.exp(-iteration/30)))
            
            plotter.update_iteration(iteration, train_loss, learning_rate, batch_accuracy)
            
            # Simulate epoch updates every 20 iterations
            if iteration % 20 == 19:
                epoch = iteration // 20
                val_loss = train_loss * 1.1 + 0.05
                val_accuracy = batch_accuracy * 0.95
                val_perplexity = math.exp(val_loss)
                
                plotter.update_epoch(epoch, train_loss, val_loss, val_accuracy, val_perplexity, learning_rate)
            
            time.sleep(0.1)  # Simulate training time
        
        print("Test complete. Saving plots...")
        os.makedirs("./test_output", exist_ok=True)
        plotter.save_plots()
        plotter.close()
        
    else:
        print("Matplotlib not available. Install with 'pip install matplotlib' to test plotting.")