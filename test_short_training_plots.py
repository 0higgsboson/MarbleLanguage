#!/usr/bin/env python3
"""
Test terminal plotting with short training runs
Shows how plots appear even with very few iterations
"""

import time
import math
from simple_plotter import SimpleLossTracker


def test_very_short_training():
    """Test with only 20 iterations to show adaptive plotting"""
    print("=" * 80)
    print("SHORT TRAINING TEST - 20 iterations")
    print("Demonstrating adaptive plotting frequencies")
    print("=" * 80)
    
    tracker = SimpleLossTracker(update_frequency=1, display_frequency=25)  # Use actual settings
    
    print("Training with only 20 iterations...")
    print("(Should see plots at iterations 10 and 20 due to adaptive frequency)")
    print()
    
    for iteration in range(20):
        # Simulate rapid loss decrease
        loss = 2.0 * math.exp(-iteration/8) + 0.1
        lr = 0.001 * (0.9 ** (iteration // 5))
        accuracy = min(0.9, 0.3 + 0.6 * (iteration / 19))
        
        tracker.update(iteration, loss, lr, accuracy)
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        time.sleep(0.05)  # Small delay
    
    print("\nTraining completed! Final plot will be shown when saving data...")
    
    # This will show the final plot
    data_file = tracker.save_final_data(".")
    print(f"Data saved to: {data_file}")


if __name__ == "__main__":
    test_very_short_training()