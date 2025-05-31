#!/usr/bin/env python3
"""
Demo of terminal plotting functionality for training loss visualization
Shows what users will see during training when matplotlib is not available
"""

import time
import math
from simple_plotter import SimpleLossTracker


def simulate_training():
    """Simulate a training run with terminal plotting"""
    print("=" * 80)
    print("MARBLE LANGUAGE TRAINING SIMULATION")
    print("Demonstrating terminal-based loss plotting (no matplotlib required)")
    print("=" * 80)
    
    # Create terminal tracker (smaller frequencies for demo)
    tracker = SimpleLossTracker(update_frequency=1, display_frequency=10)
    
    print("Starting simulated training...")
    print("Note: In real training, plots adapt to training length:")
    print("  - Every 10 iterations for first 50 iterations")  
    print("  - Every 15 iterations for iterations 50-100")
    print("  - Every 25 iterations for longer training")
    print("(Using even faster updates for demo purposes)")
    print("\n")
    
    # Simulate 60 iterations of training
    for iteration in range(60):
        # Simulate realistic loss curve: starts high, decreases with noise
        base_loss = 3.0 * math.exp(-iteration/20)  # Exponential decay
        noise = 0.2 * math.sin(iteration * 0.5)     # Some oscillation
        random_noise = 0.1 * ((iteration * 17) % 100 - 50) / 50  # Pseudo-random
        
        loss = base_loss + noise + random_noise + 0.1
        loss = max(0.01, loss)  # Ensure positive
        
        # Simulate learning rate decay
        lr = 0.001 * (0.95 ** (iteration // 10))
        
        # Simulate accuracy improvement
        accuracy = min(0.95, 0.2 + 0.75 * (1 - math.exp(-iteration/15)))
        
        # Update tracker (this will display plots periodically)
        tracker.update(iteration, loss, lr, accuracy)
        
        # Small delay for demo effect
        time.sleep(0.1)
        
        # Show progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f}, LR = {lr:.6f}, Acc = {accuracy:.3f}")
    
    print("\n" + "=" * 80)
    print("TRAINING SIMULATION COMPLETED")
    print("=" * 80)
    
    # Show final statistics
    print("\nFinal Training Statistics:")
    print(tracker.plotter.get_stats())
    
    # Save data and create HTML plot
    print("\nSaving training data...")
    data_file = tracker.save_final_data(".")
    
    try:
        from simple_plotter import create_html_plot
        html_file = data_file.replace('.json', '.html')
        create_html_plot(data_file, html_file)
        print(f"\n🎉 Success! Interactive plot created: {html_file}")
        print("   Open this file in any web browser to see a full interactive plot")
    except Exception as e:
        print(f"Failed to create HTML plot: {e}")
    
    return data_file


def show_usage_example():
    """Show how this integrates with real training"""
    print("\n" + "=" * 80)
    print("REAL TRAINING USAGE")
    print("=" * 80)
    
    usage_info = """
When you run actual training without matplotlib installed:

1. The training script automatically detects missing matplotlib
2. Falls back to terminal-based ASCII plotting
3. Shows loss plots in your terminal every 100 iterations
4. Saves data to JSON and creates interactive HTML plot at the end

Example output during real training:
  ❌ matplotlib not installed - using terminal plotting fallback
  ✓ Terminal plotting fallback available
  ✓ Terminal ASCII plotting enabled
    Loss plots will be displayed in terminal every 100 iterations

To run training with terminal plotting:
  python3 marble_transformer_pretraining.py

The plots you saw above will appear automatically during training!
"""
    print(usage_info)


if __name__ == "__main__":
    print("Marble Language v2 - Terminal Plotting Demo")
    
    # Run the simulation
    data_file = simulate_training()
    
    # Show usage information
    show_usage_example()
    
    print(f"\nDemo completed! Check the generated files:")
    print(f"  - Training data: {data_file}")
    print(f"  - Interactive plot: {data_file.replace('.json', '.html')}")
    print("\nThe terminal plotting system is ready for use!")