#!/usr/bin/env python3
"""
Test script for live plotting functionality
Simulates training with browser auto-refresh
"""

import time
import math
from simple_plotter import SimpleLossTracker
from browser_launcher import setup_live_training_view


def test_live_plotting():
    """Test the live plotting with browser auto-refresh"""
    print("🧪 Testing Live Plotting System")
    print("=" * 50)
    
    # Set up live browser view
    print("1. Setting up live browser view...")
    try:
        html_file = setup_live_training_view(".")
        print(f"✅ Browser opened with: {html_file}")
    except Exception as e:
        print(f"❌ Browser setup failed: {e}")
        return
    
    # Create tracker and link it to the HTML file
    print("\n2. Creating training tracker...")
    tracker = SimpleLossTracker(update_frequency=1, display_frequency=10)
    tracker.set_live_html_file(html_file)
    print("✅ Tracker linked to live HTML file")
    
    # Simulate training with live updates
    print("\n3. Starting simulated training...")
    print("📊 Watch your browser - it should update every 3 seconds!")
    print("💡 You should see the loss curve updating in real-time")
    
    for iteration in range(60):
        # Simulate realistic loss curve
        base_loss = 3.0 * math.exp(-iteration/20)
        noise = 0.1 * math.sin(iteration * 0.3)
        random_component = 0.05 * ((iteration * 23) % 100 - 50) / 50
        
        loss = base_loss + noise + random_component + 0.1
        loss = max(0.01, loss)
        
        # Learning rate decay
        lr = 0.001 * (0.95 ** (iteration // 10))
        
        # Accuracy improvement
        accuracy = min(0.95, 0.2 + 0.75 * (1 - math.exp(-iteration/15)))
        
        # Update tracker (this will update HTML every 15 iterations)
        tracker.update(iteration, loss, lr, accuracy)
        
        # Show progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {loss:.4f} (Check browser for live plot)")
        
        # Small delay to simulate training time
        time.sleep(0.2)
    
    print("\n4. Training simulation completed!")
    print("🎉 Final live update...")
    
    # Force final update
    tracker.update_live_html()
    
    # Save final data
    final_data_file = tracker.save_final_data(".")
    print(f"📁 Final data saved: {final_data_file}")
    
    print("\n✅ Test completed successfully!")
    print("🌐 The browser should show the complete training curve")
    print("🔄 HTML file was updated every 15 iterations during training")


if __name__ == "__main__":
    test_live_plotting()