#!/usr/bin/env python3
"""
Simple terminal-based plotting for training loss when matplotlib is not available
Creates ASCII plots and saves data for later visualization
"""

import os
import json
import math
from typing import List, Dict, Any
from datetime import datetime


class TerminalPlotter:
    """Simple ASCII plotting for terminal display"""
    
    def __init__(self, width: int = 80, height: int = 20):
        self.width = width
        self.height = height
        self.data = {
            'iterations': [],
            'losses': [],
            'accuracies': [],
            'learning_rates': [],
            'alive_marbles': []  # Track alive marbles over time
        }
        self.marble_population = 20  # Start with 20 marbles
        self.ecosystem_finite = False  # Track if any marble has died
    
    def add_point(self, iteration: int, loss: float, accuracy: float = None, lr: float = None, alive_marbles: int = None):
        """Add a data point"""
        self.data['iterations'].append(iteration)
        self.data['losses'].append(loss)
        self.data['accuracies'].append(accuracy or 0.0)
        self.data['learning_rates'].append(lr or 0.0)
        
        # Simulate finite marble ecosystem if not provided
        if alive_marbles is None:
            # Finite ecosystem: start with 20, can only decrease
            if iteration % 30 == 0 and self.marble_population > 1:
                self.marble_population -= 1  # Marble destroyed by bottom wall
                self.ecosystem_finite = True  # Mark ecosystem as finite
            # No new marbles can be created once any die
            alive_marbles = max(0, self.marble_population)
        
        self.data['alive_marbles'].append(alive_marbles)
    
    def plot_loss(self) -> str:
        """Create ASCII plot of loss over iterations"""
        if len(self.data['losses']) < 2:
            return "Not enough data points to plot"
        
        losses = self.data['losses']
        iterations = self.data['iterations']
        
        min_loss = min(losses)
        max_loss = max(losses)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
        
        min_iter = min(iterations)
        max_iter = max(iterations)
        iter_range = max_iter - min_iter if max_iter > min_iter else 1.0
        
        # Create plot grid
        plot = []
        for y in range(self.height):
            row = [' '] * self.width
            plot.append(row)
        
        # Plot points
        for i, (iteration, loss) in enumerate(zip(iterations, losses)):
            # Normalize to plot coordinates
            x = int((iteration - min_iter) / iter_range * (self.width - 1))
            y = int((1 - (loss - min_loss) / loss_range) * (self.height - 1))
            
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            plot[y][x] = '*'
        
        # Convert to string
        result = []
        result.append(f"Loss Plot (iterations {min_iter}-{max_iter})")
        result.append(f"Loss range: {min_loss:.4f} - {max_loss:.4f}")
        result.append("┌" + "─" * (self.width) + "┐")
        
        for y, row in enumerate(plot):
            # Add y-axis labels
            loss_val = max_loss - (y / (self.height - 1)) * loss_range
            label = f"{loss_val:.3f}"[:6].ljust(6)
            result.append(f"│{''.join(row)}│ {label}")
        
        result.append("└" + "─" * self.width + "┘")
        
        # Add x-axis labels
        x_labels = "    "
        for i in range(0, self.width, 20):
            iter_val = min_iter + (i / (self.width - 1)) * iter_range
            x_labels += f"{int(iter_val):>8}"
            x_labels += " " * 12
        result.append(x_labels[:self.width + 10])
        
        return "\n".join(result)
    
    def get_stats(self) -> str:
        """Get current training statistics"""
        if not self.data['losses']:
            return "No data available"
        
        current_loss = self.data['losses'][-1]
        min_loss = min(self.data['losses'])
        max_loss = max(self.data['losses'])
        avg_loss = sum(self.data['losses']) / len(self.data['losses'])
        
        # Calculate trend (last 10 points)
        recent_losses = self.data['losses'][-10:]
        if len(recent_losses) >= 2:
            trend = "↓ decreasing" if recent_losses[-1] < recent_losses[0] else "↑ increasing"
        else:
            trend = "→ stable"
        
        stats = f"""
Training Statistics:
  Current Loss: {current_loss:.4f}
  Min Loss: {min_loss:.4f}
  Max Loss: {max_loss:.4f}
  Average Loss: {avg_loss:.4f}
  Trend: {trend}
  Data Points: {len(self.data['losses'])}
"""
        return stats.strip()
    
    def save_data(self, filename: str):
        """Save plot data to JSON file"""
        data_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            'data': self.data,
            'stats': {
                'min_loss': min(self.data['losses']) if self.data['losses'] else 0,
                'max_loss': max(self.data['losses']) if self.data['losses'] else 0,
                'final_loss': self.data['losses'][-1] if self.data['losses'] else 0,
                'total_points': len(self.data['losses'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data_with_timestamp, f, indent=2)


class SimpleLossTracker:
    """Simple loss tracking and display for training"""
    
    def __init__(self, update_frequency: int = 5, display_frequency: int = 25):
        self.update_frequency = update_frequency
        self.display_frequency = display_frequency
        self.plotter = TerminalPlotter()
        self.iteration_count = 0
        self.current_html_file = None
        self.html_update_frequency = 15  # Update HTML every 15 iterations
    
    def update(self, iteration: int, loss: float, lr: float = None, accuracy: float = None, alive_marbles: int = None):
        """Update with new training data"""
        self.iteration_count += 1
        
        # Add data point
        if self.iteration_count % self.update_frequency == 0:
            self.plotter.add_point(iteration, loss, accuracy, lr, alive_marbles)
        
        # Display plot periodically - more frequent for short runs
        display_interval = self.display_frequency
        
        # For very short training runs, show plots more frequently
        if iteration < 50:
            display_interval = 10  # Show every 10 iterations for first 50
        elif iteration < 100:
            display_interval = 15  # Show every 15 iterations for next 50
        
        if self.iteration_count % display_interval == 0:
            self.display_progress()
        
        # Update HTML file periodically for live viewing
        if self.iteration_count % self.html_update_frequency == 0:
            self.update_live_html()
    
    def display_progress(self):
        """Display current training progress"""
        print("\n" + "=" * 80)
        print("TRAINING PROGRESS")
        print("=" * 80)
        print(self.plotter.get_stats())
        print("\n" + self.plotter.plot_loss())
        print("=" * 80)
    
    def display_final_progress(self):
        """Display final training progress (called at end of epoch/training)"""
        print("\n" + "=" * 80)
        print("FINAL TRAINING PROGRESS")
        print("=" * 80)
        print(self.plotter.get_stats())
        print("\n" + self.plotter.plot_loss())
        print("=" * 80)
    
    def save_intermediate_data(self, output_dir: str = ".", create_html: bool = True):
        """Save intermediate training data and update HTML plot"""
        # Create training_plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(plots_dir, f"training_loss_data_{timestamp}.json")
        self.plotter.save_data(filename)
        
        if create_html:
            html_file = filename.replace('.json', '.html')
            create_html_plot(filename, html_file)
            return filename, html_file
        
        return filename, None
    
    def set_live_html_file(self, html_file_path: str):
        """Set the HTML file to update during training"""
        self.current_html_file = html_file_path
    
    def update_live_html(self):
        """Update the live HTML file with current training data"""
        if not self.current_html_file or len(self.plotter.data['losses']) < 2:
            return
        
        try:
            # Save current data to a temporary JSON file
            temp_json = self.current_html_file.replace('.html', '_temp.json')
            data_with_timestamp = {
                'timestamp': datetime.now().isoformat(),
                'data': self.plotter.data,
                'stats': {
                    'min_loss': min(self.plotter.data['losses']) if self.plotter.data['losses'] else 0,
                    'max_loss': max(self.plotter.data['losses']) if self.plotter.data['losses'] else 0,
                    'final_loss': self.plotter.data['losses'][-1] if self.plotter.data['losses'] else 0,
                    'total_points': len(self.plotter.data['losses'])
                }
            }
            
            with open(temp_json, 'w') as f:
                json.dump(data_with_timestamp, f, indent=2)
            
            # Update the HTML file with enhanced dashboard
            try:
                from enhanced_html_plotter import create_enhanced_html_plot
                create_enhanced_html_plot(temp_json, self.current_html_file)
            except ImportError:
                # Fallback to basic plotter
                create_html_plot(temp_json, self.current_html_file)
            
            # Clean up temp file
            os.remove(temp_json)
            
        except Exception as e:
            # Don't interrupt training for plotting errors
            pass
    
    def save_final_data(self, output_dir: str = "."):
        """Save final training data"""
        # Always show final progress
        self.display_final_progress()
        
        # Create training_plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, "training_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(plots_dir, f"training_loss_data_{timestamp}.json")
        self.plotter.save_data(filename)
        print(f"\nTraining loss data saved to: {filename}")
        
        # Also create HTML plot
        html_file = filename.replace('.json', '.html')
        create_html_plot(filename, html_file)
        
        return filename


def create_html_plot(data_file: str, output_file: str = None):
    """Create an HTML plot from saved training data with auto-refresh"""
    # Try to use enhanced plotter first
    try:
        from enhanced_html_plotter import create_enhanced_html_plot
        create_enhanced_html_plot(data_file, output_file)
        return
    except ImportError:
        # Fallback to basic plotter
        pass
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if output_file is None:
        output_file = data_file.replace('.json', '.html')
    
    training_data = data['data']
    iterations = training_data['iterations']
    losses = training_data['losses']
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss Plot - Live</title>
    <meta http-equiv="refresh" content="3">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .status {{ background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .live-indicator {{ color: #00aa00; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Marble Language Training Loss <span class="live-indicator">● LIVE</span></h1>
    <div class="status">
        🔄 Auto-refreshing every 3 seconds | Last updated: {datetime.now().strftime('%H:%M:%S')}
    </div>
    <div id="lossPlot" style="width:100%;height:500px;"></div>
    
    <h2>Training Statistics</h2>
    <ul>
        <li>Data Points: {len(iterations)}</li>
        <li>Min Loss: {data['stats']['min_loss']:.4f}</li>
        <li>Max Loss: {data['stats']['max_loss']:.4f}</li>
        <li>Final Loss: {data['stats']['final_loss']:.4f}</li>
        <li>Generated: {data['timestamp']}</li>
    </ul>
    
    <script>
        var trace = {{
            x: {iterations},
            y: {losses},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Training Loss',
            line: {{color: 'blue'}}
        }};
        
        var layout = {{
            title: 'Training Loss Over Iterations',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Loss'}},
            showlegend: true
        }};
        
        Plotly.newPlot('lossPlot', [trace], layout);
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML plot created: {output_file}")
    print("Open this file in a web browser to view the interactive plot")


if __name__ == "__main__":
    # Test the terminal plotter
    print("Testing Terminal Plotter...")
    
    tracker = SimpleLossTracker(update_frequency=1, display_frequency=10)
    
    # Simulate training data
    for i in range(50):
        # Simulate decreasing loss with noise
        loss = 2.0 * math.exp(-i/20) + 0.1 * (0.5 - abs(0.5 - (i % 10) / 10))
        lr = 0.001 * (0.95 ** (i // 10))
        accuracy = min(0.95, 0.3 + 0.6 * (1 - math.exp(-i/15)))
        
        tracker.update(i, loss, lr, accuracy)
    
    # Save data and create HTML plot
    data_file = tracker.save_final_data()
    create_html_plot(data_file)
    
    print("\nTerminal plotter test completed!")