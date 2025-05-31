#!/usr/bin/env python3
"""
Test training plots generation without requiring PyTorch
"""

import os
import json
import random
from datetime import datetime

def generate_dummy_training_data():
    """Generate dummy training data for plotting"""
    iterations = []
    losses = []
    accuracies = []
    
    # Simulate 100 training iterations with decreasing loss
    base_loss = 2.5
    for i in range(100):
        iterations.append(i)
        # Decreasing loss with some noise
        loss = base_loss * (0.99 ** i) + random.uniform(-0.1, 0.1)
        losses.append(max(0.1, loss))  # Don't let loss go below 0.1
        
        # Increasing accuracy
        accuracy = min(0.95, 0.3 + (i / 100) * 0.6 + random.uniform(-0.05, 0.05))
        accuracies.append(max(0.0, accuracy))
    
    return {
        'iterations': iterations,
        'losses': losses,
        'accuracies': accuracies,
        'learning_rates': [0.001 * (0.95 ** (i // 10)) for i in iterations],
        'alive_marbles': [20 - (i // 20) for i in iterations]  # Decreasing marble count
    }

def create_training_plots():
    """Create training plots in the training_plots directory"""
    
    # Ensure training_plots directory exists
    os.makedirs('training_plots', exist_ok=True)
    
    # Generate dummy data
    data = generate_dummy_training_data()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON data
    json_filename = f"training_plots/training_loss_data_{timestamp}.json"
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'stats': {
            'min_loss': min(data['losses']),
            'max_loss': max(data['losses']),
            'final_loss': data['losses'][-1],
            'total_points': len(data['losses']),
            'final_accuracy': data['accuracies'][-1],
            'final_marbles': data['alive_marbles'][-1]
        }
    }
    
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📊 Training JSON data saved: {json_filename}")
    
    # Create HTML plot
    html_filename = f"training_plots/training_loss_data_{timestamp}.html"
    create_html_plot(json_data, html_filename)
    
    print(f"🌐 Training HTML plot saved: {html_filename}")
    print(f"📈 Open {html_filename} in your browser to view the training plots")

def create_html_plot(data, filename):
    """Create HTML plot from training data"""
    
    stats = data['stats']
    training_data = data['data']
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Loss Plot - {data['timestamp']}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-box {{ background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-number {{ font-size: 1.5em; font-weight: bold; color: #1976d2; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .plot-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Training Loss Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{stats['total_points']}</div>
                <div class="stat-label">Iterations</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['final_loss']:.4f}</div>
                <div class="stat-label">Final Loss</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['min_loss']:.4f}</div>
                <div class="stat-label">Min Loss</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['final_accuracy']:.3f}</div>
                <div class="stat-label">Final Accuracy</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{stats['final_marbles']}</div>
                <div class="stat-label">Final Marbles</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h3>Training Loss Over Time</h3>
            <div id="lossPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Training Accuracy Over Time</h3>
            <div id="accuracyPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Alive Marbles Over Time</h3>
            <div id="marblesPlot" style="width:100%; height:400px;"></div>
        </div>
    </div>
    
    <script>
        // Loss plot
        var lossTrace = {{
            x: {training_data['iterations']},
            y: {training_data['losses']},
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: {{color: '#f44336'}}
        }};
        
        var lossLayout = {{
            title: 'Training Loss',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Loss'}},
            showlegend: false
        }};
        
        Plotly.newPlot('lossPlot', [lossTrace], lossLayout);
        
        // Accuracy plot
        var accuracyTrace = {{
            x: {training_data['iterations']},
            y: {training_data['accuracies']},
            type: 'scatter',
            mode: 'lines',
            name: 'Accuracy',
            line: {{color: '#4caf50'}}
        }};
        
        var accuracyLayout = {{
            title: 'Training Accuracy',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Accuracy'}},
            showlegend: false
        }};
        
        Plotly.newPlot('accuracyPlot', [accuracyTrace], accuracyLayout);
        
        // Marbles plot
        var marblesTrace = {{
            x: {training_data['iterations']},
            y: {training_data['alive_marbles']},
            type: 'scatter',
            mode: 'lines',
            name: 'Alive Marbles',
            line: {{color: '#2196f3'}}
        }};
        
        var marblesLayout = {{
            title: 'Alive Marbles Over Time',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Alive Marbles'}},
            showlegend: false
        }};
        
        Plotly.newPlot('marblesPlot', [marblesTrace], marblesLayout);
    </script>
</body>
</html>"""
    
    with open(filename, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    print("🧪 Testing training plots generation...")
    create_training_plots()
    print("✅ Training plots test complete!")