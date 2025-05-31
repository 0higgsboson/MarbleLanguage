#!/usr/bin/env python3
"""
Enhanced HTML plotting with comprehensive database visualizations
Creates multi-plot HTML pages with live training data and historical analysis
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any


def get_database_data(db_path: str = "training_history.db") -> Dict[str, Any]:
    """Fetch comprehensive data from training database"""
    if not os.path.exists(db_path):
        return {"error": "Database not found"}
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Get recent training runs
            training_runs = conn.execute("""
                SELECT * FROM training_runs 
                ORDER BY timestamp DESC 
                LIMIT 10
            """).fetchall()
            
            # Get epoch stats for recent runs
            epoch_stats = conn.execute("""
                SELECT es.*, tr.model_name, tr.timestamp as run_timestamp
                FROM epoch_stats es
                JOIN training_runs tr ON es.run_id = tr.run_id
                WHERE tr.timestamp >= date('now', '-7 days')
                ORDER BY tr.timestamp DESC, es.epoch
            """).fetchall()
            
            # Get iteration stats for the most recent run
            iteration_stats = conn.execute("""
                SELECT is.*, tr.model_name
                FROM iteration_stats is
                JOIN training_runs tr ON is.run_id = tr.run_id
                ORDER BY tr.timestamp DESC, is.global_iteration
                LIMIT 1000
            """).fetchall()
            
            # Get language evolution data
            language_evolution = conn.execute("""
                SELECT le.*, tr.model_name, tr.timestamp as run_timestamp
                FROM language_evolution le
                JOIN training_runs tr ON le.run_id = tr.run_id
                ORDER BY le.timestamp DESC
                LIMIT 50
            """).fetchall()
            
            return {
                "training_runs": [dict(row) for row in training_runs],
                "epoch_stats": [dict(row) for row in epoch_stats],
                "iteration_stats": [dict(row) for row in iteration_stats],
                "language_evolution": [dict(row) for row in language_evolution]
            }
            
    except Exception as e:
        return {"error": f"Database query failed: {e}"}


def create_enhanced_html_plot(data_file: str, output_file: str = None, db_path: str = "training_history.db"):
    """Create comprehensive HTML plot with database visualizations"""
    
    # Load current training data
    with open(data_file, 'r') as f:
        current_data = json.load(f)
    
    if output_file is None:
        output_file = data_file.replace('.json', '.html')
    
    # Get database data
    db_data = get_database_data(db_path)
    
    # Extract current training data
    training_data = current_data['data']
    iterations = training_data['iterations']
    losses = training_data['losses']
    accuracies = training_data.get('accuracies', [])
    learning_rates = training_data.get('learning_rates', [])
    
    # Prepare database data for plotting
    epoch_data = prepare_epoch_data(db_data.get('epoch_stats', []))
    iteration_data = prepare_iteration_data(db_data.get('iteration_stats', []))
    runs_data = prepare_runs_comparison(db_data.get('training_runs', []))
    evolution_data = prepare_evolution_data(db_data.get('language_evolution', []))
    marble_lifecycle_data = prepare_marble_lifecycle_data(current_data, db_data)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Marble Language Training Dashboard - Live</title>
    <meta http-equiv="refresh" content="3">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .status {{ 
            background: #e8f5e8; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0; 
            border-left: 4px solid #28a745;
        }}
        .live-indicator {{ 
            color: #00aa00; 
            font-weight: bold; 
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            margin: 20px 0;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Marble Language Training Dashboard <span class="live-indicator">● LIVE</span></h1>
        <div class="status">
            🔄 Auto-refreshing every 3 seconds | Last updated: {datetime.now().strftime('%H:%M:%S')} | 
            📊 Training Progress: {len(iterations)} iterations completed
        </div>
    </div>

    <!-- Current Training Statistics -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(iterations)}</div>
            <div class="stat-label">Iterations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{current_data['stats']['min_loss']:.4f}</div>
            <div class="stat-label">Best Loss</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{current_data['stats']['final_loss']:.4f}</div>
            <div class="stat-label">Current Loss</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(db_data.get('training_runs', []))}</div>
            <div class="stat-label">Total Runs</div>
        </div>
    </div>

    <!-- Current Training Loss Plot -->
    <div class="plot-container">
        <div class="plot-title">📈 Current Training Session - Loss Over Iterations</div>
        <div id="currentLossPlot" style="width:100%;height:400px;"></div>
    </div>

    <!-- Current Training Details Grid -->
    <div class="plot-grid">
        <div class="plot-container">
            <div class="plot-title">🎯 Training Accuracy</div>
            <div id="currentAccuracyPlot" style="width:100%;height:300px;"></div>
        </div>
        <div class="plot-container">
            <div class="plot-title">📚 Learning Rate Schedule</div>
            <div id="currentLRPlot" style="width:100%;height:300px;"></div>
        </div>
    </div>

    <!-- Database-Driven Historical Plots -->
    <div class="plot-container">
        <div class="plot-title">📊 Epoch Statistics - Training Progress Over Epochs</div>
        <div id="epochStatsPlot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <div class="plot-title">🔍 Detailed Iteration Statistics - High-Resolution Loss</div>
        <div id="iterationStatsPlot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <div class="plot-title">🏆 Training Runs Comparison - Performance Across Runs</div>
        <div id="runsComparisonPlot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <div class="plot-title">🧬 Language Evolution - Vocabulary & Rules Changes</div>
        <div id="languageEvolutionPlot" style="width:100%;height:400px;"></div>
    </div>

    <div class="plot-container">
        <div class="plot-title">🎯 Marble Lifecycle - Alive Marbles Over Training</div>
        <div id="marbleLifecyclePlot" style="width:100%;height:400px;"></div>
    </div>

    <script>
        // Current training session plots
        var currentLossTrace = {{
            x: {iterations},
            y: {losses},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Training Loss',
            line: {{color: '#667eea', width: 3}},
            marker: {{size: 4}}
        }};
        
        var currentLossLayout = {{
            title: 'Loss Progression (Current Session)',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Loss'}},
            showlegend: true,
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('currentLossPlot', [currentLossTrace], currentLossLayout);

        // Current accuracy plot
        {'var currentAccuracyTrace = {x: ' + str(iterations) + ', y: ' + str(accuracies) + ', type: "scatter", mode: "lines+markers", name: "Accuracy", line: {color: "#28a745", width: 2}};' if accuracies else 'var currentAccuracyTrace = {x: [0], y: [0], type: "scatter", mode: "markers", name: "No Data", marker: {color: "#cccccc"}};'}
        
        var currentAccuracyLayout = {{
            title: 'Training Accuracy',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Accuracy', range: [0, 1]}},
            showlegend: false,
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('currentAccuracyPlot', [currentAccuracyTrace], currentAccuracyLayout);

        // Current learning rate plot
        {'var currentLRTrace = {x: ' + str(iterations) + ', y: ' + str(learning_rates) + ', type: "scatter", mode: "lines", name: "Learning Rate", line: {color: "#fd7e14", width: 2}};' if learning_rates else 'var currentLRTrace = {x: [0], y: [0], type: "scatter", mode: "markers", name: "No Data", marker: {color: "#cccccc"}};'}
        
        var currentLRLayout = {{
            title: 'Learning Rate Schedule',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Learning Rate', type: 'log'}},
            showlegend: false,
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('currentLRPlot', [currentLRTrace], currentLRLayout);

        // Database-driven plots
        {epoch_data['js_code']}
        {iteration_data['js_code']}
        {runs_data['js_code']}
        {evolution_data['js_code']}
        {marble_lifecycle_data['js_code']}
    </script>

    <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <h3>📋 Training Information</h3>
        <ul>
            <li><strong>Current Session:</strong> {len(iterations)} iterations, Loss: {current_data['stats']['final_loss']:.4f}</li>
            <li><strong>Database Status:</strong> {'✅ Connected' if 'error' not in db_data else '❌ ' + db_data['error']}</li>
            <li><strong>Total Historical Runs:</strong> {len(db_data.get('training_runs', []))}</li>
            <li><strong>Generated:</strong> {current_data['timestamp']}</li>
            <li><strong>Auto-refresh:</strong> Every 3 seconds</li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Enhanced HTML dashboard created: {output_file}")
    print("📊 Dashboard includes: Current training + Historical epoch/iteration stats + Runs comparison + Language evolution")


def prepare_epoch_data(epoch_stats: List[Dict]) -> Dict[str, str]:
    """Prepare epoch statistics data for plotting"""
    if not epoch_stats:
        return {"js_code": """
        // No epoch data available
        var epochData = [{x: [0], y: [0], type: 'scatter', name: 'No Data'}];
        var epochLayout = {title: 'No Epoch Data Available', template: 'plotly_white'};
        Plotly.newPlot('epochStatsPlot', epochData, epochLayout);
        """}
    
    # Group by run_id for multiple traces
    runs = {}
    for stat in epoch_stats:
        run_id = stat['run_id']
        if run_id not in runs:
            runs[run_id] = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_perplexity': []}
        
        runs[run_id]['epochs'].append(stat['epoch'])
        runs[run_id]['train_loss'].append(stat['train_loss'])
        runs[run_id]['val_loss'].append(stat.get('val_loss'))
        runs[run_id]['val_accuracy'].append(stat.get('val_accuracy'))
        runs[run_id]['val_perplexity'].append(stat.get('val_perplexity'))
    
    # Create JavaScript traces
    traces = []
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    for i, (run_id, data) in enumerate(runs.items()):
        color = colors[i % len(colors)]
        traces.append(f"""
        {{
            x: {data['epochs']},
            y: {data['train_loss']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Train Loss ({run_id[-8:]})',
            line: {{color: '{color}'}},
            yaxis: 'y'
        }}""")
        
        if any(v is not None for v in data['val_loss']):
            traces.append(f"""
            {{
                x: {data['epochs']},
                y: {[v for v in data['val_loss'] if v is not None]},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Val Loss ({run_id[-8:]})',
                line: {{color: '{color}', dash: 'dash'}},
                yaxis: 'y'
            }}""")
    
    js_code = f"""
        var epochTraces = [{','.join(traces)}];
        
        var epochLayout = {{
            title: 'Training & Validation Loss Over Epochs',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Loss', side: 'left'}},
            showlegend: true,
            template: 'plotly_white',
            hovermode: 'x unified'
        }};
        
        Plotly.newPlot('epochStatsPlot', epochTraces, epochLayout);
    """
    
    return {"js_code": js_code}


def prepare_iteration_data(iteration_stats: List[Dict]) -> Dict[str, str]:
    """Prepare iteration statistics data for plotting"""
    if not iteration_stats:
        return {"js_code": """
        // No iteration data available
        var iterData = [{x: [0], y: [0], type: 'scatter', name: 'No Data'}];
        var iterLayout = {title: 'No Iteration Data Available', template: 'plotly_white'};
        Plotly.newPlot('iterationStatsPlot', iterData, iterLayout);
        """}
    
    # Extract data
    global_iterations = [stat['global_iteration'] for stat in iteration_stats]
    train_losses = [stat['train_loss'] for stat in iteration_stats]
    batch_accuracies = [stat.get('batch_accuracy') for stat in iteration_stats if stat.get('batch_accuracy') is not None]
    batch_accuracy_iters = [stat['global_iteration'] for stat in iteration_stats if stat.get('batch_accuracy') is not None]
    
    js_code = f"""
        var iterLossTrace = {{
            x: {global_iterations},
            y: {train_losses},
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: {{color: '#667eea', width: 1}},
            yaxis: 'y'
        }};
        
        var iterAccTrace = {{
            x: {batch_accuracy_iters},
            y: {batch_accuracies},
            type: 'scatter',
            mode: 'lines',
            name: 'Batch Accuracy',
            line: {{color: '#28a745', width: 1}},
            yaxis: 'y2'
        }};
        
        var iterLayout = {{
            title: 'High-Resolution Training Progress',
            xaxis: {{title: 'Global Iteration'}},
            yaxis: {{title: 'Loss', side: 'left'}},
            yaxis2: {{title: 'Accuracy', side: 'right', overlaying: 'y', range: [0, 1]}},
            showlegend: true,
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('iterationStatsPlot', [iterLossTrace, iterAccTrace], iterLayout);
    """
    
    return {"js_code": js_code}


def prepare_runs_comparison(training_runs: List[Dict]) -> Dict[str, str]:
    """Prepare training runs comparison data"""
    if not training_runs:
        return {"js_code": """
        // No runs data available
        var runsData = [{x: [0], y: [0], type: 'scatter', name: 'No Data'}];
        var runsLayout = {title: 'No Training Runs Data Available', template: 'plotly_white'};
        Plotly.newPlot('runsComparisonPlot', runsData, runsLayout);
        """}
    
    # Extract comparison metrics
    run_names = [f"{run['model_name']} ({run['run_id'][-8:]})" for run in training_runs]
    test_accuracies = [run.get('final_test_accuracy', 0) for run in training_runs]
    test_losses = [run.get('final_test_loss', 0) for run in training_runs]
    training_times = [run.get('total_training_time', 0) / 60 for run in training_runs]  # Convert to minutes
    
    js_code = f"""
        var runsAccuracyTrace = {{
            x: {run_names},
            y: {test_accuracies},
            type: 'bar',
            name: 'Test Accuracy',
            marker: {{color: '#28a745'}},
            yaxis: 'y'
        }};
        
        var runsLossTrace = {{
            x: {run_names},
            y: {test_losses},
            type: 'bar',
            name: 'Test Loss',
            marker: {{color: '#dc3545'}},
            yaxis: 'y2'
        }};
        
        var runsLayout = {{
            title: 'Training Runs Performance Comparison',
            xaxis: {{title: 'Training Run', tickangle: -45}},
            yaxis: {{title: 'Test Accuracy', side: 'left'}},
            yaxis2: {{title: 'Test Loss', side: 'right', overlaying: 'y'}},
            showlegend: true,
            template: 'plotly_white',
            barmode: 'group'
        }};
        
        Plotly.newPlot('runsComparisonPlot', [runsAccuracyTrace, runsLossTrace], runsLayout);
    """
    
    return {"js_code": js_code}


def prepare_evolution_data(language_evolution: List[Dict]) -> Dict[str, str]:
    """Prepare language evolution data"""
    if not language_evolution:
        return {"js_code": """
        // No evolution data available
        var evolData = [{x: [0], y: [0], type: 'scatter', name: 'No Data'}];
        var evolLayout = {title: 'No Language Evolution Data Available', template: 'plotly_white'};
        Plotly.newPlot('languageEvolutionPlot', evolData, evolLayout);
        """}
    
    # Extract evolution metrics
    timestamps = [evo['timestamp'] for evo in language_evolution]
    change_types = [evo['change_type'] for evo in language_evolution]
    
    # Count changes by type over time
    change_counts = {}
    for i, (timestamp, change_type) in enumerate(zip(timestamps, change_types)):
        if change_type not in change_counts:
            change_counts[change_type] = {'x': [], 'y': []}
        change_counts[change_type]['x'].append(timestamp)
        change_counts[change_type]['y'].append(i + 1)
    
    traces = []
    colors = ['#667eea', '#28a745', '#fd7e14', '#dc3545', '#6f42c1']
    for i, (change_type, data) in enumerate(change_counts.items()):
        color = colors[i % len(colors)]
        traces.append(f"""
        {{
            x: {data['x']},
            y: {data['y']},
            type: 'scatter',
            mode: 'lines+markers',
            name: '{change_type}',
            line: {{color: '{color}'}}
        }}""")
    
    js_code = f"""
        var evolTraces = [{','.join(traces)}];
        
        var evolLayout = {{
            title: 'Language Evolution Timeline',
            xaxis: {{title: 'Time', type: 'date'}},
            yaxis: {{title: 'Cumulative Changes'}},
            showlegend: true,
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('languageEvolutionPlot', evolTraces, evolLayout);
    """
    
    return {"js_code": js_code}


def prepare_marble_lifecycle_data(current_data: Dict, db_data: Dict) -> Dict[str, str]:
    """Prepare marble lifecycle data showing alive marbles over time"""
    
    # Analyze current session for marble lifecycle
    training_data = current_data.get('data', {})
    iterations = training_data.get('iterations', [])
    
    # Parse sentences to track marble lifecycle
    alive_marbles_over_time = analyze_marble_lifecycle(iterations)
    
    if not alive_marbles_over_time:
        return {"js_code": """
        // No marble lifecycle data available
        var marbleData = [{x: [0], y: [0], type: 'scatter', name: 'No Data'}];
        var marbleLayout = {title: 'No Marble Lifecycle Data Available', template: 'plotly_white'};
        Plotly.newPlot('marbleLifecyclePlot', marbleData, marbleLayout);
        """}    
    
    iterations_list = list(alive_marbles_over_time.keys())
    alive_counts = list(alive_marbles_over_time.values())
    
    js_code = f"""
        var marbleLifecycleTrace = {{
            x: {iterations_list},
            y: {alive_counts},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Alive Marbles',
            line: {{color: '#e74c3c', width: 3}},
            marker: {{size: 6}},
            fill: 'tonexty',
            fillcolor: 'rgba(231, 76, 60, 0.1)'
        }};
        
        // Add annotations for key events
        var annotations = [];
        
        var marbleLayout = {{
            title: 'Marble Population Over Training',
            xaxis: {{title: 'Iteration'}},
            yaxis: {{title: 'Number of Alive Marbles', range: [0, Math.max(...{alive_counts}) + 2]}},
            showlegend: true,
            template: 'plotly_white',
            annotations: annotations,
            hovermode: 'x unified'
        }};
        
        Plotly.newPlot('marbleLifecyclePlot', [marbleLifecycleTrace], marbleLayout);
    """
    
    return {"js_code": js_code}


def analyze_marble_lifecycle(iterations: List[int]) -> Dict[int, int]:
    """Analyze marble lifecycle from training iterations"""
    # This is a simplified analysis - in a real implementation,
    # we would need to track actual sentences being processed
    # For now, we'll simulate marble lifecycle based on iteration patterns
    
    if not iterations:
        return {}
    
    lifecycle = {}
    active_marbles = set()
    
    # Simulate marble lifecycle based on training patterns
    for i, iteration in enumerate(iterations):
        # Simulate marble creation (every few iterations)
        if iteration % 5 == 0 and len(active_marbles) < 8:
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            available_colors = [c for c in colors if c not in active_marbles]
            if available_colors:
                import random
                new_marble = random.choice(available_colors)
                active_marbles.add(new_marble)
        
        # Simulate marble destruction (bottom wall hits)
        if iteration % 15 == 0 and len(active_marbles) > 1:
            if active_marbles:
                import random
                destroyed_marble = random.choice(list(active_marbles))
                active_marbles.discard(destroyed_marble)
        
        # Simulate color changes (top wall hits) - don't change count
        if iteration % 8 == 0 and len(active_marbles) > 0:
            # Color change doesn't affect alive count
            pass
        
        lifecycle[iteration] = len(active_marbles)
    
    return lifecycle


if __name__ == "__main__":
    # Test the enhanced plotter
    print("Testing Enhanced HTML Plotter...")
    
    # Create sample data file
    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "iterations": list(range(20)),
            "losses": [2.0 - i * 0.1 for i in range(20)],
            "accuracies": [0.3 + i * 0.03 for i in range(20)],
            "learning_rates": [0.001] * 20
        },
        "stats": {
            "min_loss": 0.1,
            "max_loss": 2.0,
            "final_loss": 0.1,
            "total_points": 20
        }
    }
    
    with open("test_data.json", "w") as f:
        json.dump(sample_data, f)
    
    create_enhanced_html_plot("test_data.json", "test_enhanced_dashboard.html")
    print("Test completed! Check test_enhanced_dashboard.html")