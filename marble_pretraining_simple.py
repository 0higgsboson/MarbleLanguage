#!/usr/bin/env python3
"""
Simplified Marble Language Pretraining
Creates training datasets and simulates training without requiring PyTorch
"""

import os
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple

# Import our marble language generator
from marble_language.core.generator import EnhancedMarbleSentenceGenerator


class SimpleMarbleTrainer:
    """Simplified trainer that creates datasets and simulates training"""
    
    def __init__(self, dataset_size: int = 1000):
        self.dataset_size = dataset_size
        self.generator = EnhancedMarbleSentenceGenerator()
        
    def create_vocabulary(self, sentences: List[str]) -> Dict[str, int]:
        """Create vocabulary from sentences"""
        vocab = {}
        vocab_idx = 0
        
        # Special tokens
        special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        for token in special_tokens:
            vocab[token] = vocab_idx
            vocab_idx += 1
        
        # Extract words from sentences
        all_words = set()
        for sentence in sentences:
            # Remove marble count if present
            sentence_text = sentence.split(' | ')[0] if ' | ' in sentence else sentence
            words = sentence_text.split()
            all_words.update(words)
        
        # Add words to vocabulary
        for word in sorted(all_words):
            if word not in vocab:
                vocab[word] = vocab_idx
                vocab_idx += 1
                
        return vocab
    
    def tokenize_sentences(self, sentences: List[str], vocab: Dict[str, int]) -> List[List[int]]:
        """Convert sentences to token sequences"""
        tokenized = []
        
        for sentence in sentences:
            # Remove marble count if present
            sentence_text = sentence.split(' | ')[0] if ' | ' in sentence else sentence
            words = sentence_text.split()
            
            # Convert to token IDs
            token_ids = [vocab['[BOS]']]
            for word in words:
                token_ids.append(vocab.get(word, vocab['[UNK]']))
            token_ids.append(vocab['[EOS]'])
            
            tokenized.append(token_ids)
            
        return tokenized
    
    def simulate_training(self, train_data: List[List[int]], vocab: Dict[str, int], epochs: int = 5):
        """Simulate training process and create visualizations"""
        
        print(f"🚀 Starting simulated training...")
        print(f"📊 Dataset size: {len(train_data)} sentences")
        print(f"📚 Vocabulary size: {len(vocab)} tokens")
        print(f"🔄 Training for {epochs} epochs")
        print("=" * 50)
        
        # Simulate training metrics
        training_history = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'test_losses': [],
            'accuracies': [],
            'perplexities': [],
            'marble_stats': [],
            'iteration_data': [],
            'epoch_stats': [],
            'language_evolution': []
        }
        
        for epoch in range(epochs):
            # Simulate training loss (decreases fastest)
            base_train_loss = 2.5 * (0.82 ** epoch) + random.uniform(-0.15, 0.15)
            train_loss = max(0.1, base_train_loss)
            
            # Simulate validation loss (decreases slower, higher than training)
            base_val_loss = 2.7 * (0.87 ** epoch) + random.uniform(-0.1, 0.2)
            val_loss = max(0.15, base_val_loss)
            
            # Simulate test loss (similar to validation but with more variance)
            base_test_loss = 2.6 * (0.86 ** epoch) + random.uniform(-0.2, 0.25)
            test_loss = max(0.12, base_test_loss)
            
            # Simulate increasing accuracy (based on validation performance)
            accuracy = min(0.95, 0.3 + (epoch / epochs) * 0.6 + random.uniform(-0.05, 0.05))
            
            # Calculate perplexity (based on validation loss)
            perplexity = 2 ** val_loss
            
            # Generate realistic marble stats for this epoch
            # Simulate marble ecosystem: starts at 20, drops to ~5-15 range with resurrections
            if epoch == 0:
                alive_marbles = 20  # Start with full population
            else:
                # Simulate ecosystem dynamics: fluctuate between 5-18 marbles
                # Some randomness but generally trending toward ecosystem equilibrium
                base_population = 8 + random.randint(-3, 7)  # Base around 8 ± 7 = 5-15 range
                # Add some training progression (slight decline over time)
                progression_factor = max(0.7, 1.0 - (epoch / epochs) * 0.3)
                alive_marbles = max(5, int(base_population * progression_factor))
                # Ensure realistic bounds
                alive_marbles = min(20, max(5, alive_marbles))
            
            # Simulate iteration-level data for this epoch (e.g., 10 iterations per epoch)
            iterations_per_epoch = 10
            epoch_iterations = []
            for iteration in range(iterations_per_epoch):
                iter_loss = train_loss + random.uniform(-0.1, 0.1)
                iter_acc = accuracy + random.uniform(-0.05, 0.05)
                iter_lr = 0.001 * (0.95 ** epoch)  # Learning rate decay
                
                global_iteration = epoch * iterations_per_epoch + iteration
                epoch_iterations.append({
                    'global_iteration': global_iteration,
                    'epoch': epoch + 1,
                    'iteration': iteration + 1,
                    'loss': max(0.05, iter_loss),
                    'accuracy': max(0.1, min(0.98, iter_acc)),
                    'learning_rate': iter_lr,
                    'marbles': alive_marbles
                })
            
            training_history['iteration_data'].extend(epoch_iterations)
            
            # Detailed epoch statistics
            epoch_stat = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'marbles': alive_marbles,
                'learning_rate': epoch_iterations[0]['learning_rate'],
                'min_iter_loss': min(iter_data['loss'] for iter_data in epoch_iterations),
                'max_iter_loss': max(iter_data['loss'] for iter_data in epoch_iterations),
                'loss_variance': sum((iter_data['loss'] - train_loss)**2 for iter_data in epoch_iterations) / len(epoch_iterations)
            }
            training_history['epoch_stats'].append(epoch_stat)
            
            # Language evolution metrics (vocabulary usage, sentence complexity, etc.)
            vocab_utilization = min(0.95, 0.4 + (epoch / epochs) * 0.5 + random.uniform(-0.05, 0.05))
            avg_sentence_length = 20 + (epoch / epochs) * 5 + random.uniform(-2, 2)
            collision_frequency = max(0.1, 0.6 - (epoch / epochs) * 0.2 + random.uniform(-0.1, 0.1))
            
            language_stat = {
                'epoch': epoch + 1,
                'vocabulary_utilization': vocab_utilization,
                'avg_sentence_length': max(15, avg_sentence_length),
                'collision_frequency': collision_frequency,
                'marble_diversity': alive_marbles / 20.0,
                'grammar_complexity': min(0.9, 0.3 + (epoch / epochs) * 0.6)
            }
            training_history['language_evolution'].append(language_stat)
            
            training_history['epochs'].append(epoch + 1)
            training_history['train_losses'].append(train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['test_losses'].append(test_loss)
            training_history['accuracies'].append(accuracy)
            training_history['perplexities'].append(perplexity)
            training_history['marble_stats'].append(alive_marbles)
            
            print(f"Epoch {epoch + 1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}, Acc: {accuracy:.3f}, Marbles: {alive_marbles}")
        
        return training_history
    
    def save_training_results(self, history: Dict, vocab: Dict[str, int], train_data: List[List[int]]):
        """Save training results and create visualizations"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        results = {
            'timestamp': timestamp,
            'dataset_size': len(train_data),
            'vocabulary_size': len(vocab),
            'training_history': history,
            'final_metrics': {
                'final_train_loss': history['train_losses'][-1],
                'final_val_loss': history['val_losses'][-1],
                'final_test_loss': history['test_losses'][-1],
                'final_accuracy': history['accuracies'][-1],
                'final_perplexity': history['perplexities'][-1],
                'final_marbles': history['marble_stats'][-1]
            }
        }
        
        # Save to marble_model directory
        os.makedirs('marble_model', exist_ok=True)
        results_file = f"marble_model/training_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Training results saved: {results_file}")
        
        # Create training visualization
        self.create_training_visualization(results, timestamp)
        
        return results_file
    
    def create_training_visualization(self, results: Dict, timestamp: str):
        """Create HTML visualization of training results"""
        
        history = results['training_history']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marble Language Training Results - {timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-box {{ background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-number {{ font-size: 1.8em; font-weight: bold; color: #1976d2; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .plot-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Marble Language Training Dashboard</h1>
            <p>Training completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{results['dataset_size']}</div>
                <div class="stat-label">Training Sentences</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results['vocabulary_size']}</div>
                <div class="stat-label">Vocabulary Size</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results['final_metrics']['final_val_loss']:.3f}</div>
                <div class="stat-label">Final Val Loss</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results['final_metrics']['final_accuracy']:.3f}</div>
                <div class="stat-label">Final Accuracy</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results['final_metrics']['final_perplexity']:.1f}</div>
                <div class="stat-label">Final Perplexity</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{results['final_metrics']['final_marbles']}</div>
                <div class="stat-label">Final Marbles</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h3>Training, Validation & Test Loss</h3>
            <div id="lossPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Training Accuracy</h3>
            <div id="accuracyPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Model Perplexity</h3>
            <div id="perplexityPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Marble Population During Training</h3>
            <div id="marblesPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Iteration-Level Training Progress</h3>
            <div id="iterationPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Epoch Statistics & Variance</h3>
            <div id="epochStatsPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Language Evolution Metrics</h3>
            <div id="languageEvolutionPlot" style="width:100%; height:400px;"></div>
        </div>
        
        <div class="plot-container">
            <h3>Learning Rate Schedule</h3>
            <div id="learningRatePlot" style="width:100%; height:400px;"></div>
        </div>
    </div>
    
    <script>
        // Training, Validation & Test Loss
        var trainLossTrace = {{
            x: {history['epochs']},
            y: {history['train_losses']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Training Loss',
            line: {{color: '#2196f3', width: 3}},
            marker: {{size: 6}}
        }};
        
        var valLossTrace = {{
            x: {history['epochs']},
            y: {history['val_losses']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Validation Loss',
            line: {{color: '#ff9800', width: 3}},
            marker: {{size: 6}}
        }};
        
        var testLossTrace = {{
            x: {history['epochs']},
            y: {history['test_losses']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Test Loss',
            line: {{color: '#f44336', width: 3}},
            marker: {{size: 6}}
        }};
        
        Plotly.newPlot('lossPlot', [trainLossTrace, valLossTrace, testLossTrace], {{
            title: 'Training, Validation & Test Loss Over Epochs',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Loss'}},
            showlegend: true,
            legend: {{x: 0.7, y: 0.95}}
        }});
        
        // Training Accuracy
        var accuracyTrace = {{
            x: {history['epochs']},
            y: {history['accuracies']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Accuracy',
            line: {{color: '#4caf50', width: 3}},
            marker: {{size: 8}}
        }};
        
        Plotly.newPlot('accuracyPlot', [accuracyTrace], {{
            title: 'Training Accuracy Over Epochs',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Accuracy'}},
            showlegend: false
        }});
        
        // Perplexity
        var perplexityTrace = {{
            x: {history['epochs']},
            y: {history['perplexities']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Perplexity',
            line: {{color: '#ff9800', width: 3}},
            marker: {{size: 8}}
        }};
        
        Plotly.newPlot('perplexityPlot', [perplexityTrace], {{
            title: 'Model Perplexity Over Epochs',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Perplexity'}},
            showlegend: false
        }});
        
        // Marble Population
        var marblesTrace = {{
            x: {history['epochs']},
            y: {history['marble_stats']},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Alive Marbles',
            line: {{color: '#2196f3', width: 3}},
            marker: {{size: 8}}
        }};
        
        Plotly.newPlot('marblesPlot', [marblesTrace], {{
            title: 'Marble Population During Training',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Alive Marbles'}},
            showlegend: false
        }});
        
        // Iteration-Level Training Progress
        var iterationData = {history['iteration_data']};
        var iterLossTrace = {{
            x: iterationData.map(d => d.global_iteration),
            y: iterationData.map(d => d.loss),
            type: 'scatter',
            mode: 'lines',
            name: 'Iteration Loss',
            line: {{color: '#673ab7', width: 2}},
            opacity: 0.7
        }};
        
        var iterAccTrace = {{
            x: iterationData.map(d => d.global_iteration),
            y: iterationData.map(d => d.accuracy),
            type: 'scatter',
            mode: 'lines',
            name: 'Iteration Accuracy',
            line: {{color: '#4caf50', width: 2}},
            yaxis: 'y2'
        }};
        
        Plotly.newPlot('iterationPlot', [iterLossTrace, iterAccTrace], {{
            title: 'Iteration-Level Training Progress',
            xaxis: {{title: 'Global Iteration'}},
            yaxis: {{title: 'Loss', side: 'left'}},
            yaxis2: {{title: 'Accuracy', side: 'right', overlaying: 'y'}},
            showlegend: true
        }});
        
        // Epoch Statistics & Variance
        var epochStatsData = {history['epoch_stats']};
        var lossVarianceTrace = {{
            x: epochStatsData.map(d => d.epoch),
            y: epochStatsData.map(d => d.loss_variance),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss Variance',
            line: {{color: '#ff5722', width: 3}},
            marker: {{size: 6}}
        }};
        
        var lossRangeTrace = {{
            x: epochStatsData.map(d => d.epoch),
            y: epochStatsData.map(d => d.max_iter_loss - d.min_iter_loss),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss Range',
            line: {{color: '#ff9800', width: 3}},
            marker: {{size: 6}}
        }};
        
        Plotly.newPlot('epochStatsPlot', [lossVarianceTrace, lossRangeTrace], {{
            title: 'Epoch Statistics: Loss Variance & Range',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Loss Metrics'}},
            showlegend: true
        }});
        
        // Language Evolution Metrics
        var languageData = {history['language_evolution']};
        var vocabUtilTrace = {{
            x: languageData.map(d => d.epoch),
            y: languageData.map(d => d.vocabulary_utilization),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Vocabulary Utilization',
            line: {{color: '#9c27b0', width: 3}},
            marker: {{size: 6}}
        }};
        
        var sentLengthTrace = {{
            x: languageData.map(d => d.epoch),
            y: languageData.map(d => d.avg_sentence_length),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Avg Sentence Length',
            line: {{color: '#3f51b5', width: 3}},
            marker: {{size: 6}},
            yaxis: 'y2'
        }};
        
        var collisionFreqTrace = {{
            x: languageData.map(d => d.epoch),
            y: languageData.map(d => d.collision_frequency),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Collision Frequency',
            line: {{color: '#009688', width: 3}},
            marker: {{size: 6}}
        }};
        
        var grammarComplexTrace = {{
            x: languageData.map(d => d.epoch),
            y: languageData.map(d => d.grammar_complexity),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Grammar Complexity',
            line: {{color: '#795548', width: 3}},
            marker: {{size: 6}}
        }};
        
        Plotly.newPlot('languageEvolutionPlot', [vocabUtilTrace, collisionFreqTrace, grammarComplexTrace], {{
            title: 'Language Evolution: Vocabulary, Collisions & Grammar',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Metrics (0-1 scale)'}},
            showlegend: true
        }});
        
        // Learning Rate Schedule
        var learningRateTrace = {{
            x: epochStatsData.map(d => d.epoch),
            y: epochStatsData.map(d => d.learning_rate),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Learning Rate',
            line: {{color: '#e91e63', width: 3}},
            marker: {{size: 8}}
        }};
        
        Plotly.newPlot('learningRatePlot', [learningRateTrace], {{
            title: 'Learning Rate Schedule',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Learning Rate', type: 'log'}},
            showlegend: false
        }});
    </script>
</body>
</html>"""
        
        # Save training visualization
        os.makedirs('training_plots', exist_ok=True)
        html_file = f"training_plots/training_results_{timestamp}.html"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Training visualization saved: {html_file}")
        
        # Try to open in browser
        try:
            import subprocess
            import platform
            
            abs_path = os.path.abspath(html_file)
            if platform.system().lower() == "darwin":  # macOS
                subprocess.run(["open", abs_path], check=False)
                print(f"🌐 Opened training results in browser")
        except Exception as e:
            print(f"⚠️  Please manually open: {html_file}")
    
    def create_training_runs_comparison(self):
        """Create a comparison visualization of multiple training runs"""
        import glob
        import json
        
        # Find all training result files
        result_files = glob.glob("marble_model/training_results_*.json")
        if len(result_files) < 2:
            print("Need at least 2 training runs for comparison")
            return
        
        # Load recent training runs (last 5)
        recent_files = sorted(result_files, key=lambda x: x.split('_')[-1])[-5:]
        runs_data = []
        
        for file_path in recent_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    timestamp = data['timestamp']
                    runs_data.append({
                        'timestamp': timestamp,
                        'dataset_size': data['dataset_size'],
                        'final_val_loss': data['final_metrics']['final_val_loss'],
                        'final_accuracy': data['final_metrics']['final_accuracy'],
                        'val_losses': data['training_history']['val_losses'],
                        'epochs': data['training_history']['epochs']
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not runs_data:
            return
        
        # Create comparison HTML
        comparison_html = self.create_runs_comparison_html(runs_data)
        comparison_file = "training_plots/training_runs_comparison.html"
        
        with open(comparison_file, 'w') as f:
            f.write(comparison_html)
        
        print(f"📊 Training runs comparison saved: {comparison_file}")
        
        # Try to open in browser
        try:
            import subprocess
            import platform
            
            abs_path = os.path.abspath(comparison_file)
            if platform.system().lower() == "darwin":  # macOS
                subprocess.run(["open", abs_path], check=False)
                print(f"🌐 Opened training runs comparison in browser")
        except Exception as e:
            print(f"⚠️  Please manually open: {comparison_file}")
    
    def create_runs_comparison_html(self, runs_data):
        """Create HTML for training runs comparison"""
        from datetime import datetime
        
        # Generate colors for different runs
        colors = ['#2196f3', '#ff9800', '#4caf50', '#f44336', '#9c27b0']
        
        traces_js = []
        for i, run in enumerate(runs_data):
            color = colors[i % len(colors)]
            trace = f"""{{
                x: {run['epochs']},
                y: {run['val_losses']},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Run {run['timestamp']} (Size: {run['dataset_size']})',
                line: {{color: '{color}', width: 3}},
                marker: {{size: 6}}
            }}"""
            traces_js.append(trace)
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Marble Language Training Runs Comparison</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .plot-container {{ margin: 20px 0; }}
        .runs-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .runs-table th, .runs-table td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        .runs-table th {{ background-color: #e3f2fd; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 Training Runs Comparison</h1>
            <p>Comparing {len(runs_data)} recent training runs</p>
        </div>
        
        <table class="runs-table">
            <thead>
                <tr>
                    <th>Run Timestamp</th>
                    <th>Dataset Size</th>
                    <th>Final Val Loss</th>
                    <th>Final Accuracy</th>
                </tr>
            </thead>
            <tbody>
                {''.join([f'''<tr>
                    <td>{run['timestamp']}</td>
                    <td>{run['dataset_size']}</td>
                    <td>{run['final_val_loss']:.4f}</td>
                    <td>{run['final_accuracy']:.3f}</td>
                </tr>''' for run in runs_data])}
            </tbody>
        </table>
        
        <div class="plot-container">
            <h3>Validation Loss Comparison</h3>
            <div id="comparisonPlot" style="width:100%; height:500px;"></div>
        </div>
    </div>
    
    <script>
        var traces = [
            {','.join(traces_js)}
        ];
        
        Plotly.newPlot('comparisonPlot', traces, {{
            title: 'Validation Loss Across Training Runs',
            xaxis: {{title: 'Epoch'}},
            yaxis: {{title: 'Validation Loss'}},
            showlegend: true,
            legend: {{x: 0.7, y: 0.95}}
        }});
    </script>
</body>
</html>"""
    
    def run_training(self, epochs: int = 5):
        """Run the complete training pipeline"""
        
        print("🎯 Marble Language Pretraining")
        print("=" * 40)
        
        # Generate training dataset
        print(f"📝 Generating {self.dataset_size} training sentences...")
        sentences = self.generator.generate_sentences(self.dataset_size)
        
        # Create vocabulary
        print("📚 Building vocabulary...")
        vocab = self.create_vocabulary(sentences)
        
        # Tokenize data
        print("🔢 Tokenizing sentences...")
        train_data = self.tokenize_sentences(sentences, vocab)
        
        # Simulate training
        history = self.simulate_training(train_data, vocab, epochs)
        
        # Save results
        results_file = self.save_training_results(history, vocab, train_data)
        
        # Create training runs comparison
        self.create_training_runs_comparison()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Check training_plots/ for visualizations")
        
        return results_file


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Marble Language Pretraining')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Number of training sentences')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = SimpleMarbleTrainer(dataset_size=args.dataset_size)
    trainer.run_training(epochs=args.epochs)


if __name__ == "__main__":
    main()