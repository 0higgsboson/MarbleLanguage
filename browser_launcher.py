#!/usr/bin/env python3
"""
Browser launcher utility for opening live training plots
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def open_safari_with_auto_refresh(html_file_path: str):
    """
    Open Safari browser with the HTML plot file and enable auto-refresh
    
    Args:
        html_file_path: Path to the HTML file to open
    """
    # Convert to absolute path
    abs_path = os.path.abspath(html_file_path)
    file_url = f"file://{abs_path}"
    
    try:
        # On macOS, use 'open' command to open Safari specifically
        subprocess.run([
            "open", 
            "-a", "Safari",
            file_url
        ], check=True)
        
        print(f"🌐 Opened Safari with live training plot: {file_url}")
        print("📊 Plot will auto-refresh every 3 seconds during training")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to open Safari: {e}")
        # Fallback to default browser
        try:
            subprocess.run(["open", file_url], check=True)
            print(f"🌐 Opened default browser with training plot: {file_url}")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to open browser. Please manually open: {file_url}")
            return False
    except FileNotFoundError:
        print("Safari not found. Trying default browser...")
        try:
            subprocess.run(["open", file_url], check=True)
            print(f"🌐 Opened default browser with training plot: {file_url}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Failed to open browser. Please manually open: {file_url}")
            return False


def create_initial_plot_file(output_dir: str = "."):
    """
    Create an initial HTML plot file that shows "Training Starting..." message
    
    Args:
        output_dir: Directory where to create the initial plot file
        
    Returns:
        Path to the created HTML file
    """
    plots_dir = os.path.join(output_dir, "training_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = os.path.join(plots_dir, f"training_loss_data_{timestamp}.html")
    
    initial_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss Plot - Live</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .status {{ background: #fff3cd; padding: 20px; border-radius: 5px; margin: 10px 0; border: 1px solid #ffeaa7; }}
        .live-indicator {{ color: #00aa00; font-weight: bold; }}
        .waiting {{ text-align: center; padding: 50px; }}
        .spinner {{ border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 50px; height: 50px; animation: spin 2s linear infinite; margin: 0 auto; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <h1>Marble Language Training Loss <span class="live-indicator">● LIVE</span></h1>
    <div class="status">
        🔄 Auto-refreshing every 3 seconds | Last updated: {datetime.now().strftime('%H:%M:%S')}
    </div>
    
    <div class="waiting">
        <div class="spinner"></div>
        <h2>🚀 Training Starting...</h2>
        <p>Waiting for training data. This page will automatically update with loss plots once training begins.</p>
        <p><strong>Keep this tab open</strong> to see live training progress!</p>
    </div>
    
    <h2>Training Configuration</h2>
    <ul>
        <li>Auto-refresh: ✅ Every 3 seconds</li>
        <li>Live plotting: ✅ Enabled</li>
        <li>Browser: 🦁 Safari</li>
        <li>Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
    </ul>
</body>
</html>
"""
    
    with open(html_file, 'w') as f:
        f.write(initial_html)
    
    return html_file


def setup_live_training_view(output_dir: str = "."):
    """
    Set up live training view by creating initial plot and opening browser
    
    Args:
        output_dir: Directory for training plots
        
    Returns:
        Path to the HTML file that will be updated during training
    """
    print("🔧 Setting up live training visualization...")
    
    # Create initial plot file
    html_file = create_initial_plot_file(output_dir)
    print(f"📄 Created initial plot file: {html_file}")
    
    # Open in Safari
    success = open_safari_with_auto_refresh(html_file)
    
    if success:
        print("✅ Live training view ready!")
        print("💡 The plot will update automatically as training progresses")
    else:
        print("⚠️ Browser opening failed, but training will continue")
        print(f"🔗 Manual link: file://{os.path.abspath(html_file)}")
    
    return html_file


if __name__ == "__main__":
    # Test the browser launcher
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
        if os.path.exists(html_file):
            open_safari_with_auto_refresh(html_file)
        else:
            print(f"File not found: {html_file}")
    else:
        # Demo mode
        html_file = setup_live_training_view(".")
        print(f"Demo HTML file created: {html_file}")