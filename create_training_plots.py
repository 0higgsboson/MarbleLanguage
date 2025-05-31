#!/usr/bin/env python3
"""
Create training plots for marble language training
This creates sample training plots in the training_plots directory
"""

import os
import sys
from test_training_plots import create_training_plots

def main():
    print("🎯 Creating marble language training plots...")
    print("=" * 50)
    
    try:
        create_training_plots()
        print("\n✅ Training plots successfully created!")
        print("📂 Check the training_plots/ directory for HTML and JSON files")
        print("🌐 Open the HTML files in your browser to view interactive plots")
        
        # List the most recent files
        print("\n📋 Recent training plot files:")
        os.system("ls -la training_plots/ | tail -5")
        
    except Exception as e:
        print(f"❌ Error creating training plots: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()