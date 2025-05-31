#!/usr/bin/env python3
"""
Training Run Historical Database
Tracks and stores statistics from all training runs for analysis and comparison
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TrainingRunDatabase:
    """Database for storing and querying training run statistics"""
    
    def __init__(self, db_path: str = "training_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        dataset_files TEXT NOT NULL,
                        num_sentences INTEGER,
                        vocab_size INTEGER,
                        model_parameters INTEGER,
                        
                        -- Training configuration
                        epochs INTEGER,
                        batch_size INTEGER,
                        learning_rate REAL,
                        device TEXT,
                        
                        -- Final results
                        best_epoch INTEGER,
                        best_train_loss REAL,
                        best_val_loss REAL,
                        best_val_accuracy REAL,
                        best_val_perplexity REAL,
                        final_test_loss REAL,
                        final_test_accuracy REAL,
                        final_test_perplexity REAL,
                        
                        -- Training metadata
                        total_training_time REAL,
                        early_stopped BOOLEAN,
                        completed BOOLEAN,
                        notes TEXT,
                        
                        -- Raw data paths
                        model_path TEXT,
                        results_path TEXT,
                        plots_path TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS epoch_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        epoch INTEGER NOT NULL,
                        train_loss REAL NOT NULL,
                        val_loss REAL,
                        val_accuracy REAL,
                        val_perplexity REAL,
                        learning_rate REAL,
                        epoch_time REAL,
                        FOREIGN KEY (run_id) REFERENCES training_runs (run_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS iteration_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        epoch INTEGER NOT NULL,
                        iteration INTEGER NOT NULL,
                        global_iteration INTEGER NOT NULL,
                        train_loss REAL NOT NULL,
                        batch_accuracy REAL,
                        learning_rate REAL,
                        FOREIGN KEY (run_id) REFERENCES training_runs (run_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS language_evolution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        change_description TEXT NOT NULL,
                        vocabulary_snapshot TEXT NOT NULL,
                        rules_snapshot TEXT NOT NULL,
                        version TEXT,
                        FOREIGN KEY (run_id) REFERENCES training_runs (run_id)
                    )
                """)
                
                # Create indexes for faster queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON epoch_stats (run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id_iter ON iteration_stats (run_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON training_runs (timestamp)")
                
                conn.commit()
                logger.info(f"Training database initialized at: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def start_training_run(self, config: Dict[str, Any]) -> str:
        """Start a new training run and return the run ID"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.get('model_name', 'unknown')}"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO training_runs (
                        run_id, timestamp, model_name, dataset_files, num_sentences,
                        vocab_size, model_parameters, epochs, batch_size, learning_rate,
                        device, completed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    config.get('model_name', 'MarbleTransformer'),
                    json.dumps(config.get('dataset_files', [])),
                    config.get('num_sentences', 0),
                    config.get('vocab_size', 0),
                    config.get('model_parameters', 0),
                    config.get('epochs', 0),
                    config.get('batch_size', 32),
                    config.get('learning_rate', 0.001),
                    config.get('device', 'cpu'),
                    False  # Not completed yet
                ))
                conn.commit()
                
            logger.info(f"Started training run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start training run: {e}")
            raise
    
    def log_epoch_stats(self, run_id: str, epoch: int, stats: Dict[str, float], epoch_time: float = None):
        """Log statistics for a completed epoch"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO epoch_stats (
                        run_id, epoch, train_loss, val_loss, val_accuracy,
                        val_perplexity, learning_rate, epoch_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    epoch,
                    stats.get('train_loss'),
                    stats.get('val_loss'),
                    stats.get('val_accuracy'),
                    stats.get('val_perplexity'),
                    stats.get('learning_rate'),
                    epoch_time
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log epoch stats: {e}")
    
    def log_iteration_stats(self, run_id: str, epoch: int, iteration: int, 
                           global_iteration: int, stats: Dict[str, float]):
        """Log statistics for a training iteration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO iteration_stats (
                        run_id, epoch, iteration, global_iteration, train_loss,
                        batch_accuracy, learning_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    epoch,
                    iteration,
                    global_iteration,
                    stats.get('train_loss'),
                    stats.get('batch_accuracy'),
                    stats.get('learning_rate')
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log iteration stats: {e}")
    
    def complete_training_run(self, run_id: str, final_results: Dict[str, Any], 
                             training_time: float, early_stopped: bool = False):
        """Mark a training run as completed with final results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE training_runs SET
                        best_epoch = ?, best_train_loss = ?, best_val_loss = ?,
                        best_val_accuracy = ?, best_val_perplexity = ?,
                        final_test_loss = ?, final_test_accuracy = ?, final_test_perplexity = ?,
                        total_training_time = ?, early_stopped = ?, completed = ?,
                        model_path = ?, results_path = ?, plots_path = ?, notes = ?
                    WHERE run_id = ?
                """, (
                    final_results.get('best_epoch'),
                    final_results.get('best_train_loss'),
                    final_results.get('best_val_loss'),
                    final_results.get('best_val_accuracy'),
                    final_results.get('best_val_perplexity'),
                    final_results.get('test_loss'),
                    final_results.get('test_accuracy'),
                    final_results.get('test_perplexity'),
                    training_time,
                    early_stopped,
                    True,
                    final_results.get('model_path'),
                    final_results.get('results_path'),
                    final_results.get('plots_path'),
                    final_results.get('notes', ''),
                    run_id
                ))
                conn.commit()
                
            logger.info(f"Completed training run: {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to complete training run: {e}")
    
    def get_training_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent training runs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM training_runs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                runs = []
                for row in cursor.fetchall():
                    run_dict = dict(row)
                    run_dict['dataset_files'] = json.loads(run_dict['dataset_files'] or '[]')
                    runs.append(run_dict)
                
                return runs
                
        except Exception as e:
            logger.error(f"Failed to get training runs: {e}")
            return []
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get run info
                cursor = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,))
                run = cursor.fetchone()
                
                if not run:
                    return None
                
                run_dict = dict(run)
                run_dict['dataset_files'] = json.loads(run_dict['dataset_files'] or '[]')
                
                # Get epoch stats
                cursor = conn.execute("""
                    SELECT * FROM epoch_stats 
                    WHERE run_id = ? 
                    ORDER BY epoch
                """, (run_id,))
                run_dict['epoch_stats'] = [dict(row) for row in cursor.fetchall()]
                
                # Get iteration stats summary
                cursor = conn.execute("""
                    SELECT COUNT(*) as total_iterations,
                           MIN(train_loss) as min_loss,
                           MAX(train_loss) as max_loss,
                           AVG(train_loss) as avg_loss
                    FROM iteration_stats 
                    WHERE run_id = ?
                """, (run_id,))
                run_dict['iteration_summary'] = dict(cursor.fetchone())
                
                return run_dict
                
        except Exception as e:
            logger.error(f"Failed to get run details: {e}")
            return None
    
    def get_best_runs(self, metric: str = 'best_val_accuracy', limit: int = 10) -> List[Dict[str, Any]]:
        """Get the best training runs by a specific metric"""
        valid_metrics = [
            'best_val_accuracy', 'best_val_loss', 'best_val_perplexity',
            'final_test_accuracy', 'final_test_loss', 'final_test_perplexity'
        ]
        
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of: {valid_metrics}")
        
        try:
            order = "DESC" if 'accuracy' in metric else "ASC"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(f"""
                    SELECT * FROM training_runs 
                    WHERE completed = 1 AND {metric} IS NOT NULL
                    ORDER BY {metric} {order}
                    LIMIT ?
                """, (limit,))
                
                runs = []
                for row in cursor.fetchall():
                    run_dict = dict(row)
                    run_dict['dataset_files'] = json.loads(run_dict['dataset_files'] or '[]')
                    runs.append(run_dict)
                
                return runs
                
        except Exception as e:
            logger.error(f"Failed to get best runs: {e}")
            return []
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get overall training statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                stats = {}
                
                # Basic counts
                cursor = conn.execute("SELECT COUNT(*) as total_runs FROM training_runs")
                stats['total_runs'] = cursor.fetchone()['total_runs']
                
                cursor = conn.execute("SELECT COUNT(*) as completed_runs FROM training_runs WHERE completed = 1")
                stats['completed_runs'] = cursor.fetchone()['completed_runs']
                
                cursor = conn.execute("SELECT COUNT(*) as early_stopped FROM training_runs WHERE early_stopped = 1")
                stats['early_stopped_runs'] = cursor.fetchone()['early_stopped']
                
                # Best results
                if stats['completed_runs'] > 0:
                    cursor = conn.execute("""
                        SELECT MAX(best_val_accuracy) as best_accuracy,
                               MIN(best_val_loss) as best_loss,
                               MIN(best_val_perplexity) as best_perplexity,
                               AVG(total_training_time) as avg_training_time
                        FROM training_runs 
                        WHERE completed = 1
                    """)
                    best_results = dict(cursor.fetchone())
                    stats.update(best_results)
                
                # Recent activity
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as runs_count
                    FROM training_runs 
                    WHERE timestamp > datetime('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 10
                """)
                stats['recent_activity'] = [dict(row) for row in cursor.fetchall()]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get training statistics: {e}")
            return {}
    
    def export_run_data(self, run_id: str, output_path: str):
        """Export all data for a specific run to JSON"""
        run_data = self.get_run_details(run_id)
        
        if not run_data:
            raise ValueError(f"Run {run_id} not found")
        
        # Get full iteration data
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM iteration_stats 
                    WHERE run_id = ? 
                    ORDER BY global_iteration
                """, (run_id,))
                run_data['iteration_stats'] = [dict(row) for row in cursor.fetchall()]
            
            with open(output_path, 'w') as f:
                json.dump(run_data, f, indent=2, default=str)
            
            logger.info(f"Exported run data to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export run data: {e}")
            raise


if __name__ == "__main__":
    # Test the database system
    db = TrainingRunDatabase("test_training_history.db")
    
    # Test starting a run
    config = {
        'model_name': 'TestModel',
        'dataset_files': ['test1.txt', 'test2.txt'],
        'num_sentences': 1000,
        'vocab_size': 25,
        'model_parameters': 17000,
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'cpu'
    }
    
    run_id = db.start_training_run(config)
    print(f"Started test run: {run_id}")
    
    # Test logging some stats
    for epoch in range(3):
        epoch_stats = {
            'train_loss': 2.0 - epoch * 0.3,
            'val_loss': 2.1 - epoch * 0.25,
            'val_accuracy': 0.3 + epoch * 0.2,
            'val_perplexity': 8.0 - epoch * 1.5,
            'learning_rate': 0.001 * (0.9 ** epoch)
        }
        db.log_epoch_stats(run_id, epoch, epoch_stats, epoch_time=60.0)
        
        # Log some iteration stats
        for iteration in range(5):
            iter_stats = {
                'train_loss': epoch_stats['train_loss'] + (iteration - 2) * 0.1,
                'batch_accuracy': epoch_stats['val_accuracy'] + (iteration - 2) * 0.05,
                'learning_rate': epoch_stats['learning_rate']
            }
            global_iter = epoch * 5 + iteration
            db.log_iteration_stats(run_id, epoch, iteration, global_iter, iter_stats)
    
    # Complete the run
    final_results = {
        'best_epoch': 2,
        'best_train_loss': 1.4,
        'best_val_loss': 1.45,
        'best_val_accuracy': 0.7,
        'best_val_perplexity': 5.0,
        'test_loss': 1.5,
        'test_accuracy': 0.68,
        'test_perplexity': 5.2,
        'model_path': './test_model.pt',
        'results_path': './test_results.json'
    }
    
    db.complete_training_run(run_id, final_results, training_time=180.0)
    
    # Test queries
    print("\nRecent runs:")
    runs = db.get_training_runs(limit=5)
    for run in runs:
        print(f"  {run['run_id']}: {run['model_name']} - Accuracy: {run['best_val_accuracy']}")
    
    print("\nTraining statistics:")
    stats = db.get_training_statistics()
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Completed: {stats['completed_runs']}")
    print(f"  Best accuracy: {stats.get('best_accuracy', 'N/A')}")
    
    # Clean up test database
    os.remove("test_training_history.db")
    print("\nTest completed successfully!")