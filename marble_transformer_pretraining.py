#!/usr/bin/env python3
"""
Marble Language Transformer Pre-training Program
Trains a transformer model on marble language datasets from text files
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import logging

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Install with 'pip3 install matplotlib' for plotting functionality.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarbleLanguageDataset(Dataset):
    """Dataset class for marble language sentences"""
    
    def __init__(self, sentences: List[str], vocab: Dict[str, int], max_length: int = 16):
        self.sentences = sentences
        self.vocab = vocab
        self.max_length = max_length
        self.tokenized_sentences = self._tokenize_sentences()
    
    def _tokenize_sentences(self) -> List[List[int]]:
        """Tokenize all sentences using the vocabulary"""
        tokenized = []
        for sentence in self.sentences:
            tokens = sentence.strip().split()
            # Add BOS token at start, EOS token at end
            token_ids = [self.vocab['[BOS]']]
            
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    logger.warning(f"Unknown token '{token}' in sentence: {sentence}")
                    token_ids.append(self.vocab['[UNK]'])
            
            token_ids.append(self.vocab['[EOS]'])
            
            # Pad or truncate to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids.extend([self.vocab['[PAD]']] * (self.max_length - len(token_ids)))
            
            tokenized.append(token_ids)
        
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_sentences)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_sentences[idx]
        # For causal LM, input is tokens[:-1], target is tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.tensor([1 if token != self.vocab['[PAD]'] else 0 for token in tokens[:-1]], dtype=torch.bool)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask
        }


class MarbleTransformer(nn.Module):
    """Simple transformer model for marble language"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 2, 
                 num_layers: int = 2, ff_dim: int = 128, max_length: int = 16):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        if x.device.type == 'cuda':
            causal_mask = causal_mask.cuda()
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to the format expected by transformer
            attention_mask = ~attention_mask  # Invert: True for positions to mask
        
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=attention_mask)
        
        # Apply final layer norm and head
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


class MarbleLanguageTrainer:
    """Trainer class for the marble language transformer"""
    
    def __init__(self, model: MarbleTransformer, vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Training components
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def train_epoch(self, dataloader: DataLoader, device: torch.device, epoch: int = 0) -> float:
        """Train for one epoch with detailed iteration logging"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        logger.info(f"Starting epoch {epoch + 1} with {num_batches} iterations (batches)")
        
        for iteration, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log every 10 iterations or at the end
            if (iteration + 1) % 10 == 0 or iteration == num_batches - 1:
                avg_loss_so_far = total_loss / (iteration + 1)
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"  Iteration {iteration + 1}/{num_batches} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Avg Loss: {avg_loss_so_far:.4f} | "
                          f"LR: {lr:.6f}")
                
                # Show a sample prediction every 50 iterations
                if (iteration + 1) % 50 == 0:
                    self._log_sample_prediction(input_ids[0], target_ids[0], logits[0])
        
        return total_loss / num_batches
    
    def _log_sample_prediction(self, input_ids: torch.Tensor, target_ids: torch.Tensor, logits: torch.Tensor):
        """Log a sample prediction vs target for monitoring"""
        with torch.no_grad():
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Convert to tokens (skip padding)
            input_tokens = [self.reverse_vocab[id.item()] for id in input_ids if id.item() != self.vocab['[PAD]']]
            target_tokens = [self.reverse_vocab[id.item()] for id in target_ids if id.item() != self.vocab['[PAD]']]
            predicted_tokens = [self.reverse_vocab[id.item()] for id in predicted_ids[:len(target_tokens)]]
            
            logger.info(f"    Sample - Input: {' '.join(input_tokens)}")
            logger.info(f"    Sample - Target: {' '.join(target_tokens)}")
            logger.info(f"    Sample - Predicted: {' '.join(predicted_tokens)}")
            
            # Calculate token-level accuracy for this sample
            correct = sum(1 for t, p in zip(target_tokens, predicted_tokens) if t == p)
            accuracy = correct / len(target_tokens) if target_tokens else 0
            logger.info(f"    Sample - Accuracy: {accuracy:.2f} ({correct}/{len(target_tokens)})")
    
    def evaluate(self, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = target_ids != self.vocab['[PAD]']
                correct = (predictions == target_ids) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, accuracy, perplexity
    
    def generate_sentence(self, device: torch.device, max_length: int = 10) -> str:
        """Generate a single sentence"""
        self.model.eval()
        
        with torch.no_grad():
            # Start with BOS token
            generated = [self.vocab['[BOS]']]
            
            for _ in range(max_length):
                # Prepare input
                input_ids = torch.tensor([generated], dtype=torch.long, device=device)
                
                # Get logits
                logits = self.model(input_ids)
                
                # Sample next token
                next_token_logits = logits[0, -1, :]
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1).item()
                
                generated.append(next_token)
                
                # Stop if EOS token is generated
                if next_token == self.vocab['[EOS]']:
                    break
            
            # Convert to sentence
            tokens = [self.reverse_vocab[token_id] for token_id in generated[1:-1]]  # Skip BOS and EOS
            return ' '.join(tokens)


def parse_marble_files(file_paths: List[str]) -> List[str]:
    """Parse marble language files and extract sentences"""
    all_sentences = []
    
    for file_path in file_paths:
        logger.info(f"Parsing file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract sentences from numbered format (1. "sentence")
        numbered_pattern = r'\d+\.\s*"([^"]+)"'
        numbered_sentences = re.findall(numbered_pattern, content)
        
        # Also try to extract from raw array format
        array_pattern = r"\['([^']+)'"
        array_sentences = re.findall(array_pattern, content)
        
        sentences = numbered_sentences + array_sentences
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        all_sentences.extend(unique_sentences)
        logger.info(f"Extracted {len(unique_sentences)} sentences from {file_path}")
    
    logger.info(f"Total sentences: {len(all_sentences)}")
    return all_sentences


def create_vocabulary() -> Dict[str, int]:
    """Create vocabulary for marble language"""
    # Core marble language tokens
    marble_tokens = ['I', 'static', 'move', 'East', 'West', 'North', 'South', 'bump', 'then']
    
    # Special tokens
    special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
    
    vocab = {}
    for i, token in enumerate(special_tokens + marble_tokens):
        vocab[token] = i
    
    return vocab


class TrainingPlotter:
    """Real-time training progress plotter"""
    
    def __init__(self, output_dir: str, model_name: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.plot_available = PLOTTING_AVAILABLE
        
        if self.plot_available:
            # Set up matplotlib style
            style.use('default')
            plt.ion()  # Interactive mode for real-time plotting
            
            # Create figure with subplots
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle(f'Training Progress: {model_name}', fontsize=14, fontweight='bold')
            
            # Initialize empty lists for data
            self.epochs = []
            self.train_losses = []
            self.val_losses = []
            self.val_accuracies = []
            self.val_perplexities = []
            self.learning_rates = []
            
            # Set up subplot titles and labels
            self.axes[0, 0].set_title('Training & Validation Loss')
            self.axes[0, 0].set_xlabel('Epoch')
            self.axes[0, 0].set_ylabel('Loss')
            self.axes[0, 0].grid(True, alpha=0.3)
            
            self.axes[0, 1].set_title('Validation Accuracy')
            self.axes[0, 1].set_xlabel('Epoch')
            self.axes[0, 1].set_ylabel('Accuracy')
            self.axes[0, 1].grid(True, alpha=0.3)
            
            self.axes[1, 0].set_title('Validation Perplexity')
            self.axes[1, 0].set_xlabel('Epoch')
            self.axes[1, 0].set_ylabel('Perplexity')
            self.axes[1, 0].grid(True, alpha=0.3)
            
            self.axes[1, 1].set_title('Learning Rate Schedule')
            self.axes[1, 1].set_xlabel('Epoch')
            self.axes[1, 1].set_ylabel('Learning Rate')
            self.axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
    
    def update_plot(self, epoch_data: Dict):
        """Update plots with new epoch data"""
        if not self.plot_available:
            return
        
        # Add new data
        self.epochs.append(epoch_data['epoch'])
        self.train_losses.append(epoch_data['train_loss'])
        self.val_losses.append(epoch_data['val_loss'])
        self.val_accuracies.append(epoch_data['val_accuracy'])
        self.val_perplexities.append(epoch_data['val_perplexity'])
        self.learning_rates.append(epoch_data['learning_rate'])
        
        # Clear and update plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Training & Validation Loss
        self.axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        self.axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.axes[0, 0].set_title('Training & Validation Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy
        self.axes[0, 1].plot(self.epochs, self.val_accuracies, 'g-', linewidth=2)
        self.axes[0, 1].set_title('Validation Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].set_ylim(0, 1)
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Validation Perplexity
        self.axes[1, 0].plot(self.epochs, self.val_perplexities, 'purple', linewidth=2)
        self.axes[1, 0].set_title('Validation Perplexity')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Perplexity')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        self.axes[1, 1].plot(self.epochs, self.learning_rates, 'orange', linewidth=2)
        self.axes[1, 1].set_title('Learning Rate Schedule')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('Learning Rate')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Add current values as text
        current_epoch = self.epochs[-1]
        current_train_loss = self.train_losses[-1]
        current_val_loss = self.val_losses[-1]
        current_accuracy = self.val_accuracies[-1]
        current_perplexity = self.val_perplexities[-1]
        
        self.fig.suptitle(f'Training Progress: {self.model_name}\n'
                         f'Epoch {current_epoch} | Train Loss: {current_train_loss:.3f} | '
                         f'Val Loss: {current_val_loss:.3f} | Accuracy: {current_accuracy:.3f}', 
                         fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.pause(0.1)  # Brief pause to update display
    
    def save_final_plot(self):
        """Save final training plots"""
        if not self.plot_available or not self.epochs:
            return
        
        try:
            # Save the current figure
            plot_filename = os.path.join(self.output_dir, f'{self.model_name}_training_plots.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to: {plot_filename}")
            
            # Create a summary plot
            self.create_summary_plot()
            
        except Exception as e:
            logger.warning(f"Could not save plots: {e}")
    
    def create_summary_plot(self):
        """Create a clean summary plot for final results"""
        if not self.plot_available:
            return
        
        # Create a new figure for summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Final Training Summary: {self.model_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy progression
        axes[0, 1].plot(self.epochs, self.val_accuracies, 'g-', linewidth=2)
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Perplexity
        axes[1, 0].plot(self.epochs, self.val_perplexities, 'purple', linewidth=2)
        axes[1, 0].set_title('Validation Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training statistics summary
        axes[1, 1].text(0.1, 0.8, f'Final Results:', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f'Best Val Loss: {min(self.val_losses):.4f}', fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Best Accuracy: {max(self.val_accuracies):.4f}', fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f'Best Perplexity: {min(self.val_perplexities):.4f}', fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Total Epochs: {len(self.epochs)}', fontsize=10, transform=axes[1, 1].transAxes)
        
        # Find best epoch
        best_epoch = self.epochs[np.argmin(self.val_losses)]
        axes[1, 1].text(0.1, 0.3, f'Best Epoch: {best_epoch}', fontsize=10, transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_filename = os.path.join(self.output_dir, f'{self.model_name}_summary_plot.png')
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Summary plot saved to: {summary_filename}")
        
        plt.close(fig)  # Close the summary figure
    
    def close(self):
        """Close the plotting interface"""
        if self.plot_available:
            plt.ioff()  # Turn off interactive mode
            plt.close('all')
    """Split data into train/validation/test sets"""
    n = len(sentences)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Shuffle sentences
    import random
    random.shuffle(sentences)
    
    train_sentences = sentences[:train_end]
    val_sentences = sentences[train_end:val_end]
    test_sentences = sentences[val_end:]
    
    return train_sentences, val_sentences, test_sentences


def main():
    parser = argparse.ArgumentParser(description='Train marble language transformer')
    parser.add_argument('data_files', nargs='+', help='Paths to marble language data files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./marble_model', help='Output directory for model')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Parse data files
    sentences = parse_marble_files(args.data_files)
    
    if len(sentences) < 100:
        logger.warning(f"Only {len(sentences)} sentences found. Recommend at least 1000 for good training.")
    
    # Create vocabulary
    vocab = create_vocabulary()
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Split data
    train_sentences, val_sentences, test_sentences = split_data(sentences)
    logger.info(f"Data split - Train: {len(train_sentences)}, Val: {len(val_sentences)}, Test: {len(test_sentences)}")
    
    # Create datasets
    train_dataset = MarbleLanguageDataset(train_sentences, vocab)
    val_dataset = MarbleLanguageDataset(val_sentences, vocab)
    test_dataset = MarbleLanguageDataset(test_sentences, vocab)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = MarbleTransformer(vocab_size=len(vocab)).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = MarbleLanguageTrainer(model, vocab)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate total iterations
    total_iterations = len(train_loader) * args.epochs
    logger.info(f"Training Setup:")
    logger.info(f"  Total epochs: {args.epochs}")
    logger.info(f"  Iterations per epoch: {len(train_loader)}")
    logger.info(f"  Total iterations: {total_iterations}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sentences per iteration: {args.batch_size}")
    logger.info(f"  Total sentence presentations: {total_iterations * args.batch_size}")
    logger.info("="*60)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEPOCH {epoch+1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = trainer.train_epoch(train_loader, device, epoch)
        
        # Evaluate
        val_loss, val_accuracy, val_perplexity = trainer.evaluate(val_loader, device)
        
        # Update learning rate
        trainer.scheduler.step()
        
        # Calculate progress
        completed_iterations = (epoch + 1) * len(train_loader)
        progress_percent = (completed_iterations / total_iterations) * 100
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Progress: {completed_iterations}/{total_iterations} iterations ({progress_percent:.1f}%)")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Perplexity: {val_perplexity:.4f}")
        logger.info(f"  Learning Rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Generate sample sentence
        sample_sentence = trainer.generate_sentence(device)
        logger.info(f"  Generated Sample: '{sample_sentence}'")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_perplexity': val_perplexity,
                'epoch': epoch,
                'completed_iterations': completed_iterations
            }, os.path.join(args.output_dir, 'best_model.pt'))
            
            logger.info("  ✓ New best model saved!")
            
        else:
            patience_counter += 1
            logger.info(f"  Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        logger.info("="*60)
    
    # Final evaluation on test set
    test_loss, test_accuracy, test_perplexity = trainer.evaluate(test_loader, device)
    logger.info(f"Final Test Results:")
    logger.info(f"  Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Perplexity: {test_perplexity:.4f}")
    
    # Generate some sample sentences
    logger.info("Sample generated sentences:")
    for i in range(5):
        sentence = trainer.generate_sentence(device)
        logger.info(f"  {i+1}. {sentence}")
    
    # Save final results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_perplexity': test_perplexity,
        'vocab_size': len(vocab),
        'num_sentences': len(sentences),
        'training_time': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()