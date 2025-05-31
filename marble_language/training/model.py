#!/usr/bin/env python3
"""
Marble Language Transformer Model
Enhanced transformer architecture for the marble language with collision detection
"""

import torch
import torch.nn as nn
from typing import Optional


class MarbleTransformer(nn.Module):
    """Enhanced transformer model for marble language with collision awareness"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 2, 
                 num_layers: int = 2, ff_dim: int = 128, max_length: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.dropout = dropout
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # Use GELU for better performance
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights with appropriate scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings with dropout
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout_layer(token_embeds + position_embeds)
        
        # Create causal mask for autoregressive generation
        causal_mask = self._generate_square_subsequent_mask(seq_len, input_ids.device)
        
        # Apply transformer with proper masking
        if attention_mask is not None:
            # Convert boolean attention mask to float and invert
            # True = attend, False = mask
            src_key_padding_mask = ~attention_mask
        else:
            src_key_padding_mask = None
        
        x = self.transformer(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply final layer norm and head
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive modeling"""
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask
    
    def generate_sample(self, input_ids: torch.Tensor, max_new_tokens: int = 10, 
                       temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate a sample sequence using the model"""
        self.eval()
        
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Get logits for the current sequence
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we exceed max length
                if generated.size(1) >= self.max_length:
                    break
            
            return generated
    
    def get_model_size(self) -> int:
        """Get the total number of model parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MarbleTransformerConfig:
    """Configuration class for MarbleTransformer"""
    
    def __init__(self, 
                 vocab_size: int = 20,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 num_layers: int = 2,
                 ff_dim: int = 128,
                 max_length: int = 16,
                 dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_length = max_length
        self.dropout = dropout
    
    def create_model(self) -> MarbleTransformer:
        """Create a MarbleTransformer instance with this configuration"""
        return MarbleTransformer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            max_length=self.max_length,
            dropout=self.dropout
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_length': self.max_length,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MarbleTransformerConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)


if __name__ == "__main__":
    # Test the model
    config = MarbleTransformerConfig(vocab_size=25)
    model = config.create_model()
    
    print(f"Model created with {model.get_model_size():,} parameters")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    logits = model(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    sample_input = torch.randint(0, config.vocab_size, (1, 3))  # Start with 3 tokens
    generated = model.generate_sample(sample_input, max_new_tokens=5)
    print(f"Generated sequence: {generated}")