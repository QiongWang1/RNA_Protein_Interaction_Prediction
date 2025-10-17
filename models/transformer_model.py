"""
Transformer-based models for RNA-Protein interaction prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input tensor (seq_len, batch, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding
        """
        return x + self.pe[:x.size(0), :]


class TransformerRNAProteinModel(nn.Module):
    """
    Transformer-based model for RNA-Protein interaction prediction.
    
    Architecture:
    1. Embedding layers for RNA and protein sequences
    2. Positional encoding
    3. Transformer encoder layers
    4. Global pooling (mean/max)
    5. Fully connected layers for classification
    """
    
    def __init__(self, rna_vocab_size=6, protein_vocab_size=22,
                 embedding_dim=128, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        """
        Initialize the Transformer model.
        
        Args:
            rna_vocab_size (int): Size of RNA vocabulary
            protein_vocab_size (int): Size of protein vocabulary
            embedding_dim (int): Embedding dimension
            d_model (int): Transformer model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
        """
        super(TransformerRNAProteinModel, self).__init__()
        
        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim, padding_idx=0)
        
        # Project embeddings to model dimension
        self.rna_projection = nn.Linear(embedding_dim, d_model)
        self.protein_projection = nn.Linear(embedding_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        combined_dim = d_model * 2  # RNA + Protein features
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, rna_seq, protein_seq):
        """
        Forward pass.
        
        Args:
            rna_seq (torch.Tensor): RNA sequence tensor (batch, seq_len)
            protein_seq (torch.Tensor): Protein sequence tensor (batch, seq_len)
            
        Returns:
            torch.Tensor: Interaction predictions (batch, 1)
        """
        # Embed sequences
        rna_embedded = self.rna_embedding(rna_seq)  # (batch, seq_len, embedding_dim)
        protein_embedded = self.protein_embedding(protein_seq)
        
        # Project to model dimension
        rna_features = self.rna_projection(rna_embedded)  # (batch, seq_len, d_model)
        protein_features = self.protein_projection(protein_embedded)
        
        # Add positional encoding
        rna_features = rna_features.transpose(0, 1)  # (seq_len, batch, d_model)
        protein_features = protein_features.transpose(0, 1)
        
        rna_features = self.pos_encoding(rna_features)
        protein_features = self.pos_encoding(protein_features)
        
        # Transpose back for batch_first=True
        rna_features = rna_features.transpose(0, 1)  # (batch, seq_len, d_model)
        protein_features = protein_features.transpose(0, 1)
        
        # Apply Transformer encoder
        rna_transformed = self.transformer_encoder(rna_features)
        protein_transformed = self.transformer_encoder(protein_features)
        
        # Global pooling (mean pooling)
        rna_pooled = torch.mean(rna_transformed, dim=1)  # (batch, d_model)
        protein_pooled = torch.mean(protein_transformed, dim=1)
        
        # Concatenate RNA and protein features
        combined = torch.cat([rna_pooled, protein_pooled], dim=1)
        
        # Final prediction
        output = self.fc(combined)
        
        return output
    
    def predict_proba(self, rna_seq, protein_seq):
        """
        Predict interaction probability.
        
        Args:
            rna_seq (torch.Tensor): RNA sequence tensor
            protein_seq (torch.Tensor): Protein sequence tensor
            
        Returns:
            torch.Tensor: Interaction probabilities
        """
        logits = self.forward(rna_seq, protein_seq)
        return torch.sigmoid(logits)


class SimplifiedTransformerModel(nn.Module):
    """
    A simplified Transformer model for faster training and testing.
    """
    
    def __init__(self, rna_vocab_size=6, protein_vocab_size=22,
                 embedding_dim=64, d_model=128, nhead=4, 
                 num_layers=3, dim_feedforward=512, dropout=0.1):
        """
        Initialize the simplified Transformer model.
        
        Args:
            rna_vocab_size (int): Size of RNA vocabulary
            protein_vocab_size (int): Size of protein vocabulary
            embedding_dim (int): Embedding dimension
            d_model (int): Transformer model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
        """
        super(SimplifiedTransformerModel, self).__init__()
        
        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim, padding_idx=0)
        
        # Project embeddings to model dimension
        self.rna_projection = nn.Linear(embedding_dim, d_model)
        self.protein_projection = nn.Linear(embedding_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Simplified Transformer encoder (fewer layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers
        combined_dim = d_model * 2
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, rna_seq, protein_seq):
        """
        Forward pass.
        
        Args:
            rna_seq (torch.Tensor): RNA sequence tensor
            protein_seq (torch.Tensor): Protein sequence tensor
            
        Returns:
            torch.Tensor: Interaction predictions
        """
        # Embed sequences
        rna_embedded = self.rna_embedding(rna_seq)
        protein_embedded = self.protein_embedding(protein_seq)
        
        # Project to model dimension
        rna_features = self.rna_projection(rna_embedded)
        protein_features = self.protein_projection(protein_embedded)
        
        # Add positional encoding
        rna_features = rna_features.transpose(0, 1)
        protein_features = protein_features.transpose(0, 1)
        
        rna_features = self.pos_encoding(rna_features)
        protein_features = self.pos_encoding(protein_features)
        
        # Transpose back for batch_first=True
        rna_features = rna_features.transpose(0, 1)
        protein_features = protein_features.transpose(0, 1)
        
        # Apply Transformer encoder
        rna_transformed = self.transformer_encoder(rna_features)
        protein_transformed = self.transformer_encoder(protein_features)
        
        # Global pooling (mean pooling)
        rna_pooled = torch.mean(rna_transformed, dim=1)
        protein_pooled = torch.mean(protein_transformed, dim=1)
        
        # Combine features
        combined = torch.cat([rna_pooled, protein_pooled], dim=1)
        
        # Final prediction
        output = self.fc(combined)
        
        return output
    
    def predict_proba(self, rna_seq, protein_seq):
        """
        Predict interaction probability.
        
        Args:
            rna_seq (torch.Tensor): RNA sequence tensor
            protein_seq (torch.Tensor): Protein sequence tensor
            
        Returns:
            torch.Tensor: Interaction probabilities
        """
        logits = self.forward(rna_seq, protein_seq)
        return torch.sigmoid(logits)
