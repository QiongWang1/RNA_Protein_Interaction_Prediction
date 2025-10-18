"""
Deep learning model for RNA-Protein interaction prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence data."""
    
    def __init__(self, hidden_dim):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim (int): Hidden dimension size
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (batch, seq_len, hidden_dim)
            
        Returns:
            tuple: (attended_output, attention_weights)
        """
        # Calculate attention scores
        attention_scores = self.attention(x)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention
        attended = torch.sum(x * attention_weights, dim=1)  # (batch, hidden_dim)
        
        return attended, attention_weights


class RNAProteinInteractionModel(nn.Module):
    """
    Deep learning model for predicting RNA-Protein interactions.
    
    Architecture:
    1. Embedding layers for RNA and protein sequences
    2. Convolutional layers for local pattern extraction
    3. Bidirectional LSTM for sequential dependencies
    4. Attention mechanism for important region focus
    5. Fully connected layers for classification
    """
    
    def __init__(self, rna_vocab_size=6, protein_vocab_size=22,
                 rna_embedding_dim=128, protein_embedding_dim=128,
                 hidden_dim=256, num_cnn_filters=128, 
                 kernel_size=3, lstm_layers=2, dropout=0.3):
        """
        Initialize the model.
        
        Args:
            rna_vocab_size (int): Size of RNA vocabulary
            protein_vocab_size (int): Size of protein vocabulary
            rna_embedding_dim (int): RNA embedding dimension
            protein_embedding_dim (int): Protein embedding dimension
            hidden_dim (int): Hidden dimension for LSTM
            num_cnn_filters (int): Number of CNN filters
            kernel_size (int): Kernel size for CNN
            lstm_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(RNAProteinInteractionModel, self).__init__()
        
        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size, rna_embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.rna_conv = nn.Sequential(
            nn.Conv1d(rna_embedding_dim, num_cnn_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.Dropout(dropout)
        )
        
        self.protein_conv = nn.Sequential(
            nn.Conv1d(protein_embedding_dim, num_cnn_filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_cnn_filters),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.rna_lstm = nn.LSTM(
            num_cnn_filters, hidden_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.protein_lstm = nn.LSTM(
            num_cnn_filters, hidden_dim, lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention layers
        self.rna_attention = AttentionLayer(hidden_dim * 2)
        self.protein_attention = AttentionLayer(hidden_dim * 2)
        
        # Fully connected layers
        combined_dim = hidden_dim * 4  # 2 * (hidden_dim * 2) for bidirectional LSTM
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
        
        # Apply convolution (need to transpose for Conv1d)
        rna_conv = self.rna_conv(rna_embedded.transpose(1, 2)).transpose(1, 2)
        protein_conv = self.protein_conv(protein_embedded.transpose(1, 2)).transpose(1, 2)
        
        # Apply LSTM
        rna_lstm_out, _ = self.rna_lstm(rna_conv)
        protein_lstm_out, _ = self.protein_lstm(protein_conv)
        
        # Apply attention
        rna_attended, _ = self.rna_attention(rna_lstm_out)
        protein_attended, _ = self.protein_attention(protein_lstm_out)
        
        # Concatenate RNA and protein features
        combined = torch.cat([rna_attended, protein_attended], dim=1)
        
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


class SimplifiedRNAProteinModel(nn.Module):
    """
    A simplified model for faster training and testing.
    """
    
    def __init__(self, rna_vocab_size=6, protein_vocab_size=22,
                 embedding_dim=64, hidden_dim=128, dropout=0.3):
        """
        Initialize the simplified model.
        
        Args:
            rna_vocab_size (int): Size of RNA vocabulary
            protein_vocab_size (int): Size of protein vocabulary
            embedding_dim (int): Embedding dimension
            hidden_dim (int): Hidden dimension
            dropout (float): Dropout rate
        """
        super(SimplifiedRNAProteinModel, self).__init__()
        
        # Embedding layers
        self.rna_embedding = nn.Embedding(rna_vocab_size, embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.rna_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.protein_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        combined_dim = hidden_dim * 4
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
        
        # Apply LSTM and take last hidden state
        _, (rna_hidden, _) = self.rna_lstm(rna_embedded)
        _, (protein_hidden, _) = self.protein_lstm(protein_embedded)
        
        # Concatenate forward and backward hidden states
        rna_features = torch.cat([rna_hidden[0], rna_hidden[1]], dim=1)
        protein_features = torch.cat([protein_hidden[0], protein_hidden[1]], dim=1)
        
        # Combine features
        combined = torch.cat([rna_features, protein_features], dim=1)
        
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

