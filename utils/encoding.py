"""
Sequence encoding utilities for RNA and protein sequences.
"""
import torch
import numpy as np


class SequenceEncoder:
    """Encoder for RNA and protein sequences."""
    
    # RNA nucleotides (including padding)
    RNA_VOCAB = {
        'PAD': 0,
        'A': 1,
        'U': 2,
        'G': 3,
        'C': 4,
        'N': 5  # Unknown nucleotide
    }
    
    # Protein amino acids (20 standard + padding)
    PROTEIN_VOCAB = {
        'PAD': 0,
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'X': 21  # Unknown amino acid
    }
    
    def __init__(self, max_rna_length=1000, max_protein_length=2000):
        """
        Initialize the sequence encoder.
        
        Args:
            max_rna_length (int): Maximum length for RNA sequences
            max_protein_length (int): Maximum length for protein sequences
        """
        self.max_rna_length = max_rna_length
        self.max_protein_length = max_protein_length
        
    def encode_rna(self, sequence):
        """
        Encode RNA sequence to integers.
        
        Args:
            sequence (str): RNA sequence
            
        Returns:
            np.ndarray: Encoded sequence
        """
        sequence = sequence.upper()
        encoded = []
        
        for nucleotide in sequence[:self.max_rna_length]:
            encoded.append(self.RNA_VOCAB.get(nucleotide, self.RNA_VOCAB['N']))
        
        # Pad sequence
        while len(encoded) < self.max_rna_length:
            encoded.append(self.RNA_VOCAB['PAD'])
            
        return np.array(encoded, dtype=np.int64)
    
    def encode_protein(self, sequence):
        """
        Encode protein sequence to integers.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Encoded sequence
        """
        sequence = sequence.upper()
        encoded = []
        
        for amino_acid in sequence[:self.max_protein_length]:
            encoded.append(self.PROTEIN_VOCAB.get(amino_acid, self.PROTEIN_VOCAB['X']))
        
        # Pad sequence
        while len(encoded) < self.max_protein_length:
            encoded.append(self.PROTEIN_VOCAB['PAD'])
            
        return np.array(encoded, dtype=np.int64)
    
    @staticmethod
    def get_rna_vocab_size():
        """Get RNA vocabulary size."""
        return len(SequenceEncoder.RNA_VOCAB)
    
    @staticmethod
    def get_protein_vocab_size():
        """Get protein vocabulary size."""
        return len(SequenceEncoder.PROTEIN_VOCAB)


def one_hot_encode_rna(sequence, max_length=1000):
    """
    One-hot encode RNA sequence.
    
    Args:
        sequence (str): RNA sequence
        max_length (int): Maximum sequence length
        
    Returns:
        np.ndarray: One-hot encoded sequence (max_length, 4)
    """
    encoding_dict = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    sequence = sequence.upper()[:max_length]
    
    one_hot = np.zeros((max_length, 4), dtype=np.float32)
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide in encoding_dict:
            one_hot[i, encoding_dict[nucleotide]] = 1.0
    
    return one_hot


def one_hot_encode_protein(sequence, max_length=2000):
    """
    One-hot encode protein sequence.
    
    Args:
        sequence (str): Protein sequence
        max_length (int): Maximum sequence length
        
    Returns:
        np.ndarray: One-hot encoded sequence (max_length, 20)
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    encoding_dict = {aa: i for i, aa in enumerate(amino_acids)}
    sequence = sequence.upper()[:max_length]
    
    one_hot = np.zeros((max_length, 20), dtype=np.float32)
    
    for i, aa in enumerate(sequence):
        if aa in encoding_dict:
            one_hot[i, encoding_dict[aa]] = 1.0
    
    return one_hot

