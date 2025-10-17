"""
Inference script for RNA-Protein interaction prediction.
"""
import argparse
import torch
import numpy as np

from models.rna_protein_model import RNAProteinInteractionModel, SimplifiedRNAProteinModel
from models.transformer_model import TransformerRNAProteinModel, SimplifiedTransformerModel
from utils.encoding import SequenceEncoder


def predict_interaction(model, rna_seq, protein_seq, encoder, device):
    """
    Predict interaction between RNA and protein sequences.
    
    Args:
        model: Trained model
        rna_seq (str): RNA sequence
        protein_seq (str): Protein sequence
        encoder: SequenceEncoder instance
        device: Device to use
        
    Returns:
        tuple: (prediction, probability)
    """
    model.eval()
    
    # Encode sequences
    rna_encoded = encoder.encode_rna(rna_seq)
    protein_encoded = encoder.encode_protein(protein_seq)
    
    # Convert to tensors and add batch dimension
    rna_tensor = torch.tensor(rna_encoded, dtype=torch.long).unsqueeze(0).to(device)
    protein_tensor = torch.tensor(protein_encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(rna_tensor, protein_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability


def main(args):
    """Main inference function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoder
    encoder = SequenceEncoder(
        max_rna_length=args.max_rna_length,
        max_protein_length=args.max_protein_length
    )
    
    # Initialize model
    if args.model_type == 'full':
        model = RNAProteinInteractionModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            rna_embedding_dim=args.embedding_dim,
            protein_embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
    elif args.model_type == 'transformer':
        model = TransformerRNAProteinModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    elif args.model_type == 'transformer_simple':
        model = SimplifiedTransformerModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    else:  # simplified (LSTM)
        model = SimplifiedRNAProteinModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully\n")
    
    # Predict
    if args.rna_seq and args.protein_seq:
        # Single prediction
        print("RNA Sequence:")
        print(args.rna_seq[:100] + "..." if len(args.rna_seq) > 100 else args.rna_seq)
        print(f"\nProtein Sequence:")
        print(args.protein_seq[:100] + "..." if len(args.protein_seq) > 100 else args.protein_seq)
        print("\nPredicting...")
        
        prediction, probability = predict_interaction(
            model, args.rna_seq, args.protein_seq, encoder, device
        )
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Interaction Probability: {probability:.4f}")
        print(f"Prediction: {'INTERACTION' if prediction == 1 else 'NO INTERACTION'}")
        print("="*50)
        
    elif args.input_file:
        # Batch prediction from file
        import pandas as pd
        
        print(f"Reading sequences from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        
        if 'rna_sequence' not in df.columns or 'protein_sequence' not in df.columns:
            raise ValueError("Input file must contain 'rna_sequence' and 'protein_sequence' columns")
        
        predictions = []
        probabilities = []
        
        print("Predicting...")
        for idx, row in df.iterrows():
            prediction, probability = predict_interaction(
                model, row['rna_sequence'], row['protein_sequence'], encoder, device
            )
            predictions.append(prediction)
            probabilities.append(probability)
        
        df['prediction'] = predictions
        df['probability'] = probabilities
        
        # Save results
        output_file = args.output_file or args.input_file.replace('.csv', '_predictions.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\nPredictions saved to {output_file}")
        print(f"\nSummary:")
        print(f"Total samples: {len(df)}")
        print(f"Predicted interactions: {sum(predictions)}")
        print(f"Predicted no interactions: {len(predictions) - sum(predictions)}")
        print(f"Average probability: {np.mean(probabilities):.4f}")
    
    else:
        print("Error: Please provide either --rna_seq and --protein_seq, or --input_file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict RNA-Protein interactions')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--rna_seq', type=str, default=None,
                       help='RNA sequence')
    parser.add_argument('--protein_seq', type=str, default=None,
                       help='Protein sequence')
    parser.add_argument('--input_file', type=str, default=None,
                       help='CSV file with rna_sequence and protein_sequence columns')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for predictions (only used with --input_file)')
    parser.add_argument('--model_type', type=str, default='simplified',
                       choices=['full', 'simplified', 'transformer', 'transformer_simple'],
                       help='Model architecture type')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--max_rna_length', type=int, default=1000,
                       help='Maximum RNA sequence length')
    parser.add_argument('--max_protein_length', type=int, default=2000,
                       help='Maximum protein sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Transformer-specific parameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    
    args = parser.parse_args()
    main(args)

