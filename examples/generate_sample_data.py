"""
Generate sample data for testing the RNA-Protein interaction model.
This creates synthetic data for demonstration purposes only.
In practice, you should use real experimental data.
"""
import argparse
import random
import pandas as pd
import os


def generate_rna_sequence(min_length=50, max_length=500):
    """Generate a random RNA sequence."""
    nucleotides = ['A', 'U', 'G', 'C']
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(nucleotides, k=length))


def generate_protein_sequence(min_length=100, max_length=1000):
    """Generate a random protein sequence."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(amino_acids, k=length))


def generate_sample_dataset(num_samples=1000, positive_ratio=0.5):
    """
    Generate a sample dataset.
    
    Args:
        num_samples (int): Number of samples to generate
        positive_ratio (float): Ratio of positive samples (interactions)
        
    Returns:
        pd.DataFrame: Generated dataset
    """
    data = []
    num_positive = int(num_samples * positive_ratio)
    
    print(f"Generating {num_samples} samples...")
    print(f"Positive samples: {num_positive}")
    print(f"Negative samples: {num_samples - num_positive}")
    
    for i in range(num_samples):
        rna_seq = generate_rna_sequence()
        protein_seq = generate_protein_sequence()
        
        # Assign label
        if i < num_positive:
            label = 1
        else:
            label = 0
        
        data.append({
            'rna_sequence': rna_seq,
            'protein_sequence': protein_seq,
            'label': label
        })
    
    # Shuffle data
    random.shuffle(data)
    
    return pd.DataFrame(data)


def main(args):
    """Main function to generate sample data."""
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    df = generate_sample_dataset(
        num_samples=args.num_samples,
        positive_ratio=args.positive_ratio
    )
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, args.output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nSample data saved to {output_path}")
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Positive samples (label=1): {sum(df['label'] == 1)}")
    print(f"Negative samples (label=0): {sum(df['label'] == 0)}")
    print(f"Average RNA length: {df['rna_sequence'].apply(len).mean():.1f}")
    print(f"Average protein length: {df['protein_sequence'].apply(len).mean():.1f}")
    
    # Show first few samples
    print("\nFirst 3 samples:")
    print(df.head(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sample RNA-Protein interaction data')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--positive_ratio', type=float, default=0.5,
                       help='Ratio of positive samples (interactions)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--output_file', type=str, default='sample_data.csv',
                       help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)

