"""
Process RPI1807 dataset from RPITER repository into our CSV format.
"""
import pandas as pd
import os
from collections import defaultdict


def read_fasta(fasta_file):
    """
    Read FASTA file and return dictionary of sequences.
    
    Args:
        fasta_file (str): Path to FASTA file
        
    Returns:
        dict: Dictionary mapping sequence ID to sequence
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                # Start new sequence
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def process_rpi1807_dataset(data_dir, output_file):
    """
    Process RPI1807 dataset and save as CSV.
    
    Args:
        data_dir (str): Directory containing RPITER data
        output_file (str): Output CSV file path
    """
    print("Processing RPI1807 dataset...")
    
    # Read sequences
    print("Reading RNA sequences...")
    rna_file = os.path.join(data_dir, 'sequence', 'RPI1807_rna_seq.fa')
    rna_sequences = read_fasta(rna_file)
    print(f"  Loaded {len(rna_sequences)} RNA sequences")
    
    print("Reading protein sequences...")
    protein_file = os.path.join(data_dir, 'sequence', 'RPI1807_protein_seq.fa')
    protein_sequences = read_fasta(protein_file)
    print(f"  Loaded {len(protein_sequences)} protein sequences")
    
    # Read interaction pairs
    print("Reading interaction pairs...")
    pairs_file = os.path.join(data_dir, 'RPI1807_pairs.txt')
    
    data = []
    missing_rna = 0
    missing_protein = 0
    
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            protein_id = parts[0]
            rna_id = parts[1]
            label = int(parts[2])
            
            # Get sequences
            if rna_id not in rna_sequences:
                missing_rna += 1
                continue
            if protein_id not in protein_sequences:
                missing_protein += 1
                continue
            
            rna_seq = rna_sequences[rna_id]
            protein_seq = protein_sequences[protein_id]
            
            data.append({
                'rna_id': rna_id,
                'protein_id': protein_id,
                'rna_sequence': rna_seq,
                'protein_sequence': protein_seq,
                'label': label
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total pairs: {len(df)}")
    print(f"  Positive pairs (interactions): {sum(df['label'] == 1)}")
    print(f"  Negative pairs (non-interactions): {sum(df['label'] == 0)}")
    print(f"  Unique RNAs: {df['rna_id'].nunique()}")
    print(f"  Unique proteins: {df['protein_id'].nunique()}")
    
    if missing_rna > 0 or missing_protein > 0:
        print(f"\nWarning:")
        print(f"  Missing RNA sequences: {missing_rna}")
        print(f"  Missing protein sequences: {missing_protein}")
    
    # Sequence length statistics
    df['rna_length'] = df['rna_sequence'].apply(len)
    df['protein_length'] = df['protein_sequence'].apply(len)
    
    print(f"\nSequence Length Statistics:")
    print(f"  RNA length - Mean: {df['rna_length'].mean():.1f}, "
          f"Min: {df['rna_length'].min()}, Max: {df['rna_length'].max()}")
    print(f"  Protein length - Mean: {df['protein_length'].mean():.1f}, "
          f"Min: {df['protein_length'].min()}, Max: {df['protein_length'].max()}")
    
    # Save full dataset with IDs
    full_output = output_file.replace('.csv', '_with_ids.csv')
    df.to_csv(full_output, index=False)
    print(f"\nFull dataset saved to: {full_output}")
    
    # Save simplified dataset for model training (without IDs)
    df_simple = df[['rna_sequence', 'protein_sequence', 'label']]
    df_simple.to_csv(output_file, index=False)
    print(f"Training dataset saved to: {output_file}")
    
    # Display sample data
    print(f"\nFirst 3 samples:")
    print(df_simple.head(3))
    
    return df


def main():
    """Main function."""
    # Paths
    rpiter_data_dir = 'data/real_datasets/RPITER/data'
    output_file = 'data/RPI1807_dataset.csv'
    
    # Check if RPITER data exists
    if not os.path.exists(rpiter_data_dir):
        print(f"Error: RPITER data directory not found: {rpiter_data_dir}")
        print("Please download RPITER dataset first.")
        return
    
    # Process dataset
    df = process_rpi1807_dataset(rpiter_data_dir, output_file)
    
    print("\n" + "="*60)
    print("Dataset processing complete!")
    print("="*60)
    print(f"\nYou can now train the model with:")
    print(f"  python train.py --data_path {output_file} --epochs 50")


if __name__ == '__main__':
    main()

