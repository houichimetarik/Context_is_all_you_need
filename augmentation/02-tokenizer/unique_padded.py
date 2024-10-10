import csv
from collections import Counter

def analyze_sequence_uniqueness(input_file):
    sequences = []
    total_rows = 0
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the header
        
        for row in reader:
            total_rows += 1
            if len(row) >= 2:  # Ensure the row has at least 2 columns
                sequence = row[1]
                sequences.append(sequence)
    
    # Count unique sequences
    unique_sequences = set(sequences)
    sequence_counts = Counter(sequences)
    
    # Calculate statistics
    num_unique_sequences = len(unique_sequences)
    percentage_unique = (num_unique_sequences / total_rows) * 100
    
    # Print results
    print(f"Total rows: {total_rows}")
    print(f"Unique sequences: {num_unique_sequences}")
    print(f"Percentage of unique sequences: {percentage_unique:.2f}%")
    
    # Display information about top duplicates
    duplicates = {seq: count for seq, count in sequence_counts.items() if count > 1}
    print(f"\nNumber of sequences that appear more than once: {len(duplicates)}")
    
    if duplicates:
        print("\nTop 10 most repeated sequences:")
        for sequence, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Sequence (first 50 chars): {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            print(f"Appears {count} times")
            print(f"Percentage: {(count / total_rows) * 100:.2f}%")
            print()


input_file = 'processed_padded_output.csv' 
analyze_sequence_uniqueness(input_file)