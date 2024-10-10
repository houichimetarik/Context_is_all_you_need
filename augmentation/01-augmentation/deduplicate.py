import csv
from collections import Counter
import os

def analyze_and_remove_duplicates(input_file, output_file):
    sequences = []
    rows_by_sequence = {}
    total_rows = 0
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader, None)  # Save the header if it exists
        
        for row in reader:
            total_rows += 1
            if len(row) >= 3:  # Ensure the row has at least 3 columns
                sequence = row[1]
                if sequence not in rows_by_sequence:
                    rows_by_sequence[sequence] = row
                sequences.append(sequence)
    
    sequence_counts = Counter(sequences)
    duplicates = {seq: count for seq, count in sequence_counts.items() if count > 1}
    
    # Calculate and print statistics
    unique_sequences = len(sequence_counts)
    sequences_with_duplicates = len(duplicates)
    total_duplicate_rows = sum(duplicates.values()) - sequences_with_duplicates
    
    print("Duplication Statistics:")
    print(f"Total rows in original file: {total_rows}")
    print(f"Number of unique sequences: {unique_sequences}")
    print(f"Number of sequences with duplicates: {sequences_with_duplicates}")
    print(f"Total number of duplicate rows: {total_duplicate_rows}")
    print(f"Percentage of rows that are duplicates: {(total_duplicate_rows / total_rows) * 100:.2f}%")
    print(f"Percentage of unique sequences: {(unique_sequences / total_rows) * 100:.2f}%")
    
    # Write the deduplicated data to the new output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['path', 'sequence', 'label'])  # Write the new header
        
        for sequence, row in rows_by_sequence.items():
            writer.writerow(row)
    
    removed_count = total_rows - len(rows_by_sequence)
    print(f"\nRemoved {removed_count} duplicate rows.")
    print(f"Deduplicated CSV saved as {output_file}")
    print(f"New file contains {len(rows_by_sequence)} rows (all unique sequences)")

    # Verify the new file
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as outfile:
            reader = csv.reader(outfile)
            next(reader)  # Skip header
            row_count = sum(1 for row in reader)
        print(f"\nVerification: The new file '{output_file}' exists and contains {row_count} rows (excluding header).")
        if row_count == len(rows_by_sequence):
            print("The number of rows matches the expected count of unique sequences.")
        else:
            print("Warning: The number of rows does not match the expected count of unique sequences.")
    else:
        print(f"\nWarning: The new file '{output_file}' was not created.")


input_file = 'augmented_output.csv'
new_output_file = 'deduplicated_filtered_output.csv'  
analyze_and_remove_duplicates(input_file, new_output_file)