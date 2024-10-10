import csv
import os
from collections import Counter

# Input and output file names
input_file = 'deduplicated_filtered_output-70929.csv'
output_file = 'processed_padded_output.csv'
vocab_file = 'vocabulary.txt'

# Special tokens
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

MAX_LENGTH = 148 

def tokenize(sequence):
    # Tokenize the sequence into 3-character tokens
    tokens = ['[CLS]'] + [sequence[i:i+3] for i in range(0, len(sequence), 3)] + ['[SEP]']
    
    # Truncate or pad the sequence to MAX_LENGTH
    if len(tokens) > MAX_LENGTH:
        tokens = tokens[:MAX_LENGTH-1] + ['[SEP]']
    else:
        tokens = tokens + ['[PAD]'] * (MAX_LENGTH - len(tokens))
    
    return tokens

def create_vocabulary(all_tokens):
    # Create vocabulary with special tokens at the beginning and all unique tokens
    unique_tokens = set(all_tokens)
    vocab = SPECIAL_TOKENS + list(unique_tokens)
    return vocab


all_tokens = []
tokenized_data = []

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)
    sequence_column = header.index('sequence')

    for row in reader:
        if len(row) > sequence_column:
            path = row[0]
            sequence = row[sequence_column]
            label = row[-1]
            tokens = tokenize(sequence)
            all_tokens.extend([t for t in tokens if t not in SPECIAL_TOKENS])
            tokenized_data.append((path, tokens, label))

# Create vocabulary
vocabulary = create_vocabulary(all_tokens)

# Write vocabulary to file
with open(vocab_file, 'w', encoding='utf-8') as f:
    for token in vocabulary:
        f.write(f"{token}\n")

# Write tokenized data to new CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)

    for path, tokens, label in tokenized_data:
        writer.writerow([path, ' '.join(tokens), label])

print(f"Tokenized and processed CSV file saved as {output_file}")
print(f"Vocabulary saved as {vocab_file}")
print(f"Vocabulary size: {len(vocabulary)}")


sequence_lengths = [len(tokens) for _, tokens, _ in tokenized_data]
print(f"Maximum sequence length: {MAX_LENGTH}")
print(f"Average sequence length (before padding): {sum(sequence_lengths) / len(sequence_lengths):.2f}")
print(f"Median sequence length (before padding): {sorted(sequence_lengths)[len(sequence_lengths)//2]}")


label_counter = Counter()
for _, _, labels in tokenized_data:
    label_counter.update(labels.split('|'))

print("\nLabel distribution:")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")

print(f"\nTotal number of sequences: {len(tokenized_data)}")
print(f"Number of unique labels: {len(label_counter)}")