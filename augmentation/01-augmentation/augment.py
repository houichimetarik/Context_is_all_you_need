import csv
import random
import string
import unicodedata


GREEK_SYMBOLS = {
    "\u03A0": "Π", "\u0394": "Δ", "\u03A8": "Ψ", "\u03F4": "ϴ", "\u039B": "Λ",
    "\u03A9": "Ω", "\u03A3": "Σ", "\u03A6": "Φ", "\u0393": "Γ", "\u03A4": "Τ", "\u039E": "Ξ"
}

def is_greek_symbol(char):
    return char in GREEK_SYMBOLS.values()

def augment_sequence(sequence):
    unique_chars = set(sequence)
    all_chars = set(string.ascii_uppercase)
    available_chars = all_chars - unique_chars
    
    char_map = {}
    for char in unique_chars:
        if char.isalpha() and not is_greek_symbol(char):
            if available_chars:
                new_char = random.choice(list(available_chars))
                char_map[char] = new_char
                available_chars.remove(new_char)
            else:
                char_map[char] = char
        else:
            char_map[char] = char
    
    augmented_sequence = ''.join(char_map[char] for char in sequence)
    return augmented_sequence

def augment_path(path, index):
    parts = path.rsplit('.', 1)
    if len(parts) == 2:
        return f"{parts[0]}_augmented_{index}.{parts[1]}"
    else:
        return f"{path}_augmented_{index}"

def augment_csv(input_file, output_file, target_count=3084):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)
        writer.writerow(header)
        
        rows = list(reader)
        
        label_data = {}
        for row in rows:
            if len(row) == 3:
                path, sequence, label = row
                if label not in label_data:
                    label_data[label] = []
                label_data[label].append((path, sequence))
        
        unique_sequences = set()
        for label, data in label_data.items():
            original_count = len(data)
            augmentation_needed = target_count - original_count
            
            # Write original data
            for path, sequence in data:
                writer.writerow([path, sequence, label])
                unique_sequences.add(sequence)
            
            # Augment data
            augmentation_count = 0
            attempts = 0
            while augmentation_count < augmentation_needed and attempts < augmentation_needed * 10:
                attempts += 1
                original_path, original_sequence = random.choice(data)
                augmented_sequence = augment_sequence(original_sequence)
                if augmented_sequence not in unique_sequences:
                    augmented_path = augment_path(original_path, augmentation_count + 1)
                    writer.writerow([augmented_path, augmented_sequence, label])
                    unique_sequences.add(augmented_sequence)
                    augmentation_count += 1
            
            if augmentation_count < augmentation_needed:
                print(f"Warning: Could not generate enough unique sequences for label {label}. Generated {augmentation_count} out of {augmentation_needed} needed.")


input_file = 'output.csv'
output_file = 'augmented_output.csv'

augment_csv(input_file, output_file)

print(f"Augmentation complete. Output saved to {output_file}")