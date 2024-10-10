import csv
from collections import Counter

def get_label_distribution(csv_file):
    label_counter = Counter()
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present
        
        for row in csv_reader:
            if len(row) >= 3:
                label = row[2]
                label_counter[label] += 1
    
    total_samples = sum(label_counter.values())
    
    print("Label Distribution:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    print(f"\nTotal samples: {total_samples}")


csv_file_path = 'deduplicated_filtered_output.csv'
get_label_distribution(csv_file_path)