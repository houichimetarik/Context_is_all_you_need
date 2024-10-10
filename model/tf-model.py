import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging
import csv
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Global parameters
params = {
    'run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
    'batch_size': 64,
    'test_size': 0.25,
    'random_state': 42,
    'd_model': 512,
    'nhead': 16,
    'num_layers': 8,
    'dropout': 0.5,
    'learning_rate': 1e-4,
    'num_epochs': 160,
}

class GreekLetterDataset(Dataset):
    def __init__(self, sequences, labels, attention_masks):
        self.sequences = sequences
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
            'attention_mask': self.attention_masks[idx]
        }

def process_data(csv_file):
    df = pd.read_csv(csv_file)

    tokens = set()
    for sequence in df['sequence']:
        tokens.update(sequence.split())
    tokens = sorted(list(tokens))

    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
    tokens = special_tokens + [t for t in tokens if t not in special_tokens]

    token_to_id = {token: i for i, token in enumerate(tokens)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    sequences = []
    attention_masks = []

    for sequence in df['sequence']:
        tokens = sequence.split()
        encoded = [token_to_id.get(token, token_to_id['[UNK]']) for token in tokens]
        attention_mask = [1] * len(encoded)
        sequences.append(encoded)
        attention_masks.append(attention_mask)

    labels = sorted(list(set(df['label'])))
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    encoded_labels = [label_to_id[label] for label in df['label']]

    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long)
    attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)

    print_dataset_stats(sequences, token_to_id, label_to_id)

    dataset = GreekLetterDataset(sequences_tensor, labels_tensor, attention_masks_tensor)

    # Splitting data into train and test 25% for test 75% for training as specified in the global params
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=params['test_size'],
        random_state=params['random_state'],
        stratify=encoded_labels
    )
   
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'])

    return train_loader, test_loader, token_to_id, label_to_id, id_to_label

# Helper method
def print_dataset_stats(sequences, token_to_id, label_to_id):
    print(f"Vocabulary size: {len(token_to_id)}")
    print(f"Number of classes: {len(label_to_id)}")
    print(f"Max sequence length: {max(len(seq) for seq in sequences)}")
    print(f"Min sequence length: {min(len(seq) for seq in sequences)}")
    print(f"Average sequence length: {sum(len(seq) for seq in sequences) / len(sequences)}")

# Defining our transformer architecture
class GreekLetterTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, nhead, num_layers, dropout, max_len=148):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        batch_size, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        embedded = self.embedding(x)
        positional = self.pos_encoder(pos)

        x = embedded + positional
        x = x.permute(1, 0, 2)

        mask = (attention_mask == 0).bool()

        x = self.transformer(x, src_key_padding_mask=mask)
        x = x[0]
        x = self.fc(x)

        return x

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['sequence'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['sequence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    
    return total_loss / len(loader), accuracy, precision, recall, f1

# Helper method to display statistics Accuracy, Precision, Recall, F1-Score percentages
def init_or_update_csv_logger(filename, is_new_file=False):
    mode = 'w' if is_new_file else 'a'
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['Run ID'] + list(params.keys()) + ['Best Train Acc (%)', 'Best Train Loss', 'Best Test Acc (%)', 'Best Test Loss'])
        writer.writerow([params['run_id']] + list(params.values()) + ['', '', '', ''])
        writer.writerow(['Run ID', 'Epoch', 'Train Loss', 'Train Acc (%)', 'Train Precision (%)', 'Train Recall (%)', 'Train F1 (%)',
                         'Test Loss', 'Test Acc (%)', 'Test Precision (%)', 'Test Recall (%)', 'Test F1 (%)'])


def update_training_results(filename, run_id, epoch, train_metrics, test_metrics):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, epoch] + train_metrics + test_metrics)

def update_best_results(filename, run_id, best_train_acc, best_train_loss, best_test_acc, best_test_loss):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith(run_id):
            parts = line.strip().split(',')
            parts[-4:] = [f"{best_train_acc:.2f}", f"{best_train_loss:.4f}", f"{best_test_acc:.2f}", f"{best_test_loss:.4f}"]
            lines[i] = ','.join(parts) + '\n'
            break

    with open(filename, 'w') as f:
        f.writelines(lines)

# Saving model for later use
def save_model(model, token_to_id, id_to_label, params, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'token_to_id': token_to_id,
        'id_to_label': id_to_label,
        'vocab_size': len(token_to_id),
        'num_classes': len(id_to_label),
        'd_model': params['d_model'],
        'nhead': params['nhead'],
        'num_layers': params['num_layers'],
        'dropout': params['dropout']
    }, filename)
    print(f"Model saved to {filename}")
    logging.info(f"Model saved to {filename}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        logging.warning("CUDA is not available. Training on CPU.")
    else:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")

    csv_filename = 'training_logs.csv'
    is_new_file = not os.path.exists(csv_filename)

    init_or_update_csv_logger(csv_filename, is_new_file)

    train_loader, test_loader, token_to_id, label_to_id, id_to_label = process_data('processed_padded_output.csv')

    print(f"Vocabulary size: {len(token_to_id)}")
    print(f"Number of classes: {len(label_to_id)}")
    logging.info(f"Vocabulary size: {len(token_to_id)}")
    logging.info(f"Number of classes: {len(label_to_id)}")

    model = GreekLetterTransformer(
        vocab_size=len(token_to_id),
        num_classes=len(label_to_id),
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    best_train_loss = float('inf')
    best_train_acc = 0.0
    best_test_loss = float('inf')
    best_test_acc = 0.0

    for epoch in range(params['num_epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        train_loss, train_acc, train_precision, train_recall, train_f1 = validate(model, train_loader, criterion, device)
        test_loss, test_acc, test_precision, test_recall, test_f1 = validate(model, test_loader, criterion, device)

        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%, F1: {train_f1:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
              f'Precision: {test_precision:.2f}%, Recall: {test_recall:.2f}%, F1: {test_f1:.2f}%')
        logging.info(f'Epoch {epoch + 1}:')
        logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                     f'Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%, F1: {train_f1:.2f}%')
        logging.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                     f'Precision: {test_precision:.2f}%, Recall: {test_recall:.2f}%, F1: {test_f1:.2f}%')

        update_training_results(csv_filename, params['run_id'], epoch + 1, 
                                [train_loss, train_acc, train_precision, train_recall, train_f1],
                                [test_loss, test_acc, test_precision, test_recall, test_f1])

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), f'best_train_accuracy_model_{params["run_id"]}.pth')
            print(f'New best train accuracy model saved at epoch {epoch + 1}')
            logging.info(f'New best train accuracy model saved at epoch {epoch + 1}')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), f'best_train_loss_model_{params["run_id"]}.pth')
            print(f'New best train loss model saved at epoch {epoch + 1}')
            logging.info(f'New best train loss model saved at epoch {epoch + 1}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'best_test_accuracy_model_{params["run_id"]}.pth')
            print(f'New best test accuracy model saved at epoch {epoch + 1}')
            logging.info(f'New best test accuracy model saved at epoch {epoch + 1}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'best_test_loss_model_{params["run_id"]}.pth')
            print(f'New best test loss model saved at epoch {epoch + 1}')
            logging.info(f'New best test loss model saved at epoch {epoch + 1}')

    print(f"Training completed. Run ID: {params['run_id']}")
    print(f"Best training accuracy: {best_train_acc:.4f}")
    print(f"Best training loss: {best_train_loss:.4f}")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Best test loss: {best_test_loss:.4f}")
    logging.info(f"Training completed. Run ID: {params['run_id']}")
    logging.info(f"Best training accuracy: {best_train_acc:.4f}")
    logging.info(f"Best training loss: {best_train_loss:.4f}")
    logging.info(f"Best test accuracy: {best_test_acc:.4f}")
    logging.info(f"Best test loss: {best_test_loss:.4f}")

    update_best_results(csv_filename, params['run_id'], best_train_acc, best_train_loss, best_test_acc, best_test_loss)

    save_model(model, token_to_id, id_to_label, params, f'full_model_{params["run_id"]}.pth')

if __name__ == "__main__":
    main()

print("Training completed")
