# Transformer-Based Sequence Classification Model

⚡ We have trained the model on 1.9k tokens and 35,462 labeled Python files.

⚡ A 26.3 million parameter model

🎇 Design Patterns Python Dataset available on : https://drive.google.com/drive/folders/1xoFFJgrpmOABEgwRvs1c9Q4ZCWer-Kkc?usp=sharing

## Achieved Results
| Metric | Test    |
|--------|---------|
| Accuracy    | 92.52%  |
| Precision   | 92.55%  |
| Recall    | 92.52%  |
| F1-score    | 92.47%  |

## Model Architecture
 - Model Type: Transformer Encoder
 - Task: Sequence Classification


## Hyperparameters
```python
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

```

### Detailed Parameter Explanation:

1. `batch_size`: 64
   - Number of samples processed before the model is updated.

2. `test_size`: 0.25
   - Proportion of the dataset to include in the test split.
   - Represents 25% of the total dataset.
   - The remaining 75% is used for training.

3. `random_state`: 42
   - Ensures that random operations produce the same results across different runs.

4. `d_model`: 512
   - Dimensionality of the model's hidden states and embeddings.

5. `nhead`: 16
   - Number of attention heads in the multi-head attention layers.

6. `num_layers`: 8
   - Number of transformer encoder layers.

7. `dropout`: 0.5
   - Dropout rate applied in the model for regularization.

8. `learning_rate`: 1e-4 (0.0001)

9. `num_epochs`: 160
   - Number of complete passes through the training dataset.
  

## Model Structure
```python
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
```

## Training Details
- Optimizer: Adam
- Loss Function: Cross-Entropy Loss
- Device: CUDA (GPU) if available, otherwise CPU 
- Train/Test Split: 75% train, 25% test

## Data Processing
- Input: CSV file containing sequences and labels
- Tokenization: Space-separated tokens
- Special Tokens: '[PAD]', '[UNK]', '[CLS]', '[SEP]'
- Sequence Padding: Up to max length in the dataset (148)


## Logging and Checkpoints
- Training progress logged to 'training.log'
- Best models saved based on:
  - Best train accuracy
  - Best train loss
  - Best test accuracy
  - Best test loss


## Project Structure

- 📁 DESIGNPATTERNSDETECTION.PY
  - 📁 config
    - 📄 requirements.txt
  - 📁 augmentation
    - 📁 01-augmentation
      - 📄 augment.py
      - 📄 deduplicate.py
      - 📄 distribution.py
      - 📄 unique.py
    - 📁 02-tokenizer
      - 📄 processed_padded_output.csv
      - 📄 tokenizer.py
      - 📄 unique_padded.py
      - 📄 vocabulary.txt
  - 📁 data
    - 📁 necessary-files
    - 📁 scrapers
      - 📁 github
      - 📁 gitlab
      - 📁 bitbucket
    - 📁 sequencer
    - 📄 deduplicated_filtered_output-35462.csv
    - 📄 processed_padded_output-35462.csv
    - 📄 vocabulary-35462.txt
  - 📁 model
    - 📄 processed_padded_output.csv
    - 📄 tf-model.py
    - 📄 vocabulary.txt
  - 📄 .gitignore
  - 📄 DOCUMENTATION.md
  - 📄 README.md