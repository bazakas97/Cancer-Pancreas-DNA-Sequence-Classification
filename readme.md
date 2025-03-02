
# Cancer Pancreas DNA Sequence Classification

This project aims to classify DNA sequences from pancreatic cancer and normal samples using a Transformer-based model.

## Project Structure

```
/home/chbazakas/Documents/SLEPC/cancerpancreas/
│
├── train_and_validation.py  # Script for training and validating the model
├── preprocessing.py         # Functions for preprocessing DNA sequences
├── model.py                 # Definition of the Transformer-based model
├── lstm_attention_v1/       # Directory for LSTM-based model (if applicable)
│   └── preprocessing.py     # Preprocessing functions for LSTM model
├── Data/                    # Directory containing the DNA sequence CSV files
│   ├── Pancreas_normal_flibase-coding.csv
│   └── FLIBASE-ALL-CANCER-SRT-sequences.csv
└── readme.md                # This README file
```

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- numpy

Install the required packages using pip:

```bash
pip install torch scikit-learn numpy
```

## How to Run

1. **Preprocess the Data:**

   The `preprocessing.py` script contains functions to read DNA sequences from CSV files, create k-mers, and encode sequences into numerical format.

2. **Train and Validate the Model:**

   Run the `train_and_validation.py` script to train the Transformer-based model:

   ```bash
   python train_and_validation.py
   ```

   This script will:
   - Load and preprocess the data.
   - Define and initialize the Transformer model.
   - Train the model on the training dataset.
   - Validate the model on the validation dataset.
   - Save the best model based on validation F1 score.

3. **Evaluate the Model:**

   After training, the script will load the best model and evaluate it on the test dataset, printing the test loss, accuracy, recall, F1 score, and confusion matrix.

## Model Details

The model is a Transformer-based classifier for DNA sequences with k-mer tokenization. It includes:
- An embedding layer for k-mer tokens.
- Positional encoding for the sequence.
- Transformer encoder with multi-head self-attention layers.
- Mean pooling, ignoring padded tokens.
- A fully connected layer for classification.

## Data

The data consists of DNA sequences from normal and cancerous pancreatic samples. The sequences are stored in CSV files in the `Data/` directory.

## Contact

For any questions or issues, please contact [your-email@example.com].
