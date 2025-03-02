import csv
import random
import torch
import numpy as np
from torch.utils.data import Dataset

def read_sequences_from_csv(filename):
    """
    Διαβάζει ένα CSV χωρίς header, όπου κάθε γραμμή είναι μία αλληλουχία DNA.
    Επιστρέφει λίστα από strings.
    """
    sequences = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                seq = row[0].strip()
                if seq:
                    sequences.append(seq)
    return sequences

def create_kmers(sequence, k=3):
    """
    Παίρνει ένα string (DNA sequence) και παράγει λίστα k-mer tokens.
    π.χ. k=3, "ACGT" -> ["ACG", "CGT"].
    """
    kmers = []
    length = len(sequence)
    for i in range(length - k + 1):
        kmers.append(sequence[i : i + k])
    return kmers

def build_kmer_vocab(all_sequences, k=3):
    """
    Φτιάχνει λεξικό k-mer -> integer index, με βάση ΟΛΑ τα data.
    """
    kmer_set = set()
    for seq in all_sequences:
        kms = create_kmers(seq, k)
        for km in kms:
            kmer_set.add(km)

    sorted_kmers = sorted(kmer_set)
    kmer_to_idx = {kmer: i+1 for i, kmer in enumerate(sorted_kmers)}  # 0 = PAD
    return kmer_to_idx

def encode_sequence_to_ids(seq, kmer_to_idx, k=3, max_length=200):
    kms = create_kmers(seq, k)
    ids = []
    for km in kms:
        if km in kmer_to_idx:
            ids.append(kmer_to_idx[km])
        else:
            ids.append(0)

    if len(ids) > max_length:
        ids = ids[:max_length]
    if len(ids) < max_length:
        ids = ids + [0]*(max_length - len(ids))

    return np.array(ids, dtype=np.int64)

class KmerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # shape: (N, max_length)
        self.y = y  # shape: (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_i = torch.tensor(self.X[idx], dtype=torch.long)
        y_i = torch.tensor(self.y[idx], dtype=torch.long)
        return x_i, y_i

def preprocess_data(normal_file, cancer_file, k=3, max_length=200):
    normal_seqs = read_sequences_from_csv(normal_file)
    cancer_seqs = read_sequences_from_csv(cancer_file)

    # Labels
    normal_labels = [0]*len(normal_seqs)
    cancer_labels = [1]*len(cancer_seqs)

    all_seqs   = normal_seqs + cancer_seqs
    all_labels = normal_labels + cancer_labels

    # Φτιάχνουμε vocab
    kmer_to_idx = build_kmer_vocab(all_seqs, k)

    X_arrays = []
    for seq in all_seqs:
        arr = encode_sequence_to_ids(seq, kmer_to_idx, k, max_length)
        X_arrays.append(arr)

    X_arrays = np.array(X_arrays, dtype=np.int64)
    y_arrays = np.array(all_labels, dtype=np.int64)

    # Χωρίζουμε σε train/val/test (60-20-20 ή 60-20-20, όπως θες)
    data = list(zip(X_arrays, y_arrays))
    random.shuffle(data)
    N = len(data)
    train_end = int(0.6*N)
    val_end   = int(0.8*N)

    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    X_train, y_train = zip(*train_data)
    X_val,   y_val   = zip(*val_data)
    X_test,  y_test  = zip(*test_data)

    X_train = np.array(X_train, dtype=np.int64)
    y_train = np.array(y_train, dtype=np.int64)
    X_val   = np.array(X_val,   dtype=np.int64)
    y_val   = np.array(y_val,   dtype=np.int64)
    X_test  = np.array(X_test,  dtype=np.int64)
    y_test  = np.array(y_test,  dtype=np.int64)

    train_dataset = KmerDataset(X_train, y_train)
    val_dataset   = KmerDataset(X_val,   y_val)
    test_dataset  = KmerDataset(X_test,  y_test)

    vocab_size = len(kmer_to_idx) + 1  # +1 για PAD=0

    return train_dataset, val_dataset, test_dataset, vocab_size
