import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Κλασική sinusoidal positional encoding όπως αυτή που χρησιμοποιείται στα Transformers.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Δημιουργία πίνακα positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: Tensor (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerDNAModel(nn.Module):
    """
    Transformer-based classifier για DNA αλληλουχίες με k-mer tokenization.
    
    Περιλαμβάνει:
      - Embedding layer για k-mer tokens (με padding index = 0)
      - Positional Encoding για την ακολουθία
      - TransformerEncoder με multi-head self-attention layers (με masking για padding)
      - Mean pooling (αγνοώντας τα padded tokens)
      - Τελική πλήρως συνδεδεμένη στρώση για ταξινόμηση
    """
    def __init__(self,
                 vocab_size,
                 embed_dim=128,
                 n_heads=4,
                 num_layers=2,
                 dropout=0.2,
                 output_dim=2,
                 max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Embedding για k-mer tokens (0 για padding)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=embed_dim,
                                              dropout=dropout,
                                              max_len=max_len)
        
        # Ορισμός Transformer encoder layer και encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)  # Χρήση batch_first=True για ευκολία
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x):
        """
        x: Tensor (B, T) με k-mer tokens (με padding στις περιπτώσεις που χρειάζεται)
        """
        # Δημιουργία padding mask: True στις θέσεις όπου το x == 0
        padding_mask = (x == 0)
        
        # 1. Embedding: (B, T) -> (B, T, embed_dim)
        x = self.embedding(x)
        
        # 2. Προσθήκη positional encoding
        x = self.pos_encoder(x)
        
        # 3. Transformer encoder (με src_key_padding_mask για να αγνοηθούν τα padded tokens)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 4. Mean pooling πάνω στους πραγματικούς tokens (αγνοώντας τα padding)
        mask = (~padding_mask).unsqueeze(-1).float()  # (B, T, 1)
        x = x * mask  # μηδενίζουμε τις θέσεις των padded tokens
        summed = x.sum(dim=1)  # (B, embed_dim)
        lengths = mask.sum(dim=1)  # (B, 1)
        pooled = summed / lengths  # (B, embed_dim)
        
        # 5. Dropout και πλήρως συνδεδεμένη στρώση για την ταξινόμηση
        out = self.dropout(pooled)
        logits = self.fc(out)  # (B, output_dim)
        
        return logits
