import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from preprocessing import preprocess_data
from model import TransformerDNAModel


def evaluate_model(model, loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            eval_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            trues = y.cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(trues)

    avg_loss = eval_loss / len(loader)
    acc = accuracy_score(all_trues, all_preds)
    rec = recall_score(all_trues, all_preds, average="binary")
    f1 = f1_score(all_trues, all_preds, average="binary")
    conf_mat = confusion_matrix(all_trues, all_preds)

    return avg_loss, acc, rec, f1, conf_mat

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    n_epochs=10,
    save_path="best_model.pt"
):
    best_val_f1 = 0.0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            trues = y.detach().cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(trues)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_trues, all_preds)
        train_rec = recall_score(all_trues, all_preds, average="binary")
        train_f1  = f1_score(all_trues, all_preds, average="binary")

        # Evaluate on validation
        val_loss, val_acc, val_rec, val_f1, val_cm = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{n_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}   | Rec: {val_rec:.4f}   | F1: {val_f1:.4f}")
        print("Val Confusion Matrix:")
        print(val_cm)
        print("-"*50)

        # Αν βελτιώθηκε το val_f1, αποθηκεύουμε το μοντέλο
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"--> New best val F1: {val_f1:.4f}. Model saved to {save_path}.\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # CSV αρχεία
    normal_file = "normalpanc/Data/FLIBASE-ALL-NORMAL-SRT-sequences.csv"
    cancer_file = "normalpanc/Data/FLIBASE-ALL-CANCER-SRT-sequences.csv"

    # Φορτώνουμε τα datasets
    train_dataset, val_dataset, test_dataset, vocab_size = preprocess_data(
        normal_file,
        cancer_file,
        k=6,
        max_length=5000
    )

    print("Train size:", len(train_dataset))
    print("Val   size:", len(val_dataset))
    print("Test  size:", len(test_dataset))
    print("Vocab size (k-mers):", vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False)

    # Φτιάχνουμε το μοντέλο
    embed_dim  = 256*2   # ή όποια τιμή θεωρείς καλύτερη
    n_heads    = 4
    num_layers = 2
    dropout    = 0.1
    output_dim = 2  # 2-class

    model = TransformerDNAModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim,
        max_len=5000  # πρέπει να ταιριάζει με το max_length στο preprocessing
    ).to(device)

    model = nn.DataParallel(model)  # Αυτό θα κατανείμει το μοντέλο και στα δύο GPU


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # === [ΝΕΟ ΣΗΜΕΙΟ] ΧΡΗΣΗ WEIGHTED CROSS ENTROPY ===
    # Παράδειγμα: Αν θέλεις να τιμωρήσεις περισσότερο τα λάθη στην κλάση 1,
    # αυξάνεις το βάρος της κλάσης 1.
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    n_epochs = 100
    best_model_path = "best_model.pt"

    # Εκπαίδευση
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        n_epochs,
        save_path=best_model_path
    )

    # Φορτώνουμε το καλύτερο μοντέλο
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} for final evaluation on Test set.")

    # Τελική αξιολόγηση στο test set
    test_loss, test_acc, test_rec, test_f1, test_cm = evaluate_model(model, test_loader, criterion, device)
    print("\n==== Test Results ====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test Rec:  {test_rec:.4f}")
    print(f"Test F1:   {test_f1:.4f}")
    print("Test Confusion Matrix:")
    print(test_cm)


if __name__ == "__main__":
    main()
