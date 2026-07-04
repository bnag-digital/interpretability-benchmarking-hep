"""
JetDNN: a small MLP classifier for the hls4ml jet-tagging features, plus a
training loop with explicit early stopping that mirrors the BDT's
early_stopping_rounds=10, so both models are tuned under comparable stopping
criteria before their explanations are compared.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class JetDNN(nn.Module):
    def __init__(self, n_features: int = 16, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _to_tensors(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def _run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (out.argmax(1) == y_batch).sum().item()
            n += len(y_batch)
    return total_loss / n, correct / n


def train_dnn(
    X_train_sc, y_train, X_val_sc, y_val,
    n_features=16, n_classes=5,
    batch_size=1024, lr=1e-4, epochs=500, early_stop_patience=10,
    checkpoint_path="best_jet_dnn.pt", verbose=True,
):
    """
    Train JetDNN with Adam + ReduceLROnPlateau + early stopping on val loss.
    Saves the best checkpoint to checkpoint_path and reloads it before returning.
    """
    train_loader = DataLoader(_to_tensors(X_train_sc, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_tensors(X_val_sc, y_val), batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    model = JetDNN(n_features=n_features, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {early_stop_patience} epochs). "
                      f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    return model, device


def evaluate_dnn(model, device, X_test_sc, y_test, class_names, batch_size=1024, verbose=True):
    """
    Returns (y_pred, y_prob, per_class_auc_dict).
    """
    test_loader = DataLoader(_to_tensors(X_test_sc, y_test), batch_size=batch_size)

    all_probs, all_preds, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out = model(X_batch.to(device))
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(1).cpu().numpy())
            all_labels.append(y_batch.numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    if verbose:
        print(f"DNN Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))

    per_class_auc = {}
    for i, cls in enumerate(class_names):
        auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        per_class_auc[cls] = auc
        if verbose:
            print(f"  {cls} tagger: AUC = {auc:.3f}")

    return y_pred, y_prob, per_class_auc


def dnn_predict_proba_factory(model, device):
    """
    Wrap a trained JetDNN as a numpy-in/numpy-out predict_proba function,
    the interface LIME's LimeTabularExplainer expects.
    """
    def predict_proba(X_np):
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            probs = torch.softmax(model(X_t), dim=1)
        return probs.cpu().numpy()
    return predict_proba
