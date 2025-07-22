import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

from opportunity_tcn.dataset import (
    load_opportunity_data,
    preprocess_opportunity_data,
    create_adaptive_windows_expansion,
    IMUDataset,
    contrastive_collate_fn,
    classification_collate_fn
)

# --------- DATA PARTITIONING ----------
def partition_data(windows, labels, num_clients, random_seed=42):
    idxs = np.arange(len(windows))
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=random_seed)
    folds = []
    for _, test_idx in skf.split(idxs, labels):
        client_windows = [windows[i] for i in test_idx]
        client_labels = [labels[i] for i in test_idx]
        folds.append((client_windows, client_labels))
    return folds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def masked_mean_pooling(x, lengths):
    mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).float()
    x_sum = (x * mask).sum(dim=1)
    lengths = lengths.unsqueeze(1).float()
    return x_sum / lengths

class GatedSensorFusionN(nn.Module):
    def __init__(self, conv_channels, num_sensors):
        super().__init__()
        self.num_sensors = num_sensors
        self.gate = nn.Sequential(
            nn.Linear(num_sensors * conv_channels, num_sensors * conv_channels),
            nn.ReLU(),
            nn.Linear(num_sensors * conv_channels, num_sensors),
            nn.Softmax(dim=-1)
        )
    def forward(self, *xs):
        concat = torch.cat(xs, dim=-1)
        weights = self.gate(concat)
        fused = 0
        for i, x in enumerate(xs):
            w = weights[:, :, i].unsqueeze(-1)
            fused = fused + w * x
        return fused

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_dim = in_channels if i == 0 else hidden_dim
            layers.append(TemporalBlock(in_dim, hidden_dim, kernel_size, stride=1, dilation=dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2)

class OpportunityTCNEncoder(nn.Module):
    def __init__(self, conv_channels=16, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1, fusion_method='gated', group_indices=None):
        super().__init__()
        if group_indices is None:
            group_indices = [
                list(range(0,8)), list(range(8,16)), list(range(16,24)),
                list(range(24,32)), list(range(32,40)), list(range(40,48)), list(range(48,56))
            ]
        self.group_indices = group_indices
        self.num_groups = len(group_indices)
        self.group_channels = len(group_indices[0])

        self.conv_pipelines = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.group_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU()
            ) for _ in range(self.num_groups)
        ])

        if fusion_method == 'gated':
            self.sensor_fusion = GatedSensorFusionN(conv_channels, num_sensors=self.num_groups)
            fusion_dim = conv_channels
        else:
            self.sensor_fusion = None
            fusion_dim = self.num_groups * conv_channels

        self.tcn = TemporalConvNet(fusion_dim, hidden_dim, num_layers, kernel_size, dropout)

    def forward(self, x, lengths=None, return_final=False):
        groups = [x[:, :, indices] for indices in self.group_indices]
        conv_outs = []
        for i, conv in enumerate(self.conv_pipelines):
            group = groups[i].transpose(1,2)
            out = conv(group)
            out = out.transpose(1,2)
            conv_outs.append(out)
        if self.sensor_fusion is not None:
            fused = self.sensor_fusion(*conv_outs)
        else:
            fused = torch.cat(conv_outs, dim=-1)
        tcn_out = self.tcn(fused)
        if return_final:
            if lengths is not None:
                tcn_out = masked_mean_pooling(tcn_out, lengths)
            else:
                tcn_out = tcn_out.mean(dim=1)
        return tcn_out

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=64, proj_dim=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimCLRModel(nn.Module):
    def __init__(self, encoder, projection):
        super().__init__()
        self.encoder = encoder
        self.proj = projection

    def forward(self, x, lengths=None):
        z_seq = self.encoder(x, lengths)
        if lengths is not None:
            z = masked_mean_pooling(z_seq, lengths)
        else:
            z = z_seq.mean(dim=1)
        z = self.proj(z)
        return z

    def get_representation(self, x, lengths=None):
        z_seq = self.encoder(x, lengths)
        if lengths is not None:
            return masked_mean_pooling(z_seq, lengths)
        else:
            return z_seq.mean(dim=1)

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=64, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor([self.alpha] * inputs.size(1), device=inputs.device)
            else:
                alpha = torch.tensor(self.alpha, device=inputs.device)
            alpha = alpha.gather(0, targets)
            logpt = logpt * alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_dataloaders(
    X_train, y_train, X_finetune, y_finetune, X_test, y_test,
    batch_size=64
):
    contrastive_dataset = IMUDataset(X_train, y_train, mode='contrastive')
    classifier_dataset = IMUDataset(X_finetune, y_finetune, mode='classification')
    test_dataset = IMUDataset(X_test, y_test, mode='classification')

    contrastive_loader = DataLoader(
        contrastive_dataset, batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate_fn)
    finetune_loader = DataLoader(
        classifier_dataset, batch_size=batch_size, shuffle=True, collate_fn=classification_collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=classification_collate_fn)
    return contrastive_loader, finetune_loader, test_loader


def split_datasets(windows, labels, train_ratio=0.1, finetune_ratio=0.9, test_ratio=0.1):
    X_temp, X_test, y_temp, y_test = train_test_split(windows, labels, test_size=test_ratio, random_state=42, stratify=labels)
    finetune_ratio_adjusted = finetune_ratio / (train_ratio + finetune_ratio)
    X_train, X_finetune, y_train, y_finetune = train_test_split(X_temp, y_temp, test_size=finetune_ratio_adjusted, random_state=42, stratify=y_temp)
    return X_train, y_train, X_finetune, y_finetune, X_test, y_test



def load_and_preprocess_data(
    folder_path,
    initial_window_size=50,
    W_min=10,
    W_max=100,
    shift=10
):
    print("Loading Opportunity data from folder...")
    raw_df = load_opportunity_data(folder_path)
    print("Shape after loading:", raw_df.shape)

    print("Preprocessing and feature engineering...")
    proc_df = preprocess_opportunity_data(raw_df)
    unique_sorted = np.sort(proc_df['ML_Both_Arms'].unique())
    mapping = {old: new for new, old in enumerate(unique_sorted)}
    proc_df['ML_Both_Arms'] = proc_df['ML_Both_Arms'].map(mapping)
    print("Unique labels after remapping:", np.sort(proc_df['ML_Both_Arms'].unique()))
    print("Shape after preprocessing:", proc_df.shape)

    print("Creating adaptive windows...")
    windows, labels = create_adaptive_windows_expansion(
        proc_df, initial_window_size=initial_window_size,
        W_min=W_min, W_max=W_max,
        expand_step=10, contract_step=10,
        threshold_factor=0.25, shift=shift
    )
    print(f"Total windows created: {len(windows)}")
    return windows, labels


def build_models(
    num_classes=4, fusion_method='gated', group_indices=None,
    conv_channels=16, hidden_dim=64, num_layers=4, kernel_size=3, dropout=0.1
):
    encoder = OpportunityTCNEncoder(
        conv_channels=conv_channels, hidden_dim=hidden_dim,
        num_layers=num_layers, kernel_size=kernel_size, dropout=dropout,
        fusion_method=fusion_method, group_indices=group_indices
    )
    projection_head = ProjectionHead(input_dim=hidden_dim, proj_dim=32, dropout=0.5)
    simclr_model = SimCLRModel(encoder, projection_head)
    classifier_head = ClassifierHead(input_dim=hidden_dim, num_classes=num_classes)
    return encoder, simclr_model, classifier_head



def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    targets = torch.arange(B, 2 * B, device=z.device)
    targets = torch.cat([targets, torch.arange(0, B, device=z.device)])
    loss = F.cross_entropy(sim, targets)
    return loss

def train_simclr(model, dataloader, optimizer, device='cpu', epochs=10, temperature=0.07):
    model.train()
    loss_hist = []
    for ep in range(epochs):
        total_loss = 0.0
        for anchor, positive, lengths in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            lengths = lengths.to(device)
            anchor_emb = model(anchor, lengths)
            positive_emb = model(positive, lengths)
            loss = nt_xent_loss(anchor_emb, positive_emb, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_hist.append(avg_loss)
        print(f"Contrastive Epoch {ep+1}/{epochs} | NT-Xent Loss = {avg_loss:.4f}")
    return loss_hist

def train_classifier(encoder, classifier, dataloader, optimizer, device='cpu', epochs=5):
    encoder.train()
    classifier.train()
    loss_hist = []
    criterion = FocalLoss(gamma=2, alpha=None, reduction='mean')
    for ep in range(epochs):
        total_loss = 0.0
        for x, lbl, lengths in dataloader:
            x = x.to(device)
            lbl = lbl.to(device)
            lengths = lengths.to(device)
            rep_seq = encoder(x, lengths, return_final=False)
            rep = masked_mean_pooling(rep_seq, lengths)
            logits = classifier(rep)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_hist.append(avg_loss)
        print(f"Classifier Epoch {ep+1}/{epochs} | Focal Loss = {avg_loss:.4f}")
    return loss_hist

def evaluate(encoder, classifier, dataloader, device='cpu'):
    from sklearn.metrics import precision_score, recall_score, f1_score
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, lbl, lengths in dataloader:
            x = x.to(device)
            lbl = lbl.to(device)
            lengths = lengths.to(device)
            rep_seq = encoder(x, lengths, return_final=False)
            rep = masked_mean_pooling(rep_seq, lengths)
            logits = classifier(rep)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == lbl).sum().item()
            total += len(lbl)
            y_true.extend(lbl.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return correct / total, precision, recall, f1



def visualize_tsne(encoder, dataloader, device='cpu'):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    encoder.eval()
    reps, labels_list = [], []
    with torch.no_grad():
        for x, lbl, lengths in dataloader:
            x = x.to(device)
            lengths = lengths.to(device)
            rep_seq = encoder(x, lengths, return_final=False)
            rep = masked_mean_pooling(rep_seq, lengths)
            reps.append(rep.cpu().numpy())
            labels_list.append(lbl.numpy())
    reps = np.concatenate(reps, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    reps_2d = tsne.fit_transform(reps)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reps_2d[:,0], reps_2d[:,1], c=labels_list, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Encoder Representations (Supervised Fine-tuning)")
    plt.show()
