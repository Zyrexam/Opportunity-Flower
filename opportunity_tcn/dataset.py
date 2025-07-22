import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.linalg import expm


#############################################
# 1. Data Loading from Multiple .dat Files
#############################################
def load_opportunity_data(folder_path):
    """
    Loads Opportunity data from all .dat files in the given folder.
    Selected 1-indexed columns are mapped and concatenated.
    """
    selected_idx = [38,39,40,41,42,43,    # BACK
                    51,52,53,54,55,56,    # RUA
                    64,65,66,67,68,69,    # RLA
                    77,78,79,80,81,82,    # LUA
                    90,91,92,93,94,95,    # LLA
                    109,110,111,112,113,114,  # L-SHOE
                    125,126,127,128,129,130,  # R-SHOE
                    250]                # Label
    selected_idx = [i - 1 for i in selected_idx]
    
    selected_names = [
        "BACK_accX", "BACK_accY", "BACK_accZ", "BACK_gyroX", "BACK_gyroY", "BACK_gyroZ",
        "RUA_accX", "RUA_accY", "RUA_accZ", "RUA_gyroX", "RUA_gyroY", "RUA_gyroZ",
        "RLA_accX", "RLA_accY", "RLA_accZ", "RLA_gyroX", "RLA_gyroY", "RLA_gyroZ",
        "LUA_accX", "LUA_accY", "LUA_accZ", "LUA_gyroX", "LUA_gyroY", "LUA_gyroZ",
        "LLA_accX", "LLA_accY", "LLA_accZ", "LLA_gyroX", "LLA_gyroY", "LLA_gyroZ",
        "LSHOE_Body_Ax", "LSHOE_Body_Ay", "LSHOE_Body_Az", "LSHOE_AngVelX", "LSHOE_AngVelY", "LSHOE_AngVelZ",
        "RSHOE_Body_Ax", "RSHOE_Body_Ay", "RSHOE_Body_Az", "RSHOE_AngVelX", "RSHOE_AngVelY", "RSHOE_AngVelZ",
        "ML_Both_Arms"
    ]

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dat')]
    df_list = []
    for file in files:
        df = pd.read_table(file, header=None, sep=r'\s+')
        
        df = df.iloc[:, selected_idx]
        df.columns = selected_names
        df_list.append(df)
    data = pd.concat(df_list, ignore_index=True)
    return data

#############################################
# 2. Data Preprocessing & Feature Engineering
#############################################
def preprocess_opportunity_data(df):
    """
    Fills NA, computes acceleration/gyroscope magnitude, standard scales features, and rearranges columns.
    """
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Compute magnitude features
    # BACK
    df["BACK_Acc"] = np.sqrt(df["BACK_accX"]**2 + df["BACK_accY"]**2 + df["BACK_accZ"]**2)
    df["BACK_Gyro"] = np.sqrt(df["BACK_gyroX"]**2 + df["BACK_gyroY"]**2 + df["BACK_gyroZ"]**2)
    # RUA
    df["RUA_Acc"] = np.sqrt(df["RUA_accX"]**2 + df["RUA_accY"]**2 + df["RUA_accZ"]**2)
    df["RUA_Gyro"] = np.sqrt(df["RUA_gyroX"]**2 + df["RUA_gyroY"]**2 + df["RUA_gyroZ"]**2)
    # RLA
    df["RLA_Acc"] = np.sqrt(df["RLA_accX"]**2 + df["RLA_accY"]**2 + df["RLA_accZ"]**2)
    df["RLA_Gyro"] = np.sqrt(df["RLA_gyroX"]**2 + df["RLA_gyroY"]**2 + df["RLA_gyroZ"]**2)
    # LUA
    df["LUA_Acc"] = np.sqrt(df["LUA_accX"]**2 + df["LUA_accY"]**2 + df["LUA_accZ"]**2)
    df["LUA_Gyro"] = np.sqrt(df["LUA_gyroX"]**2 + df["LUA_gyroY"]**2 + df["LUA_gyroZ"]**2)
    # LLA
    df["LLA_Acc"] = np.sqrt(df["LLA_accX"]**2 + df["LLA_accY"]**2 + df["LLA_accZ"]**2)
    df["LLA_Gyro"] = np.sqrt(df["LLA_gyroX"]**2 + df["LLA_gyroY"]**2 + df["LLA_gyroZ"]**2)
    # L-SHOE
    df["LSHOE_Acc"] = np.sqrt(df["LSHOE_Body_Ax"]**2 + df["LSHOE_Body_Ay"]**2 + df["LSHOE_Body_Az"]**2)
    df["LSHOE_Gyro"] = np.sqrt(df["LSHOE_AngVelX"]**2 + df["LSHOE_AngVelY"]**2 + df["LSHOE_AngVelZ"]**2)
    # R-SHOE
    df["RSHOE_Acc"] = np.sqrt(df["RSHOE_Body_Ax"]**2 + df["RSHOE_Body_Ay"]**2 + df["RSHOE_Body_Az"]**2)
    df["RSHOE_Gyro"] = np.sqrt(df["RSHOE_AngVelX"]**2 + df["RSHOE_AngVelY"]**2 + df["RSHOE_AngVelZ"]**2)

    # Column grouping/order
    back   = ["BACK_accX", "BACK_accY", "BACK_accZ", "BACK_gyroX", "BACK_gyroY", "BACK_gyroZ", "BACK_Acc", "BACK_Gyro"]
    rua    = ["RUA_accX", "RUA_accY", "RUA_accZ", "RUA_gyroX", "RUA_gyroY", "RUA_gyroZ", "RUA_Acc", "RUA_Gyro"]
    rla    = ["RLA_accX", "RLA_accY", "RLA_accZ", "RLA_gyroX", "RLA_gyroY", "RLA_gyroZ", "RLA_Acc", "RLA_Gyro"]
    lua    = ["LUA_accX", "LUA_accY", "LUA_accZ", "LUA_gyroX", "LUA_gyroY", "LUA_gyroZ", "LUA_Acc", "LUA_Gyro"]
    lla    = ["LLA_accX", "LLA_accY", "LLA_accZ", "LLA_gyroX", "LLA_gyroY", "LLA_gyroZ", "LLA_Acc", "LLA_Gyro"]
    lshoe  = ["LSHOE_Body_Ax", "LSHOE_Body_Ay", "LSHOE_Body_Az", "LSHOE_AngVelX", "LSHOE_AngVelY", "LSHOE_AngVelZ", "LSHOE_Acc", "LSHOE_Gyro"]
    rshoe  = ["RSHOE_Body_Ax", "RSHOE_Body_Ay", "RSHOE_Body_Az", "RSHOE_AngVelX", "RSHOE_AngVelY", "RSHOE_AngVelZ", "RSHOE_Acc", "RSHOE_Gyro"]

    sensor_cols = back + rua + rla + lua + lla + lshoe + rshoe  # 7 groups x 8 features = 56
    final_cols = sensor_cols + ["ML_Both_Arms"]
    
    df = df[final_cols].copy()

    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    return df

#############################################
# 3. Adaptive Windowing (Variable Length Sequences)
#############################################
def create_adaptive_windows_expansion(df, initial_window_size=50, W_min=10, W_max=300,
                                      expand_step=10, contract_step=10, threshold_factor=0.25,
                                      shift=50):
    """
    Creates adaptive windows over the time series.
    Windows are expanded or contracted based on the PDF of M.
    """
    acc_cols = ["BACK_Acc", "RUA_Acc", "RLA_Acc", "LUA_Acc", "LLA_Acc", "LSHOE_Acc", "RSHOE_Acc"]
    M = np.sqrt(np.sum(np.square(df[acc_cols].values), axis=1))
    mu = np.mean(M)
    sigma = np.std(M) + 1e-8
    peak_pdf = stats.norm.pdf(mu, loc=mu, scale=sigma) # Peak PDF(probability density function) value at mean
    threshold = threshold_factor * peak_pdf

    windows = []
    labels = []
    index = 0
    N = len(M)
    while index < N:
        W = initial_window_size
        if index >= N:
            break
        current_val = M[index]
        current_pdf = stats.norm.pdf(current_val, loc=mu, scale=sigma)
        if current_pdf > threshold:
            while index + W < N:
                next_val = M[index + W]
                next_pdf = stats.norm.pdf(next_val, loc=mu, scale=sigma)
                if next_pdf > current_pdf:
                    W += expand_step
                    current_pdf = next_pdf
                else:
                    break
        else:
            while W > W_min and current_pdf < threshold:
                W -= contract_step
        W = max(W_min, min(W, W_max))
        end_idx = index + W if index + W < N else N
        window_data = df.iloc[index:end_idx, :].copy()
        window_label = window_data["ML_Both_Arms"].mode()[0]
        window_signal = window_data.drop(columns=["ML_Both_Arms"])
        windows.append(window_signal)
        labels.append(window_label)
        index += shift
    return windows, labels

#############################################
# 4. Data Augmentation Functions
#############################################

def jitter(signal, sigma=0.05):
    """
    Adds random Gaussian noise to the input signal simulating minor shake or sensor noise, which improves model robustness.
    Parameters:
        signal (np.ndarray): Input signal of shape (T, D)
        sigma (float): Standard deviation of added noise
    Returns:
        np.ndarray: Augmented signal
    """
    return signal + np.random.normal(0, sigma, size=signal.shape)

def scaling(signal, sigma=0.15):
    """
    Randomly scales the amplitude of the signal, simulating different intensity movements.
    Parameters:
        signal (np.ndarray): Input signal of shape (T, D)
        sigma (float): Spread for random scaling factors
    Returns:
        np.ndarray: Augmented signal
    """
    factors = np.random.normal(1, sigma, size=(1, signal.shape[1]))
    return signal * factors

def DA_Rotation(X, max_angle=np.pi/12):
    """
    Applies a random rotation to the multi-dimensional signal (simulates sensor orientation change).
    Parameters:
        X (np.ndarray): Input signal of shape (T, D)
        max_angle (float): Maximum rotation angle in radians
    Returns:
        np.ndarray: Rotated signal
    """
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.random.uniform(-max_angle, max_angle)
    skew = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            skew[i, j] = -axis[j]
            skew[j, i] = axis[i]
    rotation_matrix = expm(skew * angle)
    return np.matmul(X, rotation_matrix)

def frequency_domain_augmentation(x, freq_scale_min=0.9, freq_scale_max=1.1, p=0.5):
    """
    Alters the signal in the frequency domain by scaling FFT=Fast Fourier Transform coefficients.
    Parameters:
        x (np.ndarray): Input signal of shape (T, D)
        freq_scale_min (float): Minimum FFT scale
        freq_scale_max (float): Maximum FFT scale
        p (float): Probability to apply transformation
    Returns:
        np.ndarray: Augmented signal
    """
    if np.random.rand() < p:
        time_len, num_channels = x.shape
        for c in range(num_channels):
            sig = x[:, c]
            sig_fft = np.fft.fft(sig)
            scale_factor = np.random.uniform(freq_scale_min, freq_scale_max)
            sig_fft = sig_fft * scale_factor
            augmented = np.fft.ifft(sig_fft).real
            x[:, c] = augmented
    return x

def augment(signal):
    """
    Applies a random augmentation method to the input sensor window.
    Converts torch tensor to numpy if needed.
    Randomly selects one augmentation from jitter, scaling, rotation, or frequency-domain methods.
    Parameters:
        signal (np.ndarray or torch.Tensor): Input signal window
    Returns:
        np.ndarray: Augmented signal
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    transforms = [jitter, scaling, DA_Rotation, frequency_domain_augmentation]
    transform = np.random.choice(transforms)
    return transform(signal.copy())

#############################################
# 5. PyTorch Dataset
#############################################
class IMUDataset(Dataset):
    """
    Mode: 'contrastive' (returns 2 augmented views), 'classification' (returns signal + label)
    """
    def __init__(self, windows, labels, mode='classification'):
        self.windows = windows  # list of DataFrames (variable-length sequences)
        self.labels = labels    # list of labels
        self.mode = mode

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if isinstance(window, pd.DataFrame):
            window = window.to_numpy()
        if self.mode == 'contrastive':
            x1 = torch.tensor(window, dtype=torch.float32)
            x2 = torch.tensor(augment(window), dtype=torch.float32)
            return x1, x2
        else:
            x = torch.tensor(window, dtype=torch.float32)
            label = self.labels[idx]
            return x, label

#############################################
# 6. Utility Collate Functions for Dataloaders
#############################################



def pad_sequence(seq, max_length):
    """
    Pads a variable-length sequence to a fixed length with zeros.
    Parameters:
        seq (torch.Tensor): Sequence tensor of shape (L, D)
        max_length (int): Maximum sequence length in the batch
    Returns:
        torch.Tensor: Padded sequence of shape (max_length, D)
    """
    L, D = seq.shape
    if L < max_length:
        # If the sequence is shorter than max_length, pad with zeros at the end.
        pad = torch.zeros(max_length - L, D, dtype=seq.dtype)
        return torch.cat([seq, pad], dim=0)
    else:
        # If already at or above max_length, return as-is (truncated if needed elsewhere).
        return seq

def contrastive_collate_fn(batch):
    """
    Collate function for unsupervised (SimCLR-style) contrastive learning.
    Each batch sample should be a tuple of (view1, view2), both tensors.
    Pads all sequences in the batch to the same maximum length.
    Returns:
        anchors (torch.Tensor): Batch of anchor windows (B, max_length, D)
        positives (torch.Tensor): Batch of positive windows (B, max_length, D)
        lengths (torch.Tensor): Original lengths before padding (B,)
    """
    max_length = max(sample[0].shape[0] for sample in batch)
    anchors, positives, lengths = [], [], []
    for x1, x2 in batch:
        L = x1.shape[0]
        lengths.append(L)
        anchors.append(pad_sequence(x1, max_length))
        positives.append(pad_sequence(x2, max_length))
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return anchors, positives, lengths

def classification_collate_fn(batch):
    """
    Collate function for window-level classification tasks.
    Each batch sample should be a tuple of (window, label).
    Pads all windows to the batch's maximum length.
    Returns:
        xs (torch.Tensor): Batch of sensor windows (B, max_length, D)
        lbls (torch.Tensor): Corresponding labels (B,)
        lengths (torch.Tensor): Original lengths before padding (B,)
    """
    max_length = max(sample[0].shape[0] for sample in batch)
    xs, lbls, lengths = [], [], []
    for x, label in batch:
        L = x.shape[0]
        lengths.append(L)
        xs.append(pad_sequence(x, max_length))
        lbls.append(label)
    xs = torch.stack(xs)
    lbls = torch.tensor(lbls, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return xs, lbls, lengths





def load_subject_windows(subject_folder, **window_args):
    df = load_opportunity_data(subject_folder)
    df = preprocess_opportunity_data(df)
    unique_sorted = np.sort(df['ML_Both_Arms'].unique())
    mapping = {old: new for new, old in enumerate(unique_sorted)}
    df['ML_Both_Arms'] = df['ML_Both_Arms'].map(mapping)
    windows, labels = create_adaptive_windows_expansion(df, **window_args)
    return windows, labels
