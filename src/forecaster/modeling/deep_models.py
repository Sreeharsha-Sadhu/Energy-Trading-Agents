import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class PinballLoss(nn.Module):
    """Quantile Loss (Pinball Loss) for probabilistic forecasting."""
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        error = target - preds
        loss = torch.max((self.quantile - 1) * error, self.quantile * error)
        return torch.mean(loss)

class HybridLSTMCNN(nn.Module):
    """Hybrid CNN-LSTM model for sequence forecasting."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        # 1D CNN for feature extraction
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> Conv1d needs (batch, channels, length)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

class KalmanVikingModel(nn.Module):
    """Deep learning approximation of Kalman-Viking state-space model."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Simplified representation: GRU to track state, linear to map to observation
        self.state_tracker = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.observation_mapper = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, state = self.state_tracker(x)
        pred = self.observation_mapper(out[:, -1, :])
        return pred

class PyTorchScikitWrapper(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for PyTorch models."""
    def __init__(self, model_class, input_dim, epochs=10, lr=0.001, quantile=0.5):
        self.model_class = model_class
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.quantile = quantile
        self.model = None

    def fit(self, X, y):
        # Infer input dimension from X
        if isinstance(X, np.ndarray):
            num_features = X.shape[1]
        else:
            num_features = X.shape[1]  # pandas dataframe
            
        self.model = self.model_class(input_dim=num_features)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = PinballLoss(quantile=self.quantile)

        # Convert to tensors
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        # Add sequence dimension since these models expect 3D input (batch, seq, features)
        X_tensor = X_tensor.unsqueeze(1) 
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).view(-1, 1)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Save feature names for inference.py
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        X_tensor = X_tensor.unsqueeze(1)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.numpy().flatten()
