import torch
import torch.nn as nn
import numpy as np

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.fc(out))

class Recovery(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(z_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z):
        out, _ = self.rnn(z)
        return torch.sigmoid(self.fc(out))

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        out, _ = self.rnn(h)
        return torch.sigmoid(self.fc(out))

class TimeGAN:
    """Simplified TimeGAN implementation for synthesizing market data."""
    def __init__(self, input_dim=5, hidden_dim=24, z_dim=5):
        self.embedder = Embedder(input_dim, hidden_dim)
        self.recovery = Recovery(hidden_dim, input_dim)
        self.generator = Generator(z_dim, hidden_dim)
        self.discriminator = Discriminator(hidden_dim)

    def generate(self, n_samples, seq_len):
        """Generate synthetic multivariate time series."""
        self.generator.eval()
        self.recovery.eval()
        with torch.no_grad():
            Z = torch.rand(n_samples, seq_len, self.generator.rnn.input_size)
            H_hat = self.generator(Z)
            X_hat = self.recovery(H_hat)
        return X_hat.numpy()

def generate_timegan_data(n_samples=100, seq_len=24):
    """
    Utility function to generate synthetic data imitating the structure
    of the real market dataset (Price, BaseLoad, Temperature, etc.)
    """
    model = TimeGAN(input_dim=5)
    # In a real scenario, we would load a trained TimeGAN.
    # Here we simulate generation.
    synthetic_data = model.generate(n_samples, seq_len)
    
    # Scale back to somewhat realistic values for the demo
    price = synthetic_data[:, :, 0] * 0.4  # Max ~0.4 $/kWh
    base_load = synthetic_data[:, :, 1] * 5000  # Max ~5000 kWh
    
    return {
        "price": price,
        "base_load": base_load
    }
