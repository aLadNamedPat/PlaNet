### This is where we'll build the encoder and decoders
# The encoder only takes the current observation and constructs an "encoded" observation from there
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, latent_dim=1024):
        super(Encoder, self).__init__()
        self.base_channels = base_channels
        
        # No padding to match PlaNet: 64 -> 31 -> 14 -> 6 -> 2
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2)   
        final_conv_size = base_channels * 8 * 2 * 2  # 256 * 4 = 1024
        
        self.fc = nn.Identity() if latent_dim == 1024 else nn.Linear(final_conv_size, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim=30, hidden_dim=200, base_channels=32, output_channels=3):
        super(Decoder, self).__init__()
        self.base_channels = base_channels
        
        # Input is latent_dim (stochastic state) + hidden_dim (deterministic state)
        # latent_dim=30 (posterior/prior state), hidden_dim=200 (GRU hidden state)
        self.fc = nn.Linear(latent_dim + hidden_dim, base_channels * 8 * 2 * 2)  # -> 1024
        
        # No padding to match PlaNet: 2 -> 6 -> 14 -> 31 -> 64
        self.deconv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(base_channels, output_channels, kernel_size=4, stride=2)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.base_channels * 8, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x