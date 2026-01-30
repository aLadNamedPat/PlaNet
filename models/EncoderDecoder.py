### This is where we'll build the encoder and decoders

# The encoder only takes the current observation and constructed an "encoded" observation from there

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, latent_dim=128, dropout_prob=0.2):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.dropout1 = nn.Dropout2d(dropout_prob)
        
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.dropout2 = nn.Dropout2d(dropout_prob)

        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.dropout3 = nn.Dropout2d(dropout_prob)

        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.dropout4 = nn.Dropout2d(dropout_prob)

        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x1 = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x2 = self.dropout2(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.dropout3(F.relu(self.bn3(self.conv3(x2))))
        x4 = self.dropout4(F.relu(self.bn4(self.conv4(x3))))

        x_fin = x4.view(x4.size(0), -1)

        mu = self.fc_mu(x_fin)
        logvar = self.fc_logvar(x_fin)

        out = self.reparameterize(mu, logvar)
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim = 128, base_channels=32, output_channels=3, dropout_prob=0.2):
        super(Decoder, self).__init__()

        self.base_channels = base_channels
        self.fc = nn.Linear(latent_dim + hidden_dim, base_channels * 8 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels * 4)
        self.dropout1 = nn.Dropout2d(dropout_prob)

        self.deconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.dropout2 = nn.Dropout2d(dropout_prob)

        self.deconv3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels)
        self.dropout3 = nn.Dropout2d(dropout_prob)

        self.deconv4 = nn.ConvTranspose2d(base_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.base_channels * 8, 4, 4)

        x1 = self.dropout1(F.relu(self.bn1(self.deconv1(x))))
        x2 = self.dropout2(F.relu(self.bn2(self.deconv2(x1))))
        x3 = self.dropout3(F.relu(self.bn3(self.deconv3(x2))))
        x4 = self.deconv4(x3)

        x = torch.sigmoid(x4)

        return x