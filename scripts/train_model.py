import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from helper_for_csvs import read_csv
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def flatten_and_convert(nested_list):
    flattened_list = []
    for sublist in nested_list:
        for array in sublist:
            flattened_list.append(torch.tensor(array, dtype=torch.float32))
    return flattened_list

# Convert input and output polylines
input_polylines_tensor = flatten_and_convert(read_csv(r'C:\Users\GURDARSH VIRK\OneDrive\Documents\adobe-gensolve-2024\problems\problems\isolated.csv'))
output_polylines_tensor = flatten_and_convert(read_csv(r'C:\Users\GURDARSH VIRK\OneDrive\Documents\adobe-gensolve-2024\problems\problems\isolated_sol.csv'))

class PolylineDataset(Dataset):
    def __init__(self, input_polylines, output_polylines):
        self.input_polylines = input_polylines
        self.output_polylines = output_polylines

    def __len__(self):
        return len(self.input_polylines)

    def __getitem__(self, idx):
        return self.input_polylines[idx], self.output_polylines[idx]

# Create dataset and dataloader
dataset = PolylineDataset(input_polylines_tensor, output_polylines_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define models
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return torch.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        lstm_out = lstm_out.squeeze(1)
        context = context.squeeze(1)
        output = self.out(torch.cat([lstm_out, context], 1))
        return output, hidden, cell

# Initialize models and optimizers
input_size = 2  # 2D points (x, y)
hidden_size = 128
output_size = 2  # 2D points (x, y)
encoder = Encoder(input_size, hidden_size).to(device)
decoder = Decoder(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for input_polyline, output_polyline in dataloader:
        # Move tensors to the device
        input_polyline = input_polyline.to(device)
        output_polyline = output_polyline.to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        encoder_outputs, hidden, cell = encoder(input_polyline)
        decoder_input = torch.zeros_like(output_polyline[:, 0, :]).unsqueeze(1).to(device)  # Initial input to decoder
        loss = 0
        
        for t in range(output_polyline.size(1)):
            decoder_output, hidden, cell = decoder(decoder_input, hidden, cell, encoder_outputs)
            loss += criterion(decoder_output, output_polyline[:, t, :])
            decoder_input = output_polyline[:, t, :].unsqueeze(1)  # Teacher forcing

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
