import torch
import torch.nn as nn
import torch.optim as optim
from .RevIN import RevIN
from .dataloader import TSDataset, TSDataLoader

class BaseProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def warmup(self, dataset, n_epochs=10, batch_size=64, learning_rate=0.001, log=False):
        # Create a DataLoader for the warmup data
        if not isinstance(dataset, TSDataset):
            dataset = TSDataset(torch.tensor(dataset, dtype=self.dtype))

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if log:
            print(f'Warming up with {len(data_loader)} batches of size {batch_size}')

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer     = optim.Adam(self.parameters(), lr=learning_rate)

        # Train the autoencoder for n_epochs
        for epoch in range(n_epochs):
            cum_loss = 0
            for batch in data_loader:
                inputs = batch[0]

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed = self(inputs)

                # Calculate the loss
                loss = loss_function(reconstructed, inputs)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the cumulative loss
                cum_loss += loss.item()
            if log:
                print(f'Epoch: {epoch}, Loss: {cum_loss}')

# https://stackoverflow.com/questions/44130851/simple-lstm-in-pytorch-with-sequential-module
class extract_tensor(nn.Module):
    def forward(self,input, return_type='hidden'):
        # Output shape (batch, features, hidden)
        x, (hidden_n, cell_n) = input
        # Reshape shape (batch, hidden)
        if return_type == 'hidden':
            x = hidden_n[-1]
        elif return_type == 'cell':
            x = cell_n[-1]

        return x
    
class LSTMMaskedAutoencoderProjection(BaseProjectionLayer):
    def __init__(self, input_dims, hidden_dims, output_dims, use_revin=True, dtype=torch.float32):
        super().__init__()

        self.dtype = dtype
        self.use_revin = use_revin
        if use_revin:
            self.revin_layer = RevIN(input_dims[1])
        

        self.lstm_encoder = nn.LSTM(input_size=input_dims[1], hidden_size=hidden_dims, batch_first=True)
        # self.linear_encoder = nn.Linear(hidden_dims, output_dims)

        self.linear_decoder = nn.Linear(hidden_dims, input_dims[1])
        # self.lstm_decoder = nn.LSTM(input_size=hidden_dims, hidden_size=input_dims[1], batch_first=True)


    def forward(self, x):
        # Implement the full forward pass for masked autoencoder (including decoder)

        if self.use_revin:
            x                       = self.revin_layer(x, 'norm')
        enc_output, (_, _)        = self.lstm_encoder(x)
        # enc_output              = self.linear_encoder(enc_output)
        x_decoded               = self.linear_decoder(enc_output)
        # x_decoded, _            = self.lstm_decoder(enc_output)
        if self.use_revin:
            x_decoded               = self.revin_layer(x_decoded, 'denorm')
       
        return x_decoded
    
    def encode(self, x):
        # Implement forward pass for masked autoencoder
        # For the TSFM model, we only need the embeddings, so we only return the output of the encoder.
        if self.use_revin:
            x              = self.revin_layer(x, 'norm')

        enc_output, (_, _) = self.lstm_encoder(x) #enc_output[:, -1, :]
        # enc_output         = self.linear_encoder(enc_output)

        if self.use_revin:
            enc_output     = self.revin_layer(enc_output, 'denorm')

        return enc_output
    



class VAEProjection(BaseProjectionLayer):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        # Define the architecture for VAE
        self.encoder = nn.Sequential(
            # ...
        )
        self.decoder = nn.Sequential(
            # ...
        )

    def forward(self, x):
        # Implement forward pass for VAE
        pass

class TS2VECEncoderProjection(BaseProjectionLayer):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()
        # Define the architecture for TS2VEC encoder
        self.encoder = nn.Sequential(
            # ...
        )

    def forward(self, x):
        # Implement forward pass for TS2VEC encoder
        pass
