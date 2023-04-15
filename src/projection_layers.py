import torch
import torch.nn as nn
import torch.optim as optim

from .RevIN import RevIN
from .loss import MASE, MaskedLoss
from .dataloader import TSDataLoader
from .utils import ExtractTensor, mask_input, Normalizer
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredLogError
from .dataset import TSDataset, ImputationDataset, collate_unsuperv, collate_superv


class BaseProjectionLayer(nn.Module):
    LOSS_FUNCTIONS = {
        'mae':  nn.L1Loss(),
        'mse':  nn.MSELoss(),
        'msle': MeanSquaredLogError(),
        'mape': MeanAbsolutePercentageError(),
        'mase': MASE(),
        'masked_mae': MaskedLoss(type_of_loss='mae'),
        'masked_mse': MaskedLoss(type_of_loss='mse'),

    }
    def __init__(self, type_of_layer, lose_type='mae', device='cpu', **kwargs):
        super().__init__()
        self.device         = device
        self.lose_type      = lose_type
        self.loss_function  = self.LOSS_FUNCTIONS[lose_type]
        self.normalizer     = Normalizer()
        self.type_of_layer  = type_of_layer

    def forward(self, x, mask=None):
        raise NotImplementedError

    def warmup(self, dataset, max_len, n_epochs=10, batch_size=64, learning_rate=0.001, log=False, data_set_type=TSDataset, collate_fn='unsuperv'):
        # Create a DataLoader for the warmup data
        assert isinstance(dataset, (TSDataset, ImputationDataset))

        if collate_fn == 'unsuperv' and isinstance(dataset, ImputationDataset):
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: collate_unsuperv(x, max_len=max_len))
        elif collate_fn == 'superv':
            pass
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if log:
            print(f'Warming up with {len(data_loader)} batches of size {batch_size}')

        # Define the loss function and optimizer
        
        optimizer     = optim.Adam(self.parameters(), lr=learning_rate)
        mask          = None

        # Train the autoencoder for n_epochs
        for epoch in range(n_epochs):
            cum_loss = 0
            for data in data_loader:
                # Get the inputs
                inputs, targets, target_masks, padding_masks = self.get_inputs(data, data_set_type)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed = self(inputs, mask)

                # Calculate the loss
                loss = self.compute_loss(targets, target_masks, padding_masks, reconstructed) #self.loss_function(reconstructed, inputs) # self.loss_function(self.normalizer.forward(reconstructed), self.normalizer.forward(inputs))

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the cumulative loss
                with torch.no_grad():
                    cum_loss += loss.item()
            if log:
                print(f'Epoch: {epoch}, Loss: {cum_loss / len(data_loader)}')

    def compute_loss(self, targets, target_masks, padding_masks, reconstructed):
        if 'masked' in self.lose_type:
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss_per_sample = self.loss_function(y_pred=reconstructed, y_true=targets, mask=target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            # batch_loss = torch.sum(loss_per_sample)
            # loss = batch_loss / len(loss_per_sample)  # mean loss (over active elements) used for optimization
            loss = loss_per_sample
        else:
            loss = self.loss_function(reconstructed, targets)
        
        return loss

    def get_inputs(self, data, data_set_type):
        inputs, targets, target_masks, padding_masks = None, None, None, None
        if data_set_type == ImputationDataset:
            inputs, targets, target_masks, padding_masks = data
        elif data_set_type == TSDataset:
            if len(data) == 2: inputs, targets = data
            else: inputs, targets = data, data

        if targets is not None:
            targets = targets.to(self.device)
        if target_masks is not None:
            target_masks = target_masks.to(self.device) # 1s: mask and predict, 0s: unaffected input (ignore)
        if padding_masks is not None:
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
        if inputs is not None:
            inputs = inputs.to(self.device)
        
        return inputs, targets, target_masks, padding_masks


    
class LSTMMaskedAutoencoderProjection(BaseProjectionLayer):
    def __init__(self, input_dims, hidden_dims, output_dims, device, use_revin=True, dtype=torch.float32, lose_type='mae', **kwargs):
        super().__init__("lstm_encoder", lose_type, device=device, **kwargs)

        self.dtype = dtype
        self.device = device
        self.use_revin = use_revin
        if use_revin:
            self.revin_layer = RevIN(input_dims[1])
        

        self.encoder   = nn.LSTM(input_size=input_dims[1], hidden_size=output_dims, batch_first=True) #Conv1DLSTMProjectionEncoder( input_dims, output_dims, **kwargs) #

        self.lstm_decoder   = nn.LSTM(input_size=output_dims, hidden_size=hidden_dims, batch_first=True)
        self.linear_decoder = nn.Linear(hidden_dims, input_dims[1])
       


    def forward(self, x, mask=None, training=True):
        # Implement the full forward pass for masked autoencoder (including decoder)
        # if training:  x                           = mask_input(x)

        if self.use_revin:
            x                       = self.revin_layer(x, 'norm')
        enc_output, (_, _)          = self.encoder(x)
        x_decoded, _                = self.lstm_decoder(enc_output)
        x_decoded                   = self.linear_decoder(x_decoded)
        if self.use_revin:
            x_decoded               = self.revin_layer(x_decoded, 'denorm')
       
        return x_decoded
    
    def encode(self, x, type_of_pooling='', training=True):
        # Implement forward pass for masked autoencoder
        # For the TSFM model, we only need the embeddings, so we only return the output of the encoder.
        # if training:  x              = mask_input(x)

        if self.use_revin:
            x              = self.revin_layer(x, 'norm')
        enc_output, (_, _) = self.encoder(x) 

        if type_of_pooling=='last':
            enc_output = enc_output[:, -1, :]
        elif type_of_pooling=='mean':
            enc_output = torch.mean(enc_output, dim=1)
        elif type_of_pooling=='max':
            pass

        return enc_output
    






class Conv1DLSTMProjectionEncoder(BaseProjectionLayer):
    def __init__(self, input_dims, output_dims, out_channels=12, kernel_size=5, padding=1, **kwargs):
        super().__init__("conv1d_encoder")
        # Define the architecture for Conv1D encoder

        self.con1d  = nn.Conv1d(input_dims[1], out_channels, kernel_size=kernel_size, padding=padding)

        self.lstm   = nn.LSTM(input_size=out_channels, hidden_size=output_dims, batch_first=True)

    def forward(self, x):
        # Forward pass for Conv1D encoder
        x = x.permute(0, 2, 1)
        x = self.con1d(x)
        x = x.permute(0, 2, 1)
        
        return self.lstm(x) 


class VAEProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__("vae")
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
        super().__init__("ts2vec")
        # Define the architecture for TS2VEC encoder
        self.encoder = nn.Sequential(
            # ...
        )

    def forward(self, x):
        # Implement forward pass for TS2VEC encoder
        pass
