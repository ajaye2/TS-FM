import torch
import torch.nn as nn
import torch.optim as optim

from .RevIN import RevIN
from .loss import MASE, MaskedLoss
from .dataloader import TSDataLoader
from typing import Union, Type, Tuple, Optional
from .utils import ExtractTensor, mask_input, Normalizer
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredLogError
from .dataset import TSDataset, ImputationDataset, collate_unsuperv, collate_superv
from .mvts_transformer import *
from .mvts_transformer.ts_transformer import _get_activation_fn


class BaseProjectionLayer(nn.Module):
    """
    Base class for projection layers.
    """
    LOSS_FUNCTIONS = {
        'mae':  nn.L1Loss(),
        'mse':  nn.MSELoss(),
        'msle': MeanSquaredLogError(),
        'mape': MeanAbsolutePercentageError(),
        'mase': MASE(),
        'masked_mae': MaskedLoss(type_of_loss='mae'),
        'masked_mse': MaskedLoss(type_of_loss='mse'),
    }

    def __init__(self, type_of_layer: str, loss_type: str = 'mae', device: str = 'cpu', **kwargs) -> None:
        super().__init__()
        self.device = device
        self.loss_type = loss_type
        self.loss_function = self.LOSS_FUNCTIONS[loss_type]
        self.normalizer = Normalizer()
        self.type_of_layer = type_of_layer

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, training: bool = True) -> torch.Tensor:
        raise NotImplementedError

    def warmup(self, dataset: Union[TSDataset, ImputationDataset], max_len: int, n_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001, log: bool = False, data_set_type: Type = TSDataset, collate_fn: str = 'unsuperv') -> None:
        """
        Warmup function to train the autoencoder.

        Args:
            dataset: The dataset to be used for warming up.
            max_len: The maximum length of the time series.
            n_epochs: Number of epochs for training (default: 10).
            batch_size: The size of the batches (default: 64).
            learning_rate: Learning rate for the optimizer (default: 0.001).
            log: If True, print log messages during training (default: False).
            data_set_type: The dataset class (default: TSDataset).
            collate_fn: The collate function to be used (default: 'unsuperv').

        Returns:
            None.
        """
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
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        losses    = []
        # Train the autoencoder for n_epochs
        for epoch in range(n_epochs):
            cum_loss = 0
            for data in data_loader:
                # Get the inputs
                inputs, targets, target_masks, padding_masks = self.get_inputs(data, data_set_type)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed = self(inputs, padding_masks)

                # Calculate the loss
                loss = self.compute_loss(targets, target_masks, padding_masks, reconstructed)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the cumulative loss
                with torch.no_grad():
                    cum_loss += loss.item()
            if log:
                print(f'Epoch: {epoch}, Loss: {cum_loss / len(data_loader)}')
            losses.append(cum_loss / len(data_loader))

        return losses 

    def compute_loss(self, targets: torch.Tensor, target_masks: torch.Tensor, padding_masks: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the current batch.

        Args:
            targets: The ground truth values.
            target_masks: Masks for target values.
            padding_masks: Masks for padding values.
            reconstructed: The reconstructed values from the model.

        Returns:
            The computed loss for the batch.
        """
        if 'masked' in self.loss_type:
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss_per_sample = self.loss_function(y_pred=reconstructed, y_true=targets, mask=target_masks)
            loss = loss_per_sample
        else:
            loss = self.loss_function(reconstructed, targets)
        
        return loss

    def get_inputs(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], data_set_type: Type) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the input tensors from the data.

        Args:
            data: Tuple containing the input tensors.
            data_set_type: The dataset class.

        Returns:
            Tuple of input, targets, target_masks, and padding_masks tensors.
        """
        inputs, targets, target_masks, padding_masks = None, None, None, None
        if data_set_type == ImputationDataset:
            inputs, targets, target_masks, padding_masks = data
        elif data_set_type == TSDataset:
            inputs, targets, padding_masks = data
           

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
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, device: str, use_revin: bool = True, dtype: torch.dtype = torch.float32, loss_type: str = 'mae', use_gru=False, **kwargs):
        """
        Initialize the LSTM-based masked autoencoder projection layer.

        Args:
            input_dims: Tuple containing the input dimensions.
            hidden_dims: Number of hidden dimensions.
            output_dims: Number of output dimensions.
            device: The device to use for computation.
            use_revin: Whether to use the RevIN normalization layer.
            dtype: The data type to use for tensors.
            loss_type: The type of loss function to use.
            kwargs: Additional arguments for the base class.
        """
        super().__init__("lstm_encoder", loss_type, device=device, **kwargs)

        self.dtype = dtype
        self.device = device
        self.use_revin = use_revin
        self.use_gru = use_gru
        if use_revin:
            self.revin_layer = RevIN(input_dims).to(device)


        if use_gru:
            self.encoder = nn.GRU(input_size=input_dims, hidden_size=output_dims, batch_first=True).to(device)
        else:
            self.encoder = nn.LSTM(input_size=input_dims, hidden_size=output_dims, batch_first=True).to(device)

        if use_gru:
            self.lstm_decoder = nn.GRU(input_size=output_dims, hidden_size=hidden_dims, batch_first=True).to(device)
        else:
            self.lstm_decoder = nn.LSTM(input_size=output_dims, hidden_size=hidden_dims, batch_first=True).to(device)

        # self.linear_decoder = nn.Linear(hidden_dims, input_dims[1]).to(device)
        self.linear_decoder = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims//2),
            nn.ReLU(),
            nn.Linear(hidden_dims//2, input_dims)
        ).to(device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Implement the full forward pass for masked autoencoder (including decoder).

        Args:
            x: The input tensor.
            mask: The mask to apply on the input tensor.
            training: Whether the model is in training mode.

        Returns:
            The reconstructed tensor.
        """
        if self.use_revin:
            x = self.revin_layer(x, 'norm')
        if self.use_gru:
            enc_output, _ = self.encoder(x)
        else:
            enc_output, (_, _) = self.encoder(x)
        if self.use_gru:
            x_decoded, _ = self.lstm_decoder(enc_output)
        else:
            x_decoded, _ = self.lstm_decoder(enc_output)
        x_decoded = self.linear_decoder(x_decoded)
        if self.use_revin:
            x_decoded = self.revin_layer(x_decoded, 'denorm')

        return x_decoded

    def encode(self, x: torch.Tensor, type_of_pooling: str = '', mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Implement forward pass for masked autoencoder.

        Args:
            x: The input tensor.
            type_of_pooling: The type of pooling to use on the output.
            training: Whether the model is in training mode.

        Returns:
            The output tensor after applying pooling.
        """
        if self.use_revin:
            x = self.revin_layer(x, 'norm')
        enc_output, (_, _) = self.encoder(x)

        if type_of_pooling == 'last':
            enc_output = enc_output[:, -1, :]
        elif type_of_pooling == 'mean':
            enc_output = torch.mean(enc_output, dim=1)
        elif type_of_pooling == 'max':
            pass

        return enc_output



class MLPMaskedAutoencoderProjection(BaseProjectionLayer):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, device: str, use_revin: bool = True, dtype: torch.dtype = torch.float32, loss_type: str = 'mae', **kwargs):
        """
        Initialize the MLP-based masked autoencoder projection layer.

        Args:
            input_dims: Number of input dimensions.
            hidden_dims: Number of hidden dimensions.
            output_dims: Number of output dimensions.
            device: The device to use for computation.
            use_revin: Whether to use the RevIN normalization layer.
            dtype: The data type to use for tensors.
            loss_type: The type of loss function to use.
            kwargs: Additional arguments for the base class.
        """
        super().__init__("mlp_encoder", loss_type, device=device, **kwargs)

        self.dtype = dtype
        self.device = device
        self.use_revin = use_revin
        if use_revin:
            self.revin_layer = RevIN(input_dims).to(device)

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(), #nn.ReLU(), # nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, output_dims),
            # nn.ReLU()
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(output_dims, hidden_dims),
            # nn.LayerNorm(hidden_dims),
            # nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, input_dims)
        ).to(device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Implement the full forward pass for masked autoencoder (including decoder).

        Args:
            x: The input tensor.
            mask: The mask to apply on the input tensor.
            training: Whether the model is in training mode.

        Returns:
            The reconstructed tensor.
        """
        if self.use_revin:
            x = self.revin_layer(x, 'norm')
        enc_output = self.encoder(x)
        x_decoded = self.decoder(enc_output)
        if self.use_revin:
            x_decoded = self.revin_layer(x_decoded, 'denorm')

        return x_decoded

    def encode(self, x: torch.Tensor, type_of_pooling: str = '', mask: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Implement forward pass for masked autoencoder.

        Args:
            x: The input tensor.
            type_of_pooling: The type of pooling to use on the output.
            training: Whether the model is in training mode.

        Returns:
            The output tensor after applying pooling.
        """
        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        enc_output  = self.encoder(x)

        if type_of_pooling == 'last':
            enc_output = enc_output[:, -1, :]
        elif type_of_pooling == 'mean':
            enc_output = torch.mean(enc_output, dim=1)
        elif type_of_pooling == 'max':
            pass

        return enc_output


class TransformerEncoderProjectionLayer(BaseProjectionLayer):

    # def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
    #              pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, device: str, n_heads:int,
                  num_layers:int, max_len: int, use_revin: bool = True, dtype: torch.dtype = torch.float32, 
                  loss_type: str = 'mae', dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False,
                  **kwargs):
   
        super().__init__("transformer_encoder", loss_type, device=device, **kwargs)

        self.max_len = max_len
        self.d_model = output_dims
        self.n_heads = n_heads
        self.use_revin = use_revin
        self.dtype = dtype

        if use_revin:
            self.revin_layer = RevIN(input_dims).to(device)


        self.project_inp = nn.Linear(input_dims, output_dims)
        self.pos_enc = get_pos_encoder(pos_encoding)(output_dims, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(output_dims, self.n_heads, hidden_dims, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(output_dims, self.n_heads, hidden_dims, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(output_dims, input_dims)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = input_dims

    def forward(self, X, padding_masks, training: bool = True):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        output = self.encode(X, padding_masks)
        if training:
            output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        if self.use_revin:
            output = self.revin_layer(output, 'denorm')

        return output

    def encode(self, X , padding_masks, training: bool = True):

        if self.use_revin:
            X = self.revin_layer(X, 'norm')

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        return output






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


class TransposeBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(TransposeBatchNorm1d, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        return x


## Transpose conv1d
class TransposeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(TransposeConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        return x