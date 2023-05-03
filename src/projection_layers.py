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
from torch.optim.lr_scheduler import StepLR , ReduceLROnPlateau
# from src.Time_Series_Library.models import Informer, Nonstationary_Transformer, ETSformer, DLinear, TimesNet, Transformer
from src.configs import Configs, ModelConfig
import src.Time_Series_Library.models as models
import importlib
importlib.reload(models)
import time

import torch.optim.lr_scheduler as lr_scheduler

class NoamScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1, verbose=False):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        super(NoamScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self._step_count
        return [self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
                for _ in self.base_lrs]




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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, encode: bool= False) -> torch.Tensor:
        raise NotImplementedError

    def warmup(self, dataset: Union[TSDataset, ImputationDataset], max_len: int, 
                     n_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001, 
                     log: bool = False, data_set_type: Type = TSDataset, collate_fn: str = 'unsuperv', 
                     scheduler_step_size: int = 30, scheduler_gamma: float=0.1, verbose: bool = False, dataset_name="", accelerator = None, **kwargs) -> None:
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

        start_time = time.time()
        
        # Create a DataLoader for the warmup data
        # print(type(dataset))
        assert isinstance(dataset, (TSDataset, ImputationDataset))

        if collate_fn == 'unsuperv' and isinstance(dataset, ImputationDataset):
            # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: collate_unsuperv(x, max_len=max_len))
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        elif collate_fn == 'superv':
            pass
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if log:
            print(f'Warming up with {len(data_loader)} batches of size {batch_size}. Dataset name {dataset_name}. Time took {time.time() - start_time} seconds')
        
   
        # Define the loss function and optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if accelerator is not None:
            data_loader, optimizer = accelerator.prepare(data_loader, optimizer)

        # Add the learning rate scheduler
        # scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=verbose)

        # Example usage:
        # scheduler = NoamScheduler(optimizer, model_size=32, factor=1, warmup=20)

        # Keep track of the losses
        losses    = []
        
        # Train the autoencoder for n_epochs
        for epoch in range(n_epochs):
            cum_loss = 0
            for data in data_loader:
                # Get the inputs
                inputs, targets, target_masks, padding_masks, data_time_feat, label_time_feat = self.get_inputs(data, data_set_type)

                # Zero the gradients
                optimizer.zero_grad()

                if target_masks is not None:
                    target_masks =  ~target_masks

                # Forward pass
                reconstructed = self(x=inputs, mask=target_masks)

                # Calculate the loss
                loss = self.compute_loss(targets, target_masks, padding_masks, reconstructed)

                # Backward pass
                if accelerator is None:
                    loss.backward()
                else:
                    accelerator.backward(loss)

                # Update the weights
                optimizer.step()

                # Update the cumulative loss
                with torch.no_grad():
                    cum_loss += loss.item()

            # Update the learning rate
            scheduler.step(cum_loss)

            # Print the loss
            if log:
                print(f'Epoch: {epoch}, Loss: {cum_loss / len(data_loader)}')

            # Save the loss
            losses.append(cum_loss / len(data_loader))
        
        # Print the time for warmup
        if log:
            print(f'Finished warmup in {time.time() - start_time} seconds.')

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
        data_time_feat, label_time_feat              = None, None

        if data_set_type == ImputationDataset:
            inputs, targets, target_masks, padding_masks = data
        elif data_set_type == TSDataset:
            if len(data) == 3:
                inputs, targets, padding_masks = data
            else:
                inputs, targets, padding_masks, data_time_feat, label_time_feat = data
           

        if targets is not None:
            targets = targets.to(self.device)
        if target_masks is not None:
            target_masks = target_masks.to(self.device) # 1s: mask and predict, 0s: unaffected input (ignore)
        if padding_masks is not None:
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
        if inputs is not None:
            inputs = inputs.to(self.device)
        if data_time_feat is not None:
            data_time_feat = data_time_feat.to(self.device)
        if label_time_feat is not None:
            label_time_feat = label_time_feat.to(self.device)
        
        return inputs, targets, target_masks, padding_masks, data_time_feat, label_time_feat

class TransformerAutoencoderProjection(BaseProjectionLayer):
    def __init__(self, model_config: ModelConfig, device: str, type_model: str, use_revin: bool = True, dtype: torch.dtype = torch.float32, loss_type: str = 'mae', **kwargs):
        """
        Initialize the ETSformer-based autoencoder projection layer.

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
        super().__init__(type_model + "_former_encoder", loss_type, device=device, **kwargs)

        self.dtype = dtype
        self.device = device
        self.use_revin = use_revin
        self.type_model = type_model

        assert model_config.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation']
        
        if type_model == 'ets':
            self.transformer = models.ETSformer(model_config).to(device)
        elif type_model == 'non_stationary':
            self.transformer = models.Nonstationary_Transformer(model_config).to(device)
        elif type_model == 'dlinear':
            self.transformer = models.DLinear(model_config, individual=model_config.individual).to(device)
        elif type_model == 'times_net':
            self.transformer = models.TimesNet(model_config).to(device)
        elif type_model == 'transformer':
            self.transformer = models.Transformer(model_config).to(device)
        else:
            raise ValueError(f'Unknown type model: {type_model}')
        
        #TODO: Update position embedding to be learned 
        
        if use_revin:
            self.revin_layer = RevIN(model_config.enc_in).to(device)

    def forward(self, 
                x: torch.Tensor, 
                x_dec: torch.Tensor = None, 
                mask: Optional[torch.Tensor] = None, 
                encode: bool= False) -> torch.Tensor:
        """
        Implement the full forward pass for masked autoencoder (including decoder).

        Args:
            x_enc: The input tensor.
            mask: The mask to apply on the input tensor.

        Returns:
            The reconstructed tensor.
        """
        x_enc = x
        if self.use_revin:
            x_enc = self.revin_layer(x_enc, 'norm')
            if x_dec is not None:
                x_dec = self.revin_layer._normalize(x_dec)

        if self.type_model == 'ets' or self.type_model == 'transformer':
            x_decoded = self.transformer(x_enc, None, x_dec, None)
        elif self.type_model == 'non_stationary' or self.type_model == 'times_net':
            # mask = torch.ones_like(x_enc) # Change later
     
            x_dec = x_enc #torch.zeros_like(x_enc) # TODO: Change later
            x_decoded = self.transformer(x_enc, None, x_dec, None, mask)
        elif self.type_model == 'dlinear':
            x_decoded = self.transformer(x_enc, None, x_dec, None)


        
        if encode:
            return x_decoded

        if self.use_revin:
            x_decoded = self.revin_layer(x_decoded, 'denorm')

        return x_decoded


    
class LSTMMaskedAutoencoderProjection(BaseProjectionLayer):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, device: str, use_revin: bool = True, dtype: torch.dtype = torch.float32, loss_type: str = 'mae', use_gru=False, **kwargs):
        """
        Initialize the LSTM-based masked autoencoder projection layer.

        Args:
            input_dims: int containing the input dimension.
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, encode: bool= False, type_of_pooling: str="") -> torch.Tensor:
        """
        Implement the full forward pass for masked autoencoder (including decoder).

        Args:
            x: The input tensor.
            mask: The mask to apply on the input tensor.

        Returns:
            The reconstructed tensor.
        """
        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        if self.use_gru:
            enc_output, _ = self.encoder(x)
        else:
            enc_output, (_, _) = self.encoder(x)
        
        if encode:
            return self.encode(enc_output, type_of_pooling=type_of_pooling)

        if self.use_gru:
            x_decoded, _ = self.lstm_decoder(enc_output)
        else:
            x_decoded, _ = self.lstm_decoder(enc_output)

        x_decoded = self.linear_decoder(x_decoded)

        if self.use_revin:
            x_decoded = self.revin_layer(x_decoded, 'denorm')

        return x_decoded

    def encode(self, enc_output: torch.Tensor, type_of_pooling: str = '') -> torch.Tensor:
        """
        Implement forward pass for masked autoencoder.

        Args:
            x: The input tensor.
            type_of_pooling: The type of pooling to use on the output.

        Returns:
            The output tensor after applying pooling.
        """

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, encode: bool= False) -> torch.Tensor:
        """
        Implement the full forward pass for masked autoencoder (including decoder).

        Args:
            x: The input tensor.
            mask: The mask to apply on the input tensor.

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

    def encode(self, x: torch.Tensor, type_of_pooling: str = '', mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implement forward pass for masked autoencoder.

        Args:
            x: The input tensor.
            type_of_pooling: The type of pooling to use on the output.
          

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


# class TransformerEncoderProjectionLayer(BaseProjectionLayer):

#     # def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
#     #              pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
#     def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, device: str, n_heads:int,
#                   num_layers:int, max_len: int, use_revin: bool = True, dtype: torch.dtype = torch.float32, 
#                   loss_type: str = 'mae', dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False,
#                   **kwargs):
   
#         super().__init__("transformer_encoder", loss_type, device=device, **kwargs)

#         self.max_len = max_len
#         self.d_model = output_dims
#         self.n_heads = n_heads
#         self.use_revin = use_revin
#         self.dtype = dtype

#         if use_revin:
#             self.revin_layer = RevIN(input_dims).to(device)


#         self.project_inp = nn.Linear(input_dims, output_dims)
#         self.pos_enc = get_pos_encoder(pos_encoding)(output_dims, dropout=dropout*(1.0 - freeze), max_len=max_len)

#         if norm == 'LayerNorm':
#             encoder_layer = TransformerEncoderLayer(output_dims, self.n_heads, hidden_dims, dropout*(1.0 - freeze), activation=activation)
#         else:
#             encoder_layer = TransformerBatchNormEncoderLayer(output_dims, self.n_heads, hidden_dims, dropout*(1.0 - freeze), activation=activation)

#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

#         self.output_layer = nn.Linear(output_dims, input_dims)

#         self.act = _get_activation_fn(activation)

#         self.dropout1 = nn.Dropout(dropout)

#         self.feat_dim = input_dims

#     def forward(self, X, padding_masks, encode: bool= False):
#         """
#         Args:
#             X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
#             padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
#         Returns:
#             output: (batch_size, seq_length, feat_dim)
#         """

#         output = self.encode(X, padding_masks)
#         output = self.dropout1(output)
#         # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
#         output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

#         if self.use_revin:
#             output = self.revin_layer(output, 'denorm')

#         return output

#     def encode(self, X , padding_masks, type_of_pooling: str = '',):

#         if self.use_revin:
#             X = self.revin_layer(X, 'norm')

#         # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
#         inp = X.permute(1, 0, 2)
#         inp = self.project_inp(inp) * math.sqrt(
#             self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
#         inp = self.pos_enc(inp)  # add positional encoding
#         # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
#         output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
#         output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

#         if type_of_pooling == 'last':
#             output = output[:, -1, :]
#         elif type_of_pooling == 'mean':
#             output = torch.mean(output, dim=1)


#         return output



import torch
import torch.nn as nn

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)



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