import torch
from torch import nn
from typing import Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.mvts_transformer.ts_transformer import TSTransformerEncoder, get_pos_encoder

class TFC(nn.Module):
    """
    Two contrastive encoders.
    
    Built upon code https://github.com/mims-harvard/TFC-pretraining
    """
    def __init__(self, configs):
        """
        Initialize the TFC module.

        Args:
            configs: Configuration object containing the model parameters.
        """
        super(TFC, self).__init__()

        # encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=configs.dim_feedforward, nhead=configs.n_head, batch_first=configs.batch_first)
        # self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.num_transformer_layers)

        self.transformer_encoder_t = TSTransformerEncoder(configs.features_len, configs.TSlength_aligned, configs.d_model, configs.n_head,
                                        configs.num_transformer_layers, configs.dim_feedforward, dropout=configs.dropout,
                                        pos_encoding=configs.pos_encoding, activation=configs.transformer_activation,
                                        norm=configs.transformer_normalization_layer, freeze=configs.freeze
                                        )
        
        self.adaptive_pool_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            nn.Flatten(),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(configs.time_output_size * configs.channel_output_size, configs.linear_encoder_dim),
            nn.BatchNorm1d(configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(configs.linear_encoder_dim, configs.encoder_layer_dims)
        )

        self.transformer_encoder_f = TSTransformerEncoder(
            configs.features_len, configs.TSlength_aligned, configs.d_model, configs.n_head,
            configs.num_transformer_layers, configs.dim_feedforward, dropout=configs.dropout,
            pos_encoding=configs.pos_encoding, activation=configs.transformer_activation,
            norm=configs.transformer_normalization_layer, freeze=configs.freeze
        )

        self.adaptive_pool_f = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            nn.Flatten(),
        )

        self.projector_f = nn.Sequential(
            nn.Linear(configs.time_output_size * configs.channel_output_size, configs.linear_encoder_dim),
            nn.BatchNorm1d(configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(configs.linear_encoder_dim, configs.encoder_layer_dims)
        )

    def forward(self, x_in_t: torch.Tensor, x_in_f: torch.Tensor, padding_masks: torch.Tensor, encode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for the TFC module.

        Args:
            x_in_t: Time-based input tensor.
            x_in_f: Frequency-based input tensor.
            padding_masks: Padding masks tensor.

        Returns:
            Tuple containing h_time, z_time, h_freq, and z_freq tensors.
        """

        x = self.transformer_encoder_t(x_in_t, padding_masks)
        h_time = self.adaptive_pool_t(x)
        z_time = self.projector_t(h_time)
 
        f = self.transformer_encoder_f(x_in_f, padding_masks)
        h_freq = self.adaptive_pool_f(f)
        z_freq = self.projector_f(h_freq)

        if encode:
            embeddings = torch.cat((z_time, z_freq), dim=1)
            return embeddings

        return h_time, z_time, h_freq, z_freq

    # def encode(self, x_in_t: torch.Tensor, x_in_f: torch.Tensor, padding_masks: torch.Tensor ) -> torch.Tensor:
    #     """
    #     Encode the input tensors.

    #     Args:
    #         x_in_t: Time-based input tensor.
    #         x_in_f: Frequency-based input tensor.

    #     Returns:
    #         Embeddings tensor.
    #     """
    #     h_time, z_time, h_freq, z_freq = self.forward(x_in_t, x_in_f, padding_masks)
    #     embeddings = torch.cat((z_time, z_freq), dim=1)
    #     return embeddings




