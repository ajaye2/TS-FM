import torch
from torch import nn
from typing import Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from src.mvts_transformer.ts_transformer import TSTransformerEncoder, get_pos_encoder
import src.Time_Series_Library.models as models
import importlib
importlib.reload(models)

class TFC(nn.Module):
    """
    Two contrastive encoders.
    
    Built upon code https://github.com/mims-harvard/TFC-pretraining
    """
    def __init__(self, configs, type_of_encoder: str = "transformer"):
        """
        Initialize the TFC module.

        Args:
            configs: Configuration object containing the model parameters.
        """
        super(TFC, self).__init__()

        # encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=configs.dim_feedforward, nhead=configs.n_head, batch_first=configs.batch_first)
        # self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.num_transformer_layers)

        self.configs               = configs
        self.type_of_encoder       = type_of_encoder
        self.device                = configs.device
        self.init_encoder()
        



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

        if self.type_of_encoder == "transformer":
            x = self.transformer_encoder_t(x_in_t, padding_masks)
            h_time = self.adaptive_pool_t(x)
            z_time = self.projector_t(h_time)
    
            f = self.transformer_encoder_f(x_in_f, padding_masks)
            h_freq = self.adaptive_pool_f(f)
            z_freq = self.projector_f(h_freq)
        elif self.type_of_encoder == "non_stationary_transformer":

            x = self.transformer_encoder_t(x_in_t, None, None, None)
            h_time = self.adaptive_pool_t(x)
            z_time = self.projector_t(h_time)
    
            f = self.transformer_encoder_f(x_in_f, None, None, None)
            h_freq = self.adaptive_pool_f(f)
            z_freq = self.projector_f(h_freq)

        elif self.type_of_encoder == "cnn":
            x = self.conv_block1_t(x_in_t)
            x = self.conv_block2_t(x)
            x = self.conv_block3_t(x)
            h_time = x.reshape(x.shape[0], -1)
            z_time = self.projector_t(h_time)

            f = self.conv_block1_f(x_in_f)
            f = self.conv_block2_f(f)
            f = self.conv_block3_f(f)
            h_freq = f.reshape(f.shape[0], -1)
            z_freq = self.projector_f(h_freq)
        elif self.type_of_encoder == "lstm":
            x = self.lstm_t(x_in_t)
            h_time = self.adaptive_pool_t(x) # h_time = x[:, -1, :]
            z_time = self.projector_t(h_time)

            f = self.lstm_f(x_in_f)
            h_freq = self.adaptive_pool_t(x)# h_freq = f[:, -1, :]
            z_freq = self.projector_f(h_freq)

        if encode:
            embeddings = torch.cat((z_time, z_freq), dim=1)
            return embeddings

        return h_time, z_time, h_freq, z_freq

    def init_encoder(self) -> nn.Module:
        """
        Get the encoder.

        Args:
            type_of_encoder: Type of encoder to return.

        Returns:
            Encoder.
        """

        if self.type_of_encoder == "transformer":
            self.init_transformer_encoder()
        elif self.type_of_encoder == "cnn":
            self.init_conv_encoder()
        elif self.type_of_encoder == "lstm":
            self.init_lstm_encoder()
        elif self.type_of_encoder == "non_stationary_transformer":
            self.init_non_stationary_transformer_encoder()
        else:
            raise ValueError("Invalid encoder type.")

    def init_non_stationary_transformer_encoder(self):
        self.configs.task_name = "encoder"

        self.transformer_encoder_t = models.Nonstationary_Transformer(self.configs).to(self.device)

        self.adaptive_pool_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.configs.time_output_size, self.configs.channel_output_size)),
            nn.Flatten(),
        ).to(self.device)

        self.projector_t = nn.Sequential(
            nn.Linear(self.configs.time_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
            nn.BatchNorm1d(self.configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
        ).to(self.device)

        self.transformer_encoder_f = models.Nonstationary_Transformer(self.configs).to(self.device)

        self.adaptive_pool_f = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.configs.freq_output_size, self.configs.channel_output_size)),
            nn.Flatten(),
        ).to(self.device)

        self.projector_f = nn.Sequential(
            nn.Linear(self.configs.freq_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
            nn.BatchNorm1d(self.configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
        ).to(self.device)



    def init_transformer_encoder(self):
        self.transformer_encoder_t = TSTransformerEncoder(self.configs.features_len, self.configs.TSlength_aligned, self.configs.d_model, self.configs.n_head,
                                    self.configs.num_transformer_layers, self.configs.dim_feedforward, dropout=self.configs.dropout,
                                    pos_encoding=self.configs.pos_encoding, activation=self.configs.transformer_activation,
                                    norm=self.configs.transformer_normalization_layer, freeze=self.configs.freeze
                                    ).to(self.device)
    
        self.adaptive_pool_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.configs.time_output_size, self.configs.channel_output_size)),
            nn.Flatten(),
        ).to(self.device)

        self.projector_t = nn.Sequential(
            nn.Linear(self.configs.time_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
            nn.BatchNorm1d(self.configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
        ).to(self.device)
        self.transformer_encoder_f = TSTransformerEncoder(self.configs.features_len, self.configs.TSlength_aligned, self.configs.d_model, self.configs.n_head,
                                    self.configs.num_transformer_layers, self.configs.dim_feedforward, dropout=self.configs.dropout,
                                    pos_encoding=self.configs.pos_encoding, activation=self.configs.transformer_activation,
                                    norm=self.configs.transformer_normalization_layer, freeze=self.configs.freeze
                                    ).to(self.device)
    
        self.adaptive_pool_f = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(self.configs.time_output_size, self.configs.channel_output_size)),
            nn.Flatten(),
        ).to(self.device)

        self.projector_f = nn.Sequential(
            nn.Linear(self.configs.time_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
            nn.BatchNorm1d(self.configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
        ).to(self.device)        

        
    def init_conv_encoder(self):
            ### TODO: Make sure all the parameters are correct and passed in correctly
            self.conv_block1_t = nn.Sequential(
                nn.Conv1d(self.configs.input_channels, 32, kernel_size=self.configs.kernel_size,
                        stride=self.configs.stride, bias=False, padding=(self.configs.kernel_size//2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(self.configs.dropout)
            ).to(self.device)

            self.conv_block2_t = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            ).to(self.device)

            self.conv_block3_t = nn.Sequential(
                nn.Conv1d(64, self.configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(self.configs.final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            ).to(self.device)

            self.projector_t = nn.Sequential(
                nn.Linear(self.configs.CNNoutput_channel * self.configs.final_out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(self.device)

            self.conv_block1_f = nn.Sequential(
                nn.Conv1d(self.configs.input_channels, 32, kernel_size=self.configs.kernel_size,
                        stride=self.configs.stride, bias=False, padding=(self.configs.kernel_size // 2)),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(self.configs.dropout)
            ).to(self.device)

            self.conv_block2_f = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            ).to(self.device)

            self.conv_block3_f = nn.Sequential(
                nn.Conv1d(64, self.configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(self.configs.final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            ).to(self.device)

            self.projector_f = nn.Sequential(
                nn.Linear(self.configs.CNNoutput_channel * self.configs.final_out_channels, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ).to(self.device)
    def init_lstm_encoder(self):
            self.lstm_t = nn.LSTM(input_size=self.configs.input_channels, hidden_size=self.configs.hidden_size, num_layers=self.configs.num_layers, batch_first=True)
            self.adaptive_pool_t = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(self.configs.time_output_size, self.configs.channel_output_size)),
                nn.Flatten(),
            ).to(self.device)

            self.projector_t = nn.Sequential(
                nn.Linear(self.configs.time_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
                nn.BatchNorm1d(self.configs.linear_encoder_dim),
                nn.ReLU(),
                nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
            ).to(self.device)

            self.lstm_f = nn.LSTM(input_size=self.configs.input_channels, hidden_size=self.configs.hidden_size, num_layers=self.configs.num_layers, batch_first=True)
            self.adaptive_pool_f = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(self.configs.time_output_size, self.configs.channel_output_size)),
                nn.Flatten(),
            ).to(self.device)

            self.projector_f = nn.Sequential(
                nn.Linear(self.configs.time_output_size * self.configs.channel_output_size, self.configs.linear_encoder_dim),
                nn.BatchNorm1d(self.configs.linear_encoder_dim),
                nn.ReLU(),
                nn.Linear(self.configs.linear_encoder_dim, self.configs.encoder_layer_dims)
            ).to(self.device)


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


