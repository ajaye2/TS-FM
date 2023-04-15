from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Code from https://github.com/mims-harvard/TFC-pretraining
# ```
# @inproceedings{zhang2022self,
# title = {Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency},
# author = {Zhang, Xiang and Zhao, Ziyuan and Tsiligkaridis, Theodoros and Zitnik, Marinka},
# booktitle = {Proceedings of Neural Information Processing Systems, NeurIPS},
# year      = {2022}
# }
# ```

"""Two contrastive encoders"""

#TODO: Add more parameters to config to set model parameters
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=configs.dim_feedforward, nhead=configs.n_head, batch_first=configs.batch_first)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.num_transformer_layers)

        self.adaptive_pool_t = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            nn.Flatten(),
        )

        self.projector_t = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            # nn.Flatten(),
            nn.Linear(configs.time_output_size * configs.channel_output_size, configs.linear_encoder_dim),
            nn.BatchNorm1d(configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(configs.linear_encoder_dim, configs.encoder_layer_dims)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=configs.dim_feedforward, nhead=configs.n_head, batch_first=configs.batch_first)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, configs.num_transformer_layers)

        self.adaptive_pool_f = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            nn.Flatten(),
        )

        self.projector_f = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(configs.time_output_size, configs.channel_output_size)),
            # nn.Flatten(),
            nn.Linear(configs.time_output_size * configs.channel_output_size, configs.linear_encoder_dim),
            nn.BatchNorm1d(configs.linear_encoder_dim),
            nn.ReLU(),
            nn.Linear(configs.linear_encoder_dim, configs.encoder_layer_dims)
        )



    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = self.adaptive_pool_t(x)      #.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = self.adaptive_pool_f(f)      #.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
    
        #: fea_concat = torch.cat((z_t, z_f), dim=1)
    def encode(self, x_in_t, x_in_f):
        h_time, z_time, h_freq, z_freq = self.forward(x_in_t, x_in_f)
        embeddings                     = torch.cat((z_time, z_freq), dim=1)
        return embeddings


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred