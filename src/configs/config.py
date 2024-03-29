import torch
class Config(object):
    def __init__(self, 
                input_channels=1,
                kernel_size=8,
                stride=1,
                final_out_channels=32,
                num_classes=2,
                num_classes_target=3,
                dropout=0.35,
                features_len=24,
                features_len_f=24,
                num_epoch=40,
                beta1=0.9,
                beta2=0.99,
                lr=3e-4,
                lr_f=3e-4,
                drop_last=True,
                batch_size=32,
                target_batch_size=16,
                temperature=0.2,
                use_cosine_similarity=True,
                use_cosine_similarity_f=True,
                jitter_scale_ratio = 0.001,
                jitter_ratio = 0.001,
                max_seg = 5,
                hidden_dim = 100,
                timesteps = 10,
                TSlength_aligned = 1500,
                n_head=2,
                num_transformer_layers=2,
                linear_encoder_dim=256,
                encoder_layer_dims=128,
                dim_feedforward=128,
                channel_output_size=10,
                time_output_size=10,
                batch_first=True,
                lam  = 0.2,
                d_model=128,
                pos_encoding='learnable',
                transformer_activation='gelu',
                transformer_normalization_layer='LayerNorm', # 'BatchNorm'
                freeze=False,
                model_configs={},
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **kwargs
                ):
        # model configs
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.final_out_channels = final_out_channels  #128

        self.num_classes = num_classes
        self.num_classes_target =  num_classes_target
        self.dropout = dropout
        self.features_len = features_len 
        self.features_len_f = features_len_f  # 13 #self.features_len   # the output results in time domain

        self.TSlength_aligned = TSlength_aligned

        self.n_head = n_head
        self.num_transformer_layers = num_transformer_layers
        self.linear_encoder_dim = linear_encoder_dim
        self.encoder_layer_dims = encoder_layer_dims
        self.dim_feedforward = dim_feedforward
        self.channel_output_size = channel_output_size
        self.time_output_size = time_output_size
        self.batch_first = batch_first
        self.lam = lam
        self.d_model = d_model
        self.pos_encoding = pos_encoding
        self.transformer_activation = transformer_activation
        self.transformer_normalization_layer = transformer_normalization_layer
        self.freeze = freeze

        # training configs
        self.num_epoch = num_epoch
        
        # optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.lr_f = lr_f

        # data parameters
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.target_batch_size = target_batch_size # the size of target dataset (the # of samples used to fine-tune).

        # device
        self.device = device

        

        self.Context_Cont = Context_Cont_configs(temperature, use_cosine_similarity, use_cosine_similarity_f)
        self.TC = TC(hidden_dim, timesteps)
        self.augmentation = augmentations( jitter_scale_ratio, jitter_ratio, max_seg )

        if type (model_configs) == dict:
            self.model_configs = ModelConfig(**model_configs)
        elif type (model_configs) == ModelConfig:
            self.model_configs = model_configs
        else:
            raise ValueError('model_configs must be a dict or ModelConfig object')

        


class augmentations(object):
    def __init__(self,
                 jitter_scale_ratio = 0.001,
                 jitter_ratio = 0.001,
                 max_seg = 5
                ):
        self.jitter_scale_ratio     = jitter_scale_ratio
        self.jitter_ratio           = jitter_ratio
        self.max_seg                = max_seg


class Context_Cont_configs(object):
    def __init__(self,
                 temperature=0.2,
                 use_cosine_similarity=True,
                 use_cosine_similarity_f=True
                ):
        self.temperature             = temperature
        self.use_cosine_similarity   = use_cosine_similarity
        self.use_cosine_similarity_f = use_cosine_similarity_f


class TC(object):
    def __init__(self,
                 hidden_dim = 100,
                 timesteps = 10
                ):
        self.hidden_dim = hidden_dim
        self.timesteps  = timesteps


class ModelConfig(object):
    def __init__(self,
                 task_name='forecast',
                 enc_in=7,
                 dec_in=7,
                 c_out=7,
                 d_model=32,
                 n_heads=4,
                 e_layers=2,
                 d_layers=1,
                 d_ff=64,
                 dropout=0.1,
                 activation='relu',
                 factor=5,
                 freq='h',
                 embed='fixed',
                 output_attention=False,
                 distil=True,
                 pred_len=24,
                 label_len=24,
                 num_class=1,
                 seq_len=96,
                 use_temporal_embed=False,
                 p_hidden_dims=[128, 64, 32],
                 p_hidden_layers=3,
                 **kwargs):
        self.task_name = task_name
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.freq = freq
        self.embed = embed
        self.output_attention = output_attention
        self.distil = distil
        self.pred_len = pred_len
        self.label_len = label_len
        self.num_class = num_class
        self.seq_len = seq_len
        self.use_temporal_embed = use_temporal_embed
        self.p_hidden_dims = p_hidden_dims
        self.p_hidden_layers = p_hidden_layers
        for k, v in kwargs.items():
            setattr(self, k, v)
