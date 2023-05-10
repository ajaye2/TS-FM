import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .dataloader import TSDataLoader
from .dataset import TSDataset, ImputationDataset
from .projection_layers import LSTMMaskedAutoencoderProjection, MLPMaskedAutoencoderProjection, Unsqueeze
import numpy as np
import math
from .RevIN import RevIN

from src.TFC.dataloader import TFCDataset
from src.feature_transforms import GaussianFourierFeatureTransform
from .encoders import TFC
from .configs import Configs

from .TFC.augmentations import DataTransform_FD, DataTransform_TD
from .TFC.loss import NTXentLoss_poly, NTXentLoss
import torch.fft as fft
import os
from accelerate import Accelerator
import time as time_stamp
import pickle


class TSFM:
    '''The Time Series Foundation Model (TSFM) model'''

    PROJECTION_LAYER_TYPES = {
        'lstm': LSTMMaskedAutoencoderProjection,
        'mlp': MLPMaskedAutoencoderProjection,
        'gaussian_feature_transform': GaussianFourierFeatureTransform
    }
    
    def __init__(
        self,
        input_data_shapes_dict,
        model_name,
        projection_layer_encoder='lstm',
        encoder_layer='TFC',
        projection_layer_dims=64,
        depth=10,
        device='cuda',
        dtype=torch.float32,
        max_seq_length=None,
        encoder_config=None,
        univariate_forcast_hidden_dim=64,
        use_revin=True,
        univariate_criterion='mse',
        type_of_encoder='transformer',

    ):
        ''' 
        Initialize a TSFM model.
        Args:
        '''
        
        super().__init__()

        """Check if GPU is available"""
        with_gpu = torch.cuda.is_available()
        if with_gpu and device == 'cuda':
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        """Initialize the model parameters"""
        self.device = device
        self.max_seq_length = max_seq_length
        self.projection_layer_dims = projection_layer_dims
        self.encoder_layer_dims = encoder_config.encoder_layer_dims
        self.dtype = dtype
        self.encoder_layer = encoder_layer
        self.projection_layer_encoder = projection_layer_encoder
        self.depth = depth
        self.encoder_config = encoder_config
        self.n_epochs = 0
        self.n_iters = 0
        self.projection_layers = {}
        self._projection_layers = {}
        self.configs = {}
        self.use_revin = use_revin
        self.loss_dict_by_steps = {}
        self.model_name = model_name
        self.n_iters_dict = {}
        self.type_of_encoder = type_of_encoder

        #TODO: Add optionality to use accelerator
        self.accelerator = Accelerator()
        self.device      = self.accelerator.device
        #TODO: Add projection layer configs for initialization of each dataset
        """Initialize the projection layers"""
        for dataset_name, data_shape in input_data_shapes_dict.items():
            assert len(data_shape) == 2
            use_revin_ = self.use_revin
            self._projection_layers[dataset_name] = self.get_projection_layer(data_shape, projection_layer_encoder, use_revin = use_revin_).to(self.device)
            self.projection_layers[dataset_name]  = torch.optim.swa_utils.AveragedModel(self._projection_layers[dataset_name]).to(self.device)
            self.projection_layers[dataset_name].update_parameters(self._projection_layers[dataset_name])
            self.configs[dataset_name] = {}

            self._projection_layers[dataset_name], self.projection_layers[dataset_name] = self.accelerator.prepare(self._projection_layers[dataset_name], self.projection_layers[dataset_name])

        """Initialize the encoder"""
        self._encoder = self.get_encoder_layer().to(self.device)  #Encoder(encoder_layer, encoder_layer_dims, depth).to(self.device)
        self.encoder = torch.optim.swa_utils.AveragedModel(self._encoder).to(self.device)
        self.encoder.update_parameters(self._encoder)
        

        """Initialize the univariate forcaster"""
        embed_dim = encoder_config.encoder_layer_dims
        if encoder_layer == 'TFC':
            embed_dim = encoder_config.encoder_layer_dims * 2
        self._univariate_forcaster = nn.Sequential(
            nn.Linear(embed_dim, univariate_forcast_hidden_dim),
            nn.ReLU(),
            # nn.Linear(univariate_forcast_hidden_dim, max_seq_length),
            # Unsqueeze(dim=-1),
            nn.Linear(univariate_forcast_hidden_dim, max_seq_length)
        ).to(self.device)
        self.univariate_forcaster = torch.optim.swa_utils.AveragedModel(self._univariate_forcaster).to(self.device)
        self.univariate_forcaster.update_parameters(self._univariate_forcaster)

        self.univariate_revin_layer = RevIN(1).to(self.device)

        """Initialize the criterion for univariate forcasting"""
        if univariate_criterion == 'mse':
            self._univariate_forcast_criterion = nn.MSELoss()
        elif univariate_criterion == 'mae':
            self._univariate_forcast_criterion = nn.L1Loss()

        (   
            self._encoder, 
            self.encoder, 
            self._univariate_forcaster, 
            self.univariate_forcaster, 
            self._univariate_forcast_criterion, 
            self.univariate_revin_layer
        ) = self.accelerator.prepare(
            self._encoder, 
            self.encoder, 
            self._univariate_forcaster, 
            self.univariate_forcaster, 
            self._univariate_forcast_criterion, 
            self.univariate_revin_layer
        )
    
    def add_dataset(self, dataset_name, data_shape, train_data, warmup_projection_layers = True, warmup_batch_size=512, warmup_epochs = 10, log=True, lr= 1e-4, warmup_config_kwargs = {}, shuffle=True, **kwargs):
        """Add a dataset to the model"""

        if dataset_name in self.projection_layers.keys(): 
            print(f"Dataset {dataset_name} already exists")
            return
        assert len(data_shape) == 2
        self._projection_layers[dataset_name] = self.get_projection_layer(data_shape, self.projection_layer_encoder).to(self.device)
        self.projection_layers[dataset_name]  = torch.optim.swa_utils.AveragedModel(self._projection_layers[dataset_name]).to(self.device)
        self.projection_layers[dataset_name].update_parameters(self._projection_layers[dataset_name])
        self.configs[dataset_name] = {}

        self._projection_layers[dataset_name], self.projection_layers[dataset_name] = self.accelerator.prepare(self._projection_layers[dataset_name], self.projection_layers[dataset_name])
        if type(train_data) == dict:
            train_data, labels = train_data['data'], train_data['labels']
        else:
            labels = None
        assert train_data.ndim == 3
                
        """Remove the nan values"""
        # mask            = ~np.isnan(train_data).all(axis=2).all(axis=1)
        # train_data      = train_data[mask, :].astype(np.float32)
        # train_data             = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        """Convert the data to torch"""
        if type(train_data) == np.array:
            train_data = torch.from_numpy(train_data).to(self.dtype).to(self.device)
        
        """Warmup the projection layers"""
        if warmup_projection_layers and self.projection_layer_encoder != 'gaussian_feature_transform':
            batch_size = warmup_batch_size
            if batch_size > train_data.shape[0]:
                batch_size = train_data.shape[0] // 20

            warmup_config = warmup_config_kwargs[dataset_name]
            dataset = warmup_config['data_set_type'](train_data, labels=labels, shuffle=shuffle, **kwargs)
            n_epochs = warmup_epochs if 'n_epochs' not in warmup_config else warmup_config['n_epochs']
            batch_s  = batch_size if 'batch_size' not in warmup_config else warmup_config['batch_size']
            pl_kwargs = warmup_config['kwargs'] if 'kwargs' in warmup_config else {}
            pl_lr    = lr if 'lr' not in warmup_config else warmup_config['lr']
            self._projection_layers[dataset_name].warmup(dataset, n_epochs=n_epochs, batch_size=batch_s, learning_rate=pl_lr, 
                                                            log=log, data_set_type=warmup_config['data_set_type'], 
                                                            collate_fn='unsuperv', max_len=self.max_seq_length, 
                                                            dataset_name=dataset_name, accelerator=self.accelerator,
                                                            **pl_kwargs
                                                            ) 


    #TODO: Implement logging of loss in database
    def fit(self, train_data_dict, labels=None, lr=1e-2, freeze_proj_layers=False, n_epochs=None, batch_size=512, n_iters=None, verbose=False, shuffle=True, warmup_projection_layers=True, warmup_epochs=10, log=True, subset=False, configs=None, training_mode='pre_train', warmup_config_kwargs=None,  warmup_batch_size=512, print_every_iter=10000, **kwargs):

        """Get the total number of data points"""
        total_number_of_data_points = 0
        all_dataset_names = ""
        for dataset_name, dataset in train_data_dict.items():
            if type(dataset) == dict:
                total_number_of_data_points += dataset['data'].shape[0]
                total_number_of_data_points += dataset['labels'].shape[0]
            else:
                total_number_of_data_points += dataset.shape[0]
            all_dataset_names += dataset_name + ', '
        print(f'Total number of data points: {total_number_of_data_points} from {all_dataset_names[:-2]}')

        """Warmup the projection layers"""
        datasets, optimizer_list, encoder_dataset_type = self.warmup(train_data_dict, freeze_proj_layers, warmup_projection_layers=warmup_projection_layers, 
                                                                     warmup_epochs=warmup_epochs, shuffle=shuffle, 
                                                                     warmup_config_kwargs=warmup_config_kwargs, warmup_batch_size=warmup_batch_size, 
                                                                     lr=lr, **kwargs
                                                                     )
        
        """Disable revIN for univariate forcasting"""
        if 'univariate' in self.projection_layers.keys():
            self._projection_layers['univariate'].use_revin = False

        """Initialize the optimizer and the data loader"""
        train_loader    = TSDataLoader(datasets, batch_size=batch_size, shuffle=shuffle, max_len=self.max_seq_length)
        optimizer       = torch.optim.AdamW(optimizer_list, lr=lr)

        optimizer, train_loader = self.accelerator.prepare(optimizer, train_loader)
        
        """Set the number of iterations"""
        if n_iters is None and n_epochs is None:
            n_iters = 200 if total_number_of_data_points <= 100000 else 600  # default param for n_iters
        
        """Initialize the loss dict"""
        loss_dict = {name: [] for name in train_data_dict.keys() }
        self.loss_dict_by_steps = {name: [] for name in train_data_dict.keys() }
        self.n_iters_dict = {name: 0 for name in train_data_dict.keys()}

        """Initialize the time"""
        start_time = time_stamp.time()

        if freeze_proj_layers:
            for proj_layer in self.projection_layers.values():
                for param in proj_layer.parameters():
                    param.requires_grad = False

        """Start training"""
        while True:
            """Check if the number of epochs is reached"""
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            """Initialize the loss dict and the number of epoch iterations"""
            cum_loss_dict = {name: 0 for name in train_data_dict.keys()}
            n_epoch_iters = 0
            interrupted   = False

            """Iterate over the batches"""
            for batch in train_loader:

                """Check if the number of iterations is reached"""
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                """Train Step"""
                for dataset_name, data in batch.items():
                    loss = self.train_step(data, dataset_name, optimizer, encoder_dataset_type, batch_size)
                    cum_loss_dict[dataset_name] += loss.item()
                    self.loss_dict_by_steps[dataset_name].append(loss.item())

                    if self.n_iters_dict[dataset_name] % print_every_iter == 0:
                        print(f'Epoch #{self.n_epochs}, Iter #{self.n_iters_dict[dataset_name]}: loss={loss.item()} for {dataset_name}')
                

                    self.n_iters_dict[dataset_name] += 1

                avg_n_iters = int( np.mean(list(self.n_iters_dict.values())) )
                # min_n_iters = min(list(self.n_iters_dict.values()))
                
                if avg_n_iters % print_every_iter <= 10:
                    model_path = f'./models/{self.model_name}/iter_{avg_n_iters}/'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    self.save(model_path)
                    
                n_epoch_iters += 1
                self.n_iters  += 1
                
                        
            """Compute the average loss"""
            for dataset_name, cum_loss in cum_loss_dict.items():
                cum_loss /= n_epoch_iters
                loss_dict[dataset_name].append(cum_loss)
                if verbose: print(f"Epoch #{self.n_epochs}: loss={cum_loss} for {dataset_name}")

            """Update the number of epochs"""
            self.n_epochs += 1
            if interrupted: break
        
        """Save the model"""
        model_path = f'./models/{self.model_name}/final/'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save(model_path)

        """Print the time"""
        print(f'Training time: {time_stamp.time() - start_time} seconds')
    
        return loss_dict
    
    def warmup(self, train_data_dict, freeze_proj_layers, labels=None, lr=1e-2, n_epochs=None, n_iters=None, verbose=False, shuffle=True, warmup_projection_layers=True, warmup_epochs=10, log=True, subset=False, configs=None, training_mode='pre_train', warmup_config_kwargs=None, warmup_batch_size=512, **kwargs):

        """Initialize data set dict and optimizer list"""
        datasets       = {}
        optimizer_list = [{ "params": self._encoder.parameters()}]

        for dataset_name, train_data in train_data_dict.items():
            if type(train_data) == dict:
                train_data, labels = train_data['data'], train_data['labels']
            else:
                labels = None
            assert train_data.ndim == 3
                    
            """Remove the nan values"""
            # mask            = ~np.isnan(train_data).all(axis=2).all(axis=1)
            # train_data      = train_data[mask, :].astype(np.float32)
            # train_data             = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

            """Convert the data to torch"""
            if type(train_data) == np.array:
                train_data = torch.from_numpy(train_data).to(self.dtype).to(self.device)
            
            """Warmup the projection layers"""
            if warmup_projection_layers and self.projection_layer_encoder != 'gaussian_feature_transform':
                batch_size = warmup_batch_size
                if batch_size > train_data.shape[0]:
                    batch_size = train_data.shape[0] // 20

                warmup_config = warmup_config_kwargs[dataset_name]
                dataset = warmup_config['data_set_type'](train_data, labels=labels, shuffle=shuffle, **kwargs)
                n_epochs = warmup_epochs if 'n_epochs' not in warmup_config else warmup_config['n_epochs']
                batch_s  = batch_size if 'batch_size' not in warmup_config else warmup_config['batch_size']
                pl_kwargs = warmup_config['kwargs'] if 'kwargs' in warmup_config else {}
                pl_lr    = lr if 'lr' not in warmup_config else warmup_config['lr']
                self._projection_layers[dataset_name].warmup(dataset, n_epochs=n_epochs, batch_size=batch_s, learning_rate=pl_lr, 
                                                             log=log, data_set_type=warmup_config['data_set_type'], 
                                                             collate_fn='unsuperv', max_len=self.max_seq_length, 
                                                             dataset_name=dataset_name, accelerator=self.accelerator,
                                                             **pl_kwargs
                                                             ) 

            """Initialize datasets"""
            encoder_dataset_type = TSDataset
            if self.encoder_layer == 'TFC':
                if configs is None or dataset_name not in configs:
                    if warmup_config_kwargs is None or dataset_name not in warmup_config_kwargs:
                        raise ValueError('Either configs or config_kwargs must be specified.')
                    config = Configs(**warmup_config_kwargs[dataset_name])
                else:
                    config     = configs[dataset_name]

                self.configs[dataset_name] = config
                datasets[dataset_name]     = TSDataset(train_data, max_len=self.max_seq_length)

                # if labels is None:
                # labels = torch.zeros((train_data.shape[0], 1)) # create labels as dummy array with shape (n_samples, 1), Labels are not used in pre-training, but are required for the TFCDataset class
                # ds = {"samples": train_data, "labels": labels}
                # datasets[dataset_name] = TFCDataset(ds, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset)
            else:
                raise NotImplementedError(f'Encoder {self.encoder_layer} is not implemented yet.')
            
            """Add the projection layers parameters to the optimizer list"""
            if self.projection_layer_encoder != 'gaussian_feature_transform' and freeze_proj_layers is False:
                optimizer_list.append({"params": self._projection_layers[dataset_name].parameters()})
            
        
        return datasets, optimizer_list, encoder_dataset_type

    def train_step(self, data, dataset_name, optimizer, data_set_type, batch_size):

        """Set the gradients to zero"""
        optimizer.zero_grad()

        """Train Step"""
        if self.encoder_layer == 'TFC':
            loss = self._train_step_TFC(data, dataset_name, data_set_type, batch_size)

        """Backpropagation"""
        # loss.backward()
        self.accelerator.backward(loss)
        optimizer.step()

        """Update the parameters"""
        self.projection_layers[dataset_name].update_parameters(self._projection_layers[dataset_name])
        self.encoder.update_parameters(self.encoder)
        
        return loss

    def _train_step_TFC(self, batch, dataset_name, data_set_type, batch_size):

        """Get the inputs"""
        config                              = self.configs[dataset_name]
        data_inputs                         = self.get_inputs(batch, data_set_type, dataset_name)
        inputs                              = data_inputs[0]
        targets                             = data_inputs[1]
        target_masks                        = data_inputs[2]
        padding_masks                       = data_inputs[3]
        data_time_feat                      = data_inputs[4]
        label_time_feat                     = data_inputs[5]
        
        """Get the projections"""
        x_data                              = self._projection_layers[dataset_name](inputs, encode=True)
        x_data_f                            = fft.fft(x_data).abs()
        aug1                                = DataTransform_TD(x_data, config) # TODO: Check if transformation is done on the time dimension or the feature dimension
        aug1_f                              = DataTransform_FD(x_data_f, config)

        """Get the embeddings"""
        h_t, z_t, h_f, z_f                  = self._encoder(x_data, x_data_f, padding_masks)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug  = self._encoder(aug1, aug1_f, padding_masks)

        """Compute the univariate loss"""
        univariate_forcast_loss = 0
        # if dataset_name == 'univariate':
            
        #     # print(z_t.shape, z_f.shape, z_t_aug.shape, z_f_aug.shape)
        #     embeddings       = torch.cat((z_t, z_f), dim=1)
        #     embeddings_aug   = torch.cat((z_t_aug, z_f_aug), dim=1)

        #     uni_forcast      = self._univariate_forcaster(embeddings).unsqueeze(-1)
        #     uni_forcast_aug  = self._univariate_forcaster(embeddings_aug).unsqueeze(-1)

        #     # print(uni_forcast.shape)

        #     uni_forcast      = self.univariate_revin_layer(uni_forcast, 'denorm')
        #     uni_forcast_aug  = self.univariate_revin_layer(uni_forcast_aug, 'denorm')

        #     univariate_forcast_loss = self._univariate_forcast_criterion(uni_forcast, targets) + self._univariate_forcast_criterion(uni_forcast_aug, targets)

        """Compute the context contrastive loss - NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion                   = NTXentLoss_poly(self.device, batch_size, config.Context_Cont.temperature,
                                                config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

        loss_t                              = nt_xent_criterion(h_t, h_t_aug)
        loss_f                              = nt_xent_criterion(h_f, h_f_aug)
        l_TF                                = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

        l_1, l_2, l_3                       = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c                              = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam                                 = self.encoder_config.lam
        loss                                = lam*(loss_t + loss_f) + l_TF
        loss                                = loss + univariate_forcast_loss

        return loss


    def get_inputs(self, data, data_set_type, dataset_name):

        """Get the inputs"""
        inputs, targets, target_masks, padding_masks = None, None, None, None
        data_time_feat, label_time_feat              = None, None

        if data_set_type == ImputationDataset:
            inputs, targets, target_masks, padding_masks = data
        elif data_set_type == TSDataset:
            if len(data) == 3:
                inputs, targets, padding_masks = data
            else:
                inputs, targets, padding_masks, data_time_feat, label_time_feat = data
        
        """Move the inputs to the device"""
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

        """Normalize the univariate data"""
        if dataset_name == 'univariate':
            inputs = self.univariate_revin_layer(inputs, 'norm')
        
        return inputs, targets, target_masks, padding_masks, data_time_feat, label_time_feat
    

    def get_encoder_layer(self):
        if self.encoder_layer == 'TFC':
            encoder = TFC(self.encoder_config, self.type_of_encoder).to(self.device)
        else:
            raise NotImplementedError(f'Encoder layer {self.encoder_layer} is not implemented.')
        
        return encoder

    def get_projection_layer(self, data_shape, projection_layer_encoder, use_revin=True, loss_type='mae', use_gru=False, **kwargs):
        ts_length  = data_shape[0]
        input_dims = data_shape[1]

        if projection_layer_encoder == "gaussian_feature_transform":
            proj_layer = self.PROJECTION_LAYER_TYPES[projection_layer_encoder]( input_dims, self.projection_layer_dims, [1, 10, 100, 1000] )
            return proj_layer
                            
        proj_layer = self.PROJECTION_LAYER_TYPES[projection_layer_encoder](input_dims=input_dims, 
                                                                           hidden_dims=self.projection_layer_dims, 
                                                                           output_dims=self.projection_layer_dims, 
                                                                           device=self.device, 
                                                                           use_revin=use_revin,
                                                                           loss_type=loss_type,
                                                                           use_gru=use_gru,
                                                                           **kwargs
                                                                           ).to(self.device)
        return proj_layer
    
    
    def encode(self, data, batch_size, dataset_name, encoding_window=None):
        ''' Compute representations using the model.
        
        Args:

        Returns:
            repr: The representations for data.
        '''
        assert self.encoder is not None, 'please train or load a encoder first'
        assert data.ndim == 3
 
        self.encoder.eval()
        self.projection_layers[dataset_name].eval()
        
        if type(data) == np.ndarray:
            data              = torch.from_numpy(data).to(torch.float32)
        dataset           = TSDataset(data, shuffle=False)
        loader            = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
        representations   = []
        
        with torch.no_grad():
            for batch in loader:
                data_inputs                         = self.get_inputs(batch, TSDataset, dataset_name)
                inputs                              = data_inputs[0]
                targets                             = data_inputs[1]
                target_masks                        = data_inputs[2]
                padding_masks                       = data_inputs[3]
                data_time_feat                      = data_inputs[4]
                label_time_feat                     = data_inputs[5]


                repr                          = self._eval_with_pooling(inputs, dataset_name, padding_masks=padding_masks, encoding_window=encoding_window)

                representations.append(repr)
            
        self.encoder.train()
        self.projection_layers[dataset_name].train()

        representations = torch.cat(representations, dim=0)

        return representations
    
    def _eval_with_pooling(self, x, dataset_name, padding_masks=None, encoding_window=None):
       
        if self.encoder_layer == 'TFC':
            if dataset_name == 'univariate':
                x = self.univariate_revin_layer(x, 'norm')
            projection                  = self.projection_layers[dataset_name](x, encode=True)
            projection_f                = fft.fft(projection).abs()
            out                         = self.encoder(projection, projection_f, padding_masks=padding_masks, encode=True)
        else:
            raise NotImplementedError(f'Encoder layer {self.encoder_layer} is not implemented.')

        # TODO: Implement slicing and encoding_window # out = self.encoder(x.to(self.device, non_blocking=True), mask)
        # if encoding_window == 'full_series':
        #     pass
        # elif isinstance(encoding_window, int):
        #     pass
        # elif encoding_window == 'multiscale':
        #     pass
        # else:
        #     pass
            
        return out.cpu()

    # TODO: Implement saving model to s3
    def save(self, fn, projection_layers=True, encoder=True):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        if fn[-4:] == '.pkl':
            fn = fn[:-4]

        if projection_layers:
            for dataset_name, projection_layer in self.projection_layers.items():
                torch.save(projection_layer.state_dict(), fn + "_projection_layer_{}.pkl".format(dataset_name)) 

        if encoder:
            torch.save(self.encoder.state_dict(), fn + "_encoder.pkl")

        
        with open(f'{fn}_loss_dict_by_steps.pkl', 'wb') as f:
            pickle.dump(self.loss_dict_by_steps, f)
        # pickle self.n_iters_dict
         
    
    def load(self, fn, projection_layers=True, encoder=True):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        if fn[-4:] == '.pkl':
            fn = fn[:-4]
        
        if projection_layers:
            for dataset_name, projection_layer in self.projection_layers.items():
                # check if fn + "_projection_layer_{}.pkl".format(dataset_name), exists
                if not os.path.exists(fn + "_projection_layer_{}.pkl".format(dataset_name)):
                    print(f"Projection layer for {dataset_name} does not exist")
                    continue
                state_dict = torch.load(fn + "_projection_layer_{}.pkl".format(dataset_name), map_location=self.device)
                projection_layer.load_state_dict(state_dict)

        if encoder:
            state_dict = torch.load(fn + "_encoder.pkl", map_location=self.device)
            self.encoder.load_state_dict(state_dict)

        
        with open(f'{fn}_loss_dict_by_steps.pkl', 'rb') as f:
            self.loss_dict_by_steps = pickle.load(f)
        