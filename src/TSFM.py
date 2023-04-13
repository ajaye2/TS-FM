import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .dataloader import TSDataset, TSDataLoader
from .projection_layers import LSTMMaskedAutoencoderProjection
import numpy as np
import math

class TSFM:

    PROJECTION_LAYER_TYPES = {
        'autoencoder': LSTMMaskedAutoencoderProjection,
    }

    '''The TSFM model'''
    
    def __init__(
        self,
        input_data_shapes_dict,
        projection_layer_encoder='autoencoder',
        encoder_layer='TFC',
        projection_layer_dims=64,
        encoder_layer_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        log=False,
        dtype=torch.float32,

        max_train_length=None,
    ):
        ''' Initialize a TSFM model.
        
        Args:
            input_data_shapes_dict (dict): A dictionary containing the shapes of the input data. The keys are the names of the datasets, and the values are the shapes of the datasets. The shapes should be in the form of (n_timestamps, n_features).
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.projection_layer_dims = projection_layer_dims
        self.encoder_layer_dims = encoder_layer_dims
        self.log = log
        self.dtype = dtype

        self.projection_layers = {}
        self._projection_layers = {}
        
        for dataset_name, data_shape in input_data_shapes_dict.items():
            assert len(data_shape) == 2
            self._projection_layers[dataset_name] = self.get_projection_layer(data_shape, projection_layer_encoder)
            self.projection_layers[dataset_name] = torch.optim.swa_utils.AveragedModel(self._projection_layers[dataset_name])
            self.projection_layers[dataset_name].update_parameters(self._projection_layers[dataset_name])

        # self._encoder = Encoder(encoder_layer, encoder_layer_dims, depth).to(self.device)
        # self.encoder = torch.optim.swa_utils.AveragedModel(self._encoder)
        # self.encoder.update_parameters(self._encoder)
        

        self.n_epochs = 0
        self.n_iters = 0

    def get_projection_layer(self, data_shape, projection_layer_encoder):
        proj_layer = self.PROJECTION_LAYER_TYPES[projection_layer_encoder](data_shape, self.projection_layer_dims, self.projection_layer_dims).to(self.device)
        return proj_layer
    
    def fit(self, train_data_dict, n_epochs=None, n_iters=None, verbose=False, shuffle=True, warmup_projection_layers=True, warmup_epochs=10, log=True):
        ''' Training the TSFM model.
        
        Args:
            train_data_dict (dict): The training data. The keys are the names of the datasets, and the values are the datasets. The shapes of the datasets should be in the form of (n_samples, n_timestamps, n_features).
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        
        datasets = {}
        optimizer_list = []#[{ "params": self._encoder.parameters()}]

        for dataset_name, train_data in train_data_dict.items():
            assert train_data.ndim == 3

            # if self.max_train_length is not None:
            #     sections = train_data.shape[1] // self.max_train_length
            #     if sections >= 2:
            #         train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
            # temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
            # if temporal_missing[0] or temporal_missing[-1]:
            #     train_data = centerize_vary_length_series(train_data)
                    
            train_data             = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
            if type(train_data) == np.array:
                train_data = torch.from_numpy(train_data).to(self.dtype)
            datasets[dataset_name] = TSDataset(train_data)

            if warmup_projection_layers:
                self._projection_layers[dataset_name].warmup(datasets[dataset_name], n_epochs=warmup_epochs, batch_size=self.batch_size, learning_rate=self.lr, log=log)


            optimizer_list.append({"params": self._projection_layers[dataset_name].encoder.parameters()})

            
        train_loader    = TSDataLoader(datasets, batch_size=self.batch_size, shuffle=shuffle)
        optimizer       = torch.optim.AdamW(optimizer_list, lr=self.lr)
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
            
        loss_log = {name: [] for name in train_data_dict.keys()}


        # while True:
        #     if n_epochs is not None and self.n_epochs >= n_epochs:
        #         break
            
        #     cum_loss_dict = {name: 0 for name in train_data_dict.keys()}
        #     n_epoch_iters = 0
            
        #     interrupted = False
        #     for batch in train_loader:
        #         if n_iters is not None and self.n_iters >= n_iters:
        #             interrupted = True
        #             break

        #         for dataset_name, data in batch.items():
        #             if isinstance(data, tuple) and len(data) == 2:
        #                 # Unpack data and labels
        #                 inputs, labels = data
        #             else:
        #                 inputs = data
        #                 labels = None
                    
        #             # if self.max_train_length is not None and inputs.size(1) > self.max_train_length:
        #             #     window_offset = np.random.randint(inputs.size(1) - self.max_train_length + 1)
        #             #     inputs = inputs[:, window_offset : window_offset + self.max_train_length]

        #             inputs = inputs.to(self.device)

        #             optimizer.zero_grad()

        #             projected_input = self._projection_layers[dataset_name](inputs)

        #             out = self._encoder(projected_input)
                
        #             loss = self.loss(out, labels)
        #             loss.backward()
        #             optimizer.step()

        #             self.projection_layers[dataset_name].update_parameters(self._projection_layers[dataset_name])
        #             self.encoder.update_parameters(self.encoder)
                    
        #             cum_loss_dict[dataset_name] += loss.item()

        #         n_epoch_iters += 1
        #         self.n_iters  += 1
                
        #     if interrupted:
        #         break
            
        #     for dataset_name, cum_loss in cum_loss_dict.items():
        #         cum_loss /= n_epoch_iters
        #         loss_log[dataset_name].append(cum_loss)

        #         if verbose:
        #             print(f"Epoch #{self.n_epochs}: loss={cum_loss}")

        #     self.n_epochs += 1
    
        return loss_log


    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    