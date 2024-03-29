import numpy as np
from torch.utils.data import Dataset
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing.dummy as multiprocessing


### Code from:
### George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, 
    # @inproceedings{10.1145/3447548.3467401,
    # author = {Zerveas, George and Jayaraman, Srideepika and Patel, Dhaval and Bhamidipaty, Anuradha and Eickhoff, Carsten},
    # title = {A Transformer-Based Framework for Multivariate Time Series Representation Learning},
    # year = {2021},
    # isbn = {9781450383325},
    # publisher = {Association for Computing Machinery},
    # address = {New York, NY, USA},
    # url = {https://doi.org/10.1145/3447548.3467401},
    # doi = {10.1145/3447548.3467401},
    # booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining},
    # pages = {2114–2124},
    # numpages = {11},
    # keywords = {regression, framework, multivariate time series, classification, transformer, deep learning, self-supervised learning, unsupervised learning, imputation},
    # location = {Virtual Event, Singapore},
    # series = {KDD '21}
    # }


# class TSDataset(Dataset):
#     def __init__(self, data, labels=None, max_len=None, **kwargs):
#         self.data = data
#         self.labels = labels
#         self.labels_passed = labels is not None
#         self.max_len = data.shape[1] if max_len is None else max_len
    

#     def __getitem__(self, index):
#         length = [self.data[index].shape[0]]
#         padding_masks = padding_mask(torch.tensor(length, dtype=torch.int16), max_len=self.max_len) 
#         padding_masks = padding_masks.squeeze(0)
#         labels = self.labels[index] if self.labels_passed else self.data[index] # if labels are not passed, use the data as labels
#         return self.data[index], labels, padding_masks

#     def __len__(self):
#         return len(self.data)

class TSDataset(Dataset):
    def __init__(self, data, labels=None, data_time_feat=None, label_time_feat=None, max_len=None, shuffle=False, **kwargs):
        self.data = data
        self.labels = labels
        self.labels_passed = labels is not None
        self.max_len = data.shape[1] if max_len is None else max_len
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.data_time_feat = data_time_feat
        self.label_time_feat = label_time_feat

        if self.shuffle:
            self.shuffle_indices()

    def __getitem__(self, index):
        shuffled_index = self.indices[index]
        length = [self.data[shuffled_index].shape[0]]
        padding_masks = padding_mask(torch.tensor(length, dtype=torch.int16), max_len=self.max_len) 
        padding_masks = padding_masks.squeeze(0)
        labels = self.labels[shuffled_index] if self.labels_passed else self.data[shuffled_index] 

        if self.data_time_feat is not None:
            data_time_feat = self.data_time_feat[shuffled_index]
            label_time_feat = self.label_time_feat[shuffled_index]
            return self.data[shuffled_index], labels, padding_masks, data_time_feat, label_time_feat
        
        return self.data[shuffled_index], labels, padding_masks

    def __len__(self):
        return len(self.data)

    def shuffle_indices(self):
        np.random.shuffle(self.indices)

def preprocess_sample(args):
    X, masking_ratio, mean_mask_length, mode, distribution, exclude_feats = args
    mask = noise_mask(X, masking_ratio, mean_mask_length, mode, distribution, exclude_feats)
    X, mask = X, torch.from_numpy(mask)
    return (X, mask)

class ImputationDataset(Dataset):
    def __init__(self, data, labels=None, mean_mask_length=3, masking_ratio=0.75,
                 mode='separate', distribution='geometric', exclude_feats=None, max_len=None,
                 mask_compensation=False, pad_inputs=False, mask_inputs=True, **kwargs):
        super(ImputationDataset, self).__init__()

        self.data = data
        self.labels = labels
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats
        self.max_len = max_len
        self.mask_compensation = mask_compensation
        self.pad_inputs = pad_inputs
        self.mask_inputs = mask_inputs

        # Preprocess the data by applying collate_unsuperv function for each sample
        self.preprocessed_data = []
        data_with_masks = []
        for ind in range(len(data)):
            X = data[ind]
            mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution,
                              self.exclude_feats)
            data_with_masks.append((X, torch.from_numpy(mask)))

        X, targets, target_masks, padding_masks = collate_unsuperv(data_with_masks, self.max_len, self.mask_compensation,
                                                                   self.pad_inputs, self.mask_inputs)
        for i in range(len(data)):
            preprocessed_sample = (X[i], targets[i], target_masks[i], padding_masks[i])
            self.preprocessed_data.append(preprocessed_sample)

    def __getitem__(self, ind):
        return self.preprocessed_data[ind]

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.data)

class TransductionDataset(Dataset):

    def __init__(self, data, indices, mask_feats, start_hint=0.0, end_hint=0.0):
        super(TransductionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]

        self.mask_feats = mask_feats  # list/array of indices corresponding to features to be masked
        self.start_hint = start_hint  # proportion at beginning of time series which will not be masked
        self.end_hint = end_hint  # end_hint: proportion at the end of time series which will not be masked

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        mask = transduct_mask(X, self.mask_feats, self.start_hint,
                              self.end_hint)  # (seq_length, feat_dim) boolean array

        return torch.from_numpy(X), torch.from_numpy(mask), self.IDs[ind]

    def update(self):
        self.start_hint = max(0, self.start_hint - 0.1)
        self.end_hint = max(0, self.end_hint - 0.1)

    def __len__(self):
        return len(self.IDs)


def collate_superv(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs


class ClassiregressionDataset(Dataset):

    def __init__(self, data, indices):
        super(ClassiregressionDataset, self).__init__()

        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]

        self.labels_df = self.data.labels_df.loc[self.IDs]

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        y = self.labels_df.loc[self.IDs[ind]].values  # (num_labels,) array

        return torch.from_numpy(X), torch.from_numpy(y), self.IDs[ind]

    def __len__(self):
        return len(self.IDs)


def transduct_mask(X, mask_feats, start_hint=0.0, end_hint=0.0):
    """
    Creates a boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        mask_feats: list/array of indices corresponding to features to be masked
        start_hint:
        end_hint: proportion at the end of time series which will not be masked

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """

    mask = np.ones(X.shape, dtype=bool)
    start_ind = int(start_hint * X.shape[0])
    end_ind = max(start_ind, int((1 - end_hint) * X.shape[0]))
    mask[start_ind:end_ind, mask_feats] = 0

    return mask


def compensate_masking(X, mask):
    """
    Compensate feature vectors after masking values, in a way that the matrix product W @ X would not be affected on average.
    If p is the proportion of unmasked (active) elements, X' = X / p = X * feat_dim/num_active
    Args:
        X: (batch_size, seq_length, feat_dim) torch tensor
        mask: (batch_size, seq_length, feat_dim) torch tensor: 0s means mask and predict, 1s: unaffected (active) input
    Returns:
        (batch_size, seq_length, feat_dim) compensated features
    """

    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def collate_unsuperv(data, max_len=None, mask_compensation=False, pad_inputs=False, mask_inputs=True, ):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, mask).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - mask: boolean torch tensor of shape (seq_length, feat_dim); variable seq_length.
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 ignore (padding)
    """

    batch_size = len(data)
    features, masks = zip(*data) # ID REMOVED

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    max_len_orig = max(lengths)
    if max_len is None:
        max_len =max_len_orig

    # if max_len != max_len_orig:
    #     X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    #     target_masks = torch.zeros_like(X,
    #                                     dtype=torch.bool)  # (batch_size, padded_length, feat_dim) masks related to objective
    #     for i in range(batch_size):
    #         end = min(lengths[i], max_len)
    #         X[i, :end, :] = features[i][:end, :]
    #         target_masks[i, :end, :] = masks[i][:end, :]
    # else:
    X = torch.stack(features, dim=0)  # (batch_size, seq_length, feat_dim)
    target_masks = torch.stack(masks, dim=0)  # (batch_size, seq_length, feat_dim)

    targets = X.clone()
    if mask_inputs:
        X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks  # ID REMOVED


#TODO: Test for correctness
def collate_unsuperv_optimized(data, max_len=None, mask_compensation=False):
    batch_size = len(data)
    features, masks = zip(*data)

    # Get max_len if not provided
    if max_len is None:
        max_len = max([X.shape[0] for X in features])

    # Stack features and masks
    lengths = torch.tensor([min(X.shape[0], max_len) for X in features], dtype=torch.int64)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    target_masks = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.bool)

    # Index tensor for batch dimension
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, max_len)

    # Index tensor for sequence length dimension
    seq_idx = torch.arange(max_len).expand(batch_size, -1)

    # Create boolean tensor with True values for the valid indices
    valid_idx = seq_idx < lengths.unsqueeze(1)

    # Assign features and masks to the resulting tensors using valid indices
    X[batch_idx, seq_idx, :] = torch.stack([X_i[:end, :] for X_i, end in zip(features, lengths.tolist())])[valid_idx.unsqueeze(-1)].view(-1, features[0].shape[-1])
    target_masks[batch_idx, seq_idx, :] = torch.stack([masks_i[:end, :] for masks_i, end in zip(masks, lengths.tolist())])[valid_idx.unsqueeze(-1)].view(-1, masks[0].shape[-1])

    targets = X.clone()
    X = X * target_masks  # mask input
    if mask_compensation:
        X = compensate_masking(X, target_masks)

    padding_masks = padding_mask(lengths, max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
    target_masks = ~target_masks  # inverse logic: 0 now means ignore, 1 means predict
    return X, targets, target_masks, padding_masks

def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))