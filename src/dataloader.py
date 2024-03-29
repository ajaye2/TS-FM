import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from .dataset import collate_unsuperv, collate_superv, TSDataset

    
class TSDataLoader:
    def __init__(self, datasets, batch_size, max_len, shuffle=True, collate_fn=None, pad_inputs=False, mask_inputs=True):
        self.datasets = datasets
        self.batch_size = batch_size
        
        if collate_fn is None:
            self.data_loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True) for name, ds in datasets.items()}
        elif collate_fn == 'unsuperv':

            self.data_loaders = {}
            for name, ds in datasets.items():
                if name == 'univariate' or type(ds) == TSDataset:
                    self.data_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
                else:
                    # self.data_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=lambda x: collate_unsuperv(x, max_len=max_len, mask_inputs=mask_inputs, pad_inputs=pad_inputs))
                    self.data_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)

        elif collate_fn == 'superv':
            self.data_loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=lambda x: collate_superv(x, max_len=max_len)) for name, ds in datasets.items()}

        self.dataset_iterators      = {name: iter(dl) for name, dl in self.data_loaders.items()}
        self.dataset_names          = list(datasets.keys())
        self.current_dataset_index  = 0

        self.exhausted_datasets     = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted_datasets == len(self.dataset_names):
            self.refresh_iterators()
            self.exhausted_datasets = 0
            raise StopIteration

        name = self.dataset_names[self.current_dataset_index]
        iterator = self.dataset_iterators[name]

        try:
            data = next(iterator)
            # print(f"Data retrieved from {name} DataLoader")
        except StopIteration:
            # Refresh the iterator if the dataset is exhausted
            iterator = iter(self.data_loaders[name])
            self.dataset_iterators[name] = iterator
            data = next(iterator)
            # print(f"StopIteration caught for {name} DataLoader, iterator refreshed")

            # Increment the exhausted_datasets counter
            self.exhausted_datasets += 1
        else:
            # Reset the exhausted_datasets counter if we get data without StopIteration
            self.exhausted_datasets = 0

        # Update the current dataset index to cycle through the datasets
        self.current_dataset_index = (self.current_dataset_index + 1) % len(self.dataset_names)

        return {name: data}

    def refresh_iterators(self):
        for name, dl in self.data_loaders.items():
            self.dataset_iterators[name] = iter(dl)








