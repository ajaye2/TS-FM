
import torch
from torchmetrics import Metric

class MASE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("abs_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("naive_abs_errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        abs_errors = torch.abs(target - preds)
        self.abs_errors += torch.sum(abs_errors)
        
        # Compute naive forecast errors using previous observation
        naive_forecast = target[:, :-1, :]
        naive_target = target[:, 1:, :]
        naive_abs_errors = torch.abs(naive_target - naive_forecast)
        self.naive_abs_errors += torch.sum(naive_abs_errors)
        
        self.total += target.shape[0] * (target.shape[1] - 1)

    def compute(self):
        mase = self.abs_errors / (self.naive_abs_errors / self.total)
        return mase