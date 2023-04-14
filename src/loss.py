
import torch
from torch import nn
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
    

class MaskedLoss(nn.Module):
    """ Masked Loss
    """
    LOSS_FUNCTIONS = {
        'mae': nn.L1Loss,
        'mse': nn.MSELoss,
    }

    def __init__(self, reduction: str = 'mean', type_of_loss: str = 'mae'):

        super().__init__()

        self.reduction = reduction
        self.loss = self.LOSS_FUNCTIONS[type_of_loss](reduction=reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask

        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.loss(masked_pred, masked_true)