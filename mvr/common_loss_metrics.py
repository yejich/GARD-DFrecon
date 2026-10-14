"""Shared restoration-only velocity MSE diagnostics; never modify training loss."""
import torch


class CommonLossMetrics:
    """Pool squared error over channels, tokens, microbatches and DDP ranks.

    Clean/distractor select image patches only; camera selects token zero.
    Missing categories are omitted rather than reported as a misleading zero.
    """
    names = ('clean', 'distractor', 'camera_token')

    def __init__(self):
        self.totals = None

    @torch.no_grad()
    def update(self, prediction, target, distractor_mask):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError('Expected matching [B,V,T,C] prediction and target')
        if distractor_mask.shape != prediction.shape[:-1] or distractor_mask.dtype != torch.bool:
            raise ValueError('Expected boolean [B,V,T] distractor mask')
        errors = (prediction.detach().float() - target.detach().float()).square().mean(-1).double()
        patches = distractor_mask[..., 1:]
        regions = ((errors[..., 1:], ~patches), (errors[..., 1:], patches),
                   (errors[..., :1], torch.ones_like(distractor_mask[..., :1])))
        stats = torch.stack([x for values, mask in regions
                             for x in ((values * mask).sum(), mask.sum().double())])
        if self.totals is None:
            self.totals = torch.zeros_like(stats)
        self.totals += stats

    def compute(self, *, distributed=True, prefix='train', reset=True):
        if self.totals is None:
            return {}
        totals = self.totals.clone()
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        values = totals.cpu().tolist()
        result = {}
        for i, name in enumerate(self.names):
            count = values[2*i+1]
            result[f'{prefix}/{name}_tokens'] = int(count)
            if count:
                result[f'{prefix}/{name}_loss'] = values[2*i] / count
        if reset:
            self.totals = None
        return result
