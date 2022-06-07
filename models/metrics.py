import torchmetrics
import torch.nn as nn


def get_metrics(level):
  if level == 'act':
    return nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                          'acc5': torchmetrics.Accuracy(average='micro', top_k=5)})

  elif level == 'sact':
    return nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                          'acc5': torchmetrics.Accuracy(average='micro', top_k=5)})

  elif level == 'aact':
    pass

  else:
    raise NotImplementedError
