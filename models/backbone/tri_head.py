from pytorchvideo.models.weight_init import init_net_weights
import torch
import torch.nn as nn


class TriHead(nn.Module):
  def __init__(self, blocks: nn.ModuleList) -> None:
    super().__init__()
    assert blocks is not None
    self.blocks = blocks
    init_net_weights(self)

  def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    y = [self.blocks[i](x) for i in range(len(self.blocks))]
    return y
