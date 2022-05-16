import torch


def is_ddp():
  return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
  return torch.distributed.get_rank()


def barrier():
  torch.distributed.barrier()


def to(data, device):
  if isinstance(data, dict):
    data_new = {k:v.to(device) for k, v in data.items()}
  else:
    raise NotImplementedError

  return data_new
