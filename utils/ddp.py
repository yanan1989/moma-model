import torch


def is_ddp():
  return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
  return torch.distributed.get_rank()


def barrier():
  torch.distributed.barrier()
