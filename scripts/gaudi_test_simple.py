import torch
from loguru import logger

device = torch.device("hpu")
logger.info(f"Detected: {torch.hpu.device_count()}x {device}")
a = torch.tensor(1, device="hpu")
logger.info(f"Tensor on HPU: {a}")
