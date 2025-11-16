from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import functional as F

class Processor(ABC):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass

class GreedyProcessor(Processor):
    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits
    
    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)