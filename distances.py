import torch
import torch.nn as nn

class Distance(nn.Module):
    def __init__(self):
        """Parent class for distance and for similarity
        measures."""
        super().__init__()

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor[Batch, Sequence, Emb]
        y : torch.Tensor[Batch, Sequence, Emb]

        or unbatched
        
        returns
        torch.Tensor[Batch, Sequence]"""

        raise NotImplementedError
    
class CosineSimilarity(Distance):
    def __init__(self):
        """"""
        super().__init__()

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        return (x * y).sum(-1)  # TODO

class LDistance(Distance):
    def __init__(self, p : float):
        super().__init__()
        self.p : float = p

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        return (x - y).abs().pow(self.p).sum(-1).pow(1/self.p)
    
class EuclideanDistance(LDistance):
    def __init__(self):
        super().__init__(2)
    
    def forward(self, x : torch.Tensor, y : torch.Tensor):
        return (x - y).pow(2).sum(-1).sqrt()
    
class ManhattenDistance(LDistance):
    def __init__(self):
        super().__init__(1)

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        return (x - y).abs().sum(-1)
    
class BilinearDistance(Distance):
    def __init__(self, dim : int):
        super().__init__()

        self.bilinear : nn.Bilinear = nn.Bilinear(dim, dim, 1, bias = True)

    def forward(self, x : torch.Tensor, y : torch.Tensor):
        result : torch.Tensor = self.bilinear(x, y)
        return result.squeeze(-1)
    