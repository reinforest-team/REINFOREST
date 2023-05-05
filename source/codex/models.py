import torch
from torch import nn
from source.similatiry_loss import CrossMatchLoss


class CodexBasedModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str = 'text-ada',
        semantic_match_factor: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.conversion_layers = nn.ModuleList([
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
                bias=True
            ),
            # nn.Sigmoid(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
            ),
            # nn.ReLU(),
        ])
        self.drop = nn.Dropout(0.1)
        self.loss_fn = CrossMatchLoss(
            semantic_match_factor=semantic_match_factor)

    def get_vector(
        self,
        input_vector: torch.Tensor,  # (B * H)
    ):
        batched = True
        if input_vector.ndim == 1:
            batched = False
            input_vector = input_vector.unsqueeze(0)
        assert input_vector.ndim == 2
        output = input_vector
        for conversion_layer in self.conversion_layers:
            output = conversion_layer(output)
        # output = self.conversion_layer(input_vector)
        if not batched:
            output = output.squeeze(0)
        return output.detach()

    def convert_vector(
        self,
        input_vector: torch.Tensor,  
    ):
        output = input_vector
        for conversion_layer in self.conversion_layers:
            output = conversion_layer(output)
        return output

    def forward(
        self,
        input_vector: torch.Tensor,  # (B * H)
        positive_vectors: torch.Tensor = None,  # (B * P * H)
        negative_vectors: torch.Tensor = None,  # (B * N * H)
        positive_semantic_match_scores: torch.Tensor = None,  # (B * P)
        negative_semantic_match_scores: torch.Tensor = None,  # (B * N)
    ):
        # print(input_vector.shape)
        p = positive_vectors.shape[1]
        n = negative_vectors.shape[1]
        input_vector = self.drop(self.convert_vector(input_vector))
        if p > 0:
            positive_vectors = self.drop(self.convert_vector(positive_vectors))
        if n > 0:
            negative_vectors = self.drop(self.convert_vector(negative_vectors))
        return self.loss_fn(
            input_vector=input_vector,
            positive_vectors=positive_vectors if p > 0 else None,
            negative_vectors=negative_vectors if n > 0 else None,
            positive_semantic_match_scores=positive_semantic_match_scores if p > 0 else None,
            negative_semantic_match_scores=negative_semantic_match_scores if n > 0 else None,
        )
