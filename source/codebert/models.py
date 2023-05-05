import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional
import numpy as np
from source.similatiry_loss import CrossMatchLoss
from torch.optim import AdamW
from tqdm import tqdm


class CodeBERTBasedModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'codebert',
        semantic_match_factor: float = 0.1,
    ):
        super().__init__()
        assert model_name in [
            'codebert', 'graphcodebert', 'roberta'
        ], "Only codebert, graphcodebert, and roberta are supported"
        if model_name == 'codebert':
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        elif model_name == 'graphcodebert':
            self.model = AutoModel.from_pretrained(
                "microsoft/graphcodebert-base")
        else:
            self.model = AutoModel.from_pretrained("roberta-base")
        self.semantic_match_factor = semantic_match_factor
        self.loss_fn = CrossMatchLoss(
            semantic_match_factor=semantic_match_factor)

    def get_vector(
        self,
        input_ids: torch.Tensor,  # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
    ):
        batched = True
        if input_ids.ndim == 1:
            batched = False
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        assert input_ids.ndim == 2 and attention_mask.ndim == 2
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        _vector = output.pooler_output
        if not batched:
            _vector = _vector.squeeze(0)
        return _vector.detach()

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        pos_input_ids: Optional[torch.Tensor] = None,  # (B, P, L)
        pos_attn_mask: Optional[torch.Tensor] = None,  # (B, P, L)
        pos_semantic_scores: Optional[torch.Tensor] = None,  # (B, P)
        neg_input_ids: Optional[torch.Tensor] = None,  # (B, N, L)
        neg_attn_mask: Optional[torch.Tensor] = None,  # (B, N, L)
        neg_semantic_scores: Optional[torch.Tensor] = None,  # (B, N)
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        input_vector = output.pooler_output
        if pos_input_ids is not None and pos_input_ids.shape[1] > 0:
            output = self.model(
                input_ids=pos_input_ids.reshape(-1, pos_input_ids.shape[-1]),
                attention_mask=None if pos_attn_mask is None
                else pos_attn_mask.reshape(-1, pos_attn_mask.shape[-1])
            )
            positive_vectors = output.pooler_output
            positive_vectors = positive_vectors.reshape(
                pos_input_ids.shape[0], pos_input_ids.shape[1], -1
            )
        else:
            positive_vectors = None

        if neg_input_ids is not None and neg_input_ids.shape[1] > 0:
            output = self.model(
                input_ids=neg_input_ids.reshape(-1, neg_input_ids.shape[-1]),
                attention_mask=None if neg_attn_mask is None
                else neg_attn_mask.reshape(-1, neg_attn_mask.shape[-1])
            )
            negative_vectors = output.pooler_output
            negative_vectors = negative_vectors.reshape(
                neg_input_ids.shape[0], neg_input_ids.shape[1], -1
            )
        else:
            negative_vectors = None
        return self.loss_fn(
            input_vector=input_vector,
            positive_vectors=positive_vectors,
            negative_vectors=negative_vectors,
            positive_semantic_match_scores=pos_semantic_scores,
            negative_semantic_match_scores=neg_semantic_scores
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    B, L, P, N = 8, 512, 5, 5
    input_ids = torch.LongTensor(np.random.randint(0, 100, size=(B, L))).cuda()
    input_attn = torch.ones_like(input_ids)
    positive_ids = torch.LongTensor(
        np.random.randint(0, 100, size=(B, P, L))).cuda()
    positive_attn = torch.ones_like(positive_ids)
    negative_ids = torch.LongTensor(
        np.random.randint(100, 200, size=(B, N, L))).cuda()
    negative_attn = torch.ones_like(negative_ids)
    positive_scores = torch.FloatTensor(
        np.random.uniform(0.8, 1, size=(B, P))).cuda()
    negative_scores = torch.FloatTensor(
        np.random.uniform(0, 0.2, size=(B, N))).cuda()
    model = CodeBERTBasedModel()
    model.cuda()
    optim = AdamW(model.parameters())
    bar = tqdm(range(1000), desc="Loss : Inf")
    model.train()
    for idx in bar:
        model.zero_grad()
        optim.zero_grad()
        output = model(
            input_ids=input_ids,
            attention_mask=input_attn,
            pos_input_ids=positive_ids,
            pos_attn_mask=positive_attn,
            neg_input_ids=negative_ids,
            neg_attn_mask=negative_attn,
            pos_semantic_scores=positive_scores,
            neg_semantic_scores=negative_scores,
        )
        loss = output["loss"]
        bar.set_description("Loss : %.6f" % loss.detach().cpu().item())
        if idx % 10 == 9:
            print(idx, loss.detach().cpu().item())
        loss.backward()
        optim.step()
