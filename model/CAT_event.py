import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertOnlyMLMHead,
    BertPredictionHeadTransform,
)

class FrameWiseHead(nn.Module):
    """
    你给的头，原样照搬
    - 输入: x (B, L, N)
    - 输出:
        score_logit: (B,)      是否有边界（分类logit）
        center:      (B,)      边界在当前窗口内的相对位置 ∈ [0,1]
    """
    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.1, agg: str = "avg"):
        super().__init__()
        self.agg = agg
        self.proj = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, 1)
        self.reg = nn.Linear(hidden, 1)

        # 可选：简单的时序卷积替代 avg 池化
        self.use_conv = False
        if self.use_conv:
            self.temporal = nn.Sequential(
                nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

    def forward(self, x: torch.Tensor):
        B, L, N = x.shape

        if getattr(self, 'use_conv', False):
            # (B,L,N) -> (B,N,L)
            xc = x.transpose(1, 2)
            h = self.temporal(xc).squeeze(-1)  # (B,hidden)
        else:
            # 简单的时序平均
            xt = x.mean(dim=1)                 # (B, N)
            h = F.relu(self.proj(xt))          # (B, hidden)

        h = self.dropout(h)
        score_logit = self.cls(h).squeeze(1)   # (B,)
        center_raw = self.reg(h).squeeze(1)    # (B,)
        center = torch.sigmoid(center_raw)     # 映射到 [0,1]
        return score_logit, center




class BertMLMHead(BertOnlyMLMHead):
    def __init__(self, cfg):
        super(BertMLMHead, self).__init__(cfg)


class BertITMHead(nn.Module):
    def __init__(self, cfg):
        super(BertITMHead, self).__init__()
        self.transform = BertPredictionHeadTransform(cfg)
        self.decoder = nn.Linear(cfg.hidden_size, 1, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return torch.sigmoid(hidden_states).squeeze(1)


class BertLMPredictionHead(nn.Module):
    def __init__(self, cfg, num_classes):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(cfg)
        self.decoder = nn.Linear(cfg.hidden_size, num_classes, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ShotEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        nn_size = cfg.neighbor_size + 2  # +1 for center shot, +1 for cls
        self.shot_embedding = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.position_embedding = nn.Embedding(nn_size, cfg.hidden_size)
        self.mask_embedding = nn.Embedding(2, cfg.input_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        self.register_buffer("pos_ids", torch.arange(nn_size, dtype=torch.long))

    def forward(
        self,
        shot_emb: torch.Tensor,
        mask: torch.Tensor = None,
        pos_ids: torch.Tensor = None,
    ) -> torch.Tensor:

        assert len(shot_emb.size()) == 3

        if pos_ids is None:
            pos_ids = self.pos_ids

        # this for mask embedding (un-masked ones remain unchanged)
        if mask is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask_emb = self.mask_embedding(mask.long())
            shot_emb = (shot_emb * (1 - mask).float()[:, :, None]) + mask_emb

        # we set [CLS] token to averaged feature
        cls_emb = shot_emb.mean(dim=1)

        # embedding shots
        shot_emb = torch.cat([cls_emb[:, None, :], shot_emb], dim=1)
        shot_emb = self.shot_embedding(shot_emb)
        pos_emb = self.position_embedding(pos_ids)
        embeddings = shot_emb + pos_emb[None, :]
        embeddings = self.dropout(self.LayerNorm(embeddings))
        return embeddings


class TransformerCRN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.pooling_method = cfg.pooling_method
        self.shot_embedding = ShotEmbedding(cfg)
        self.encoder = BertEncoder(cfg)

        nn_size = cfg.neighbor_size + 2  # +1 for center shot, +1 for cls
        num_head = cfg.num_attention_heads
        attention_glocal_window = cfg.attention_local_window
        # self.register_buffer(
        #     "attention_mask",
        #     self._get_extended_attention_mask(torch.ones((1, nn_size)).float()),
        # )

        # local_global_attention
        self.register_buffer(
            "local_global_attention_mask",
            self._get_local_global_attention_mask(torch.zeros((1, num_head, nn_size, nn_size)).float(),
                                              attention_glocal_window)
        )

    def forward(
        self,
        shot: torch.Tensor,
        mask: torch.Tensor = None,
        pos_ids: torch.Tensor = None,
        pooling_method: str = None,
    ):
        # if self.attention_mask.shape[1] != (shot.shape[1] + 1):
        #     n_shot = shot.shape[1] + 1  # +1 for CLS token
        #     attention_mask = self._get_extended_attention_mask(
        #         torch.ones((1, n_shot), dtype=torch.float, device=shot.device)
        #     )
        # else:
        attention_mask = self.local_global_attention_mask
        shot_emb = self.shot_embedding(shot, mask=mask, pos_ids=pos_ids)
        encoded_emb = self.encoder(
            shot_emb, attention_mask=attention_mask
        )
        encoded_emb = encoded_emb.last_hidden_state

        return encoded_emb, self.pooler(encoded_emb, pooling_method=pooling_method)

    def pooler(self, sequence_output, pooling_method=None):
        if pooling_method is None:
            pooling_method = self.pooling_method

        if pooling_method == "cls":
            return sequence_output[:, 0, :]
        elif pooling_method == "avg":
            return sequence_output[:, 1:].mean(dim=1)
        elif pooling_method == "max":
            return sequence_output[:, 1:].max(dim=1)[0]
        elif pooling_method == "center":
            cidx = sequence_output.shape[1] // 2
            return sequence_output[:, cidx, :]
        else:
            raise ValueError


    def _get_local_global_attention_mask(self, local_global_attention_mask, glocal_window):
        _, h, s, _ = local_global_attention_mask.shape
        local_head_num = h//3
        for i in range(local_head_num):
            for j in range(s):
                for k in range(glocal_window):
                    local_global_attention_mask[:, i, j, min(max(j - glocal_window // 2 + k, 0), s - 1)] = 1.0

        for i in range(local_head_num, h):
            local_global_attention_mask[:, i] = 1.0

        # for cls
        local_global_attention_mask[:, :, 0, :] = 1.0

        local_global_attention_mask = (1.0 - local_global_attention_mask) * -10000.0
        return local_global_attention_mask

    def _get_extended_attention_mask(self, attention_mask):

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class EventCAT(nn.Module):
    """Context-Aware Transformer for temporal event detection.

    Input:  shot features  x         of shape [B, T, C]
            (optional) attention_mask of shape [B, T]  (1 = valid, 0 = pad)
    Output: per-timestep event logits of shape [B, T, num_classes]

    Required cfg fields (BertConfig-compatible + extras):
        input_dim, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob,
        attention_probs_dropout_prob, layer_norm_eps,
        max_position_embeddings, attention_local_window, num_classes
    Optional:
        is_decoder (False), add_cross_attention (False),
        chunk_size_feed_forward (0), classifier_dropout_prob (hidden_dropout_prob)
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_attention_heads = cfg.num_attention_heads
        self.attention_local_window = cfg.attention_local_window
        self.max_position_embeddings = cfg.max_position_embeddings

        self.input_projection = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.position_embedding = nn.Embedding(
            cfg.max_position_embeddings, cfg.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        self.encoder = BertEncoder(cfg)

        cls_dp = getattr(cfg, "classifier_dropout_prob", cfg.hidden_dropout_prob)
        # self.classifier = nn.Sequential(
        #     nn.Linear(cfg.hidden_size, cfg.hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(cls_dp),
        #     nn.Linear(cfg.hidden_size, cfg.num_classes),
        # )
        # self.center = nn.Sequential(
        #     nn.Linear(cfg.hidden_size, cfg.hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(cls_dp),
        #     nn.Linear(cfg.hidden_size, cfg.num_classes),
        # )
        self.head = FrameWiseHead(in_features=cfg.hidden_size, hidden=256, dropout=cls_dp)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_features: bool = False,
        task_name=None,
    ):
        # x: [B, T, C]
        assert x.dim() == 3, f"expected [B, T, C], got {tuple(x.shape)}"
        B, T, _ = x.shape
        assert T <= self.max_position_embeddings, (
            f"T={T} exceeds max_position_embeddings={self.max_position_embeddings}"
        )

        h = self.input_projection(x)
        pos_ids = torch.arange(T, device=x.device)
        h = h + self.position_embedding(pos_ids)[None, :, :]
        h = self.dropout(self.LayerNorm(h))

        ext_mask = self._build_attention_mask(T, attention_mask, x.device, x.dtype)

        encoded = self.encoder(h, attention_mask=ext_mask).last_hidden_state  # [B, T, H]
        # logits = self.classifier(encoded)  # [B, T, num_classes]
        score_logit, center = self.head(encoded)  # (B,), (B,)
        return score_logit, center

    def _build_attention_mask(self, T, padding_mask, device, dtype):
        """Combine local-global head mask with optional padding mask.

        Returns additive mask of shape [B or 1, num_heads, T, T] with
        0 for attend, -10000 for mask.
        """
        h = self.num_attention_heads
        w = self.attention_local_window
        local_head_num = h // 3

        # per-head base mask [1, h, T, T] — 1 = attend, 0 = block
        base = torch.ones(1, h, T, T, device=device)
        if local_head_num > 0:
            idx = torch.arange(T, device=device)
            diff = (idx[:, None] - idx[None, :]).abs()
            local_mask = (diff <= (w // 2)).to(base.dtype)  # [T, T]
            base[:, :local_head_num] = local_mask[None, None, :, :]

        if padding_mask is not None:
            # padding_mask: [B, T] -> valid positions 1, pad 0
            pm = padding_mask.to(base.dtype)
            # block any key that is padding: [B, 1, 1, T]
            key_mask = pm[:, None, None, :]
            base = base * key_mask  # broadcasts to [B, h, T, T]

        ext = (1.0 - base) * -10000.0
        return ext.to(dtype)


def get_event_cat(cfg):
    """Helper to build EventCAT from a cfg section (e.g. cfg.MODEL.event_cat)."""
    return EventCAT(cfg)

if __name__ == "__main__":
    from transformers import BertConfig

    cfg = BertConfig(
    hidden_size=768, num_hidden_layers=4, num_attention_heads=8,
    intermediate_size=3072, hidden_act="gelu",
    hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    max_position_embeddings=2048,   # T 上限
    layer_norm_eps=1e-12,
    )
    cfg.input_dim = 2048                # 你 shot encoder 输出的 C
    cfg.attention_local_window = 5      # 局部注意力窗口（奇数）
    cfg.num_classes = 1 # 事件类别数
    cfg._attn_implementation = "eager"   # 或 "sdpa" 想用 PyTorch SDPA 的话

    model = EventCAT(cfg)
    B = 8
    x = torch.randn(B, 21, 2048)
    
    score_logit, center = model(x)
    print("score_logit:", score_logit.shape)  # (B,)
    print("center:", center.shape)            # (B,)