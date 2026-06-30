import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # a: [..., d], b: [..., d] -> [...]
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@dataclass
class LGSSConfig:
    # Dimensions
    shot_feat_dim: int
    bnet_hidden_dim: int = 256
    bnet_conv_channels: int = 256

    # Clip-level window: boundary i uses shots [i-(w_b-1) .. i] and [i+1 .. i+w_b]
    w_b: int = 4

    # Segment-level
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # Coarse-to-super-shot
    coarse_threshold: float = 0.5

    # Global optimal grouping (DP)
    # For speed, you usually want to limit the number of super shots.
    max_super_shots: int = 600
    min_scenes: int = 50
    max_scenes: int = 400

    # For DP scoring: sigmoid(z)
    sigmoid_temp: float = 1.0


class BNet(nn.Module):
    """Boundary Network (BNet), as described by the paper’s Eq. (2).

    Inputs are per-shot semantic vectors s_t.

    For each boundary i (between shots i and i+1), it builds a window of 2*w_b shots:
        before: s_{i-(w_b-1)} .. s_i    (length w_b)
        after : s_{i+1} .. s_{i+w_b}    (length w_b)

    It produces a boundary representation b_i by combining:
      - Bd branch: two temporal conv embeddings (before/after) and an inner product.
      - Br branch: temporal conv over the full window and max pooling.
    """

    def __init__(self, shot_feat_dim: int, w_b: int, bnet_hidden_dim: int, bnet_conv_channels: int):
        super().__init__()
        self.shot_feat_dim = shot_feat_dim
        self.w_b = w_b

        # Bd branch convs (embed before/after separately)
        # Conv1d expects [B, C, T], so we use shot_feat_dim as C.
        self.bd_conv1 = nn.Conv1d(shot_feat_dim, bnet_conv_channels, kernel_size=3, padding=1)
        self.bd_conv2 = nn.Conv1d(bnet_conv_channels, bnet_conv_channels, kernel_size=3, padding=1)

        # Br branch conv
        self.br_conv1 = nn.Conv1d(shot_feat_dim, bnet_conv_channels, kernel_size=3, padding=1)
        self.br_conv2 = nn.Conv1d(bnet_conv_channels, bnet_conv_channels, kernel_size=3, padding=1)

        # Project Bd(inner product scalar) + Br(vector) to bnet_hidden_dim
        self.bd_proj = nn.Linear(1, bnet_hidden_dim // 2)
        self.br_proj = nn.Linear(bnet_conv_channels, bnet_hidden_dim - bnet_hidden_dim // 2)
        self.out_proj = nn.Linear(bnet_hidden_dim, bnet_hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def _embed_seq(self, x_seq: torch.Tensor, conv1: nn.Conv1d, conv2: nn.Conv1d) -> torch.Tensor:
        # x_seq: [B, T, D] -> [B, D, T]
        x = x_seq.transpose(1, 2)
        x = F.relu(conv1(x))
        x = F.relu(conv2(x))
        # pool over time -> [B, C]
        x = x.mean(dim=-1)
        return x

    def forward(self, shot_feats: torch.Tensor) -> torch.Tensor:
        """Compute boundary representations.

        Args:
            shot_feats: [n_shots, shot_feat_dim] or [B, n_shots, shot_feat_dim]

        Returns:
            boundary_reps: [n_shots-1, bnet_hidden_dim] or [B, n_shots-1, bnet_hidden_dim]
        """
        if shot_feats.dim() == 2:
            shot_feats = shot_feats.unsqueeze(0)
            squeeze_b = True
        else:
            squeeze_b = False

        B, n, D = shot_feats.shape
        assert D == self.shot_feat_dim

        # Pad on both sides so every boundary has a valid window.
        # boundary i corresponds to between shots i and i+1, with i in [0..n-2].
        pad = self.w_b  # enough for i-(w_b-1) and i+w_b at edges
        padded = F.pad(shot_feats, (0, 0, pad, pad), mode="constant", value=0.0)
        # padded index for original shot t is t + pad.

        reps = []
        for i in range(n - 1):
            # before: i-(w_b-1) .. i  => length w_b
            b_start = i - (self.w_b - 1)
            before_idx = torch.arange(b_start, i + 1, device=shot_feats.device) + pad
            after_idx = torch.arange(i + 1, i + 1 + self.w_b, device=shot_feats.device) + pad

            before_seq = padded[:, before_idx, :]  # [B, w_b, D]
            after_seq = padded[:, after_idx, :]    # [B, w_b, D]
            full_seq = padded[:, torch.cat([before_idx, after_idx], dim=0), :]  # [B, 2*w_b, D]

            # Bd: embed before and after, inner product
            emb_before = self._embed_seq(before_seq, self.bd_conv1, self.bd_conv2)  # [B, C]
            emb_after = self._embed_seq(after_seq, self.bd_conv1, self.bd_conv2)    # [B, C]
            # inner product scalar per batch
            inner = (emb_before * emb_after).sum(dim=-1, keepdim=True)  # [B, 1]
            bd_vec = self.bd_proj(inner)  # [B, hidden/2]

            # Br: temporal conv on full window + max pool
            x = full_seq.transpose(1, 2)  # [B, D, T]
            x = F.relu(self.br_conv1(x))
            x = F.relu(self.br_conv2(x))
            x = x.max(dim=-1).values  # [B, C]
            br_vec = self.br_proj(x)   # [B, hidden-hidden/2]

            b = torch.cat([bd_vec, br_vec], dim=-1)
            b = self.dropout(F.relu(b))
            b = self.out_proj(b)
            reps.append(b)

        boundary_reps = torch.stack(reps, dim=1)  # [B, n-1, hidden]
        if squeeze_b:
            boundary_reps = boundary_reps.squeeze(0)
        return boundary_reps


class SegmentLevelModel(nn.Module):
    """Segment-level coarse prediction at T level (paper Eq. (3)).

    Takes boundary representations b_i and predicts coarse probabilities p_i.
    """

    def __init__(self, bnet_hidden_dim: int, lstm_hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=bnet_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden_dim, 1)

    def forward(self, boundary_reps: torch.Tensor) -> torch.Tensor:
        # boundary_reps: [B, n-1, d] or [n-1, d]
        if boundary_reps.dim() == 2:
            boundary_reps = boundary_reps.unsqueeze(0)
            squeeze_b = True
        else:
            squeeze_b = False

        x, _ = self.lstm(boundary_reps)  # [B, n-1, 2*h]
        x = self.dropout(x)
        logits = self.fc(x).squeeze(-1)  # [B, n-1]
        probs = torch.sigmoid(logits)

        if squeeze_b:
            probs = probs.squeeze(0)
        return probs


def build_super_shots_from_boundaries(
    n_shots: int,
    coarse_boundary_mask: torch.Tensor,
    shot_feat: torch.Tensor,
) -> List[Tuple[int, int, torch.Tensor]]:
    """Convert coarse boundary predictions into super shots.

    Args:
        n_shots: number of shots
        coarse_boundary_mask: [n_shots-1] bool tensor where True indicates a boundary
        shot_feat: [n_shots, d]

    Returns:
        list of tuples (start_shot_idx, end_shot_idx, super_shot_repr)
        with end inclusive.
    """
    device = shot_feat.device
    assert coarse_boundary_mask.shape[0] == n_shots - 1

    boundaries = torch.nonzero(coarse_boundary_mask, as_tuple=False).view(-1).tolist()
    # boundaries are indices i indicating boundary between i and i+1.

    segments: List[Tuple[int, int]] = []
    start = 0
    for b in boundaries:
        end = b  # last shot before boundary
        if end >= start:
            segments.append((start, end))
        start = b + 1
    if start <= n_shots - 1:
        segments.append((start, n_shots - 1))

    # Ensure at least one super shot.
    if len(segments) == 0:
        segments = [(0, n_shots - 1)]

    out = []
    for s, e in segments:
        seg_repr = shot_feat[s : e + 1].mean(dim=0)
        out.append((s, e, seg_repr.to(device)))
    return out


def limit_super_shots(
    super_shots: List[Tuple[int, int, torch.Tensor]],
    max_super_shots: int,
) -> List[Tuple[int, int, torch.Tensor]]:
    """Downsample super shots if too many.

    We keep super shots uniformly distributed by index.
    """
    m = len(super_shots)
    if m <= max_super_shots:
        return super_shots
    idx = torch.linspace(0, m - 1, steps=max_super_shots).round().to(torch.long).tolist()
    return [super_shots[i] for i in idx]


class GlobalOptimalGroupingDP(nn.Module):
    """Global optimal grouping at movie level (paper Eq. (6) + DP).

    This implementation follows the described objective:
      g(φ_k) = sum_{C in φ_k} ( Fs(C, P) + Ft(C, P) )

    where P are super shots outside current scene.

    Fs(C, P) = 1/|P| sum_{C' in P} cos(C, C')
    Ft(C, P) = σ( max_{C' in P} cos(C, C') )

    The DP partitions the ordered super shots into t scenes.

    Note: The paper includes iterative refinement of super-shot representations
    (details in supplementary). This code uses one-pass representation:
    super shot = mean of shot features.
    """

    def __init__(
        self,
        min_scenes: int,
        max_scenes: int,
        max_super_shots: int,
        sigmoid_temp: float = 1.0,
    ):
        super().__init__()
        self.min_scenes = min_scenes
        self.max_scenes = max_scenes
        self.max_super_shots = max_super_shots
        self.sigmoid_temp = sigmoid_temp

    def _g_scene(
        self,
        scene_start: int,
        scene_end: int,
        cos_mat: torch.Tensor,
    ) -> torch.Tensor:
        # cos_mat: [m, m] precomputed for all super shots (cosine similarity)
        # scene includes indices [scene_start..scene_end] inclusive.
        m = cos_mat.shape[0]
        device = cos_mat.device

        scene_idx = torch.arange(scene_start, scene_end + 1, device=device)
        outside_mask = torch.ones(m, dtype=torch.bool, device=device)
        outside_mask[scene_start : scene_end + 1] = False
        outside_idx = torch.nonzero(outside_mask, as_tuple=False).view(-1)

        if outside_idx.numel() == 0:
            return torch.tensor(0.0, device=device)

        P = outside_idx
        # For each u in scene, compute Fs and Ft over P.
        # Fs: mean cos(u, v) over v in P
        Fs = cos_mat[scene_idx][:, P].mean(dim=1)  # [scene_len]
        # Ft: sigmoid(max cos(u, v)) over v in P
        max_cos = cos_mat[scene_idx][:, P].max(dim=1).values  # [scene_len]
        Ft = torch.sigmoid(self.sigmoid_temp * max_cos)

        g = (Fs + Ft).sum()  # scalar
        return g

    def forward(
        self,
        super_shots: List[Tuple[int, int, torch.Tensor]],
    ) -> Tuple[List[int], int, float]:
        """Return predicted scene boundary *between super shots*.

        Returns:
            scene_cuts: list of super-shot cut points (indices where a scene boundary occurs,
                        i.e., after super shot k). For m super shots, boundaries are k in [0..m-2].
            best_num_scenes: number of scenes selected by maximizing over target scene counts.
            best_score: DP objective score.
        """
        if len(super_shots) == 0:
            return [], 0, 0.0

        super_shots = limit_super_shots(super_shots, self.max_super_shots)
        m = len(super_shots)

        # Scene boundaries can only be created between super shots.
        if m <= 1:
            return [], 1, 0.0

        reps = torch.stack([ss[2] for ss in super_shots], dim=0)  # [m, d]
        reps = l2_normalize(reps)
        cos_mat = reps @ reps.t()  # [m, m]

        # Clamp scene range to feasible values.
        min_s = max(1, self.min_scenes)
        max_s = min(self.max_scenes, m)
        if min_s > max_s:
            # fallback: one scene
            return [], 1, 0.0

        # DP: dp[t][k] = best score for partitioning first k super-shots (0..k-1) into t scenes.
        # We compute dp up to t=max_s and k=m.
        maxT = max_s
        NEG = -1e30
        dp = [[NEG for _ in range(m + 1)] for __ in range(maxT + 1)]
        back = [[-1 for _ in range(m + 1)] for __ in range(maxT + 1)]

        # Base t=1
        for k in range(1, m + 1):
            dp[1][k] = self._g_scene(0, k - 1, cos_mat)
            back[1][k] = 0

        # Transitions
        for t in range(2, maxT + 1):
            for k in range(t, m + 1):
                # last scene starts at i (i..k-1)
                best_val = NEG
                best_i = -1
                # i must be at least t-1
                for i in range(t - 1, k):
                    prev = dp[t - 1][i]
                    if prev <= NEG / 2:
                        continue
                    val = prev + self._g_scene(i, k - 1, cos_mat)
                    if val > best_val:
                        best_val = val
                        best_i = i
                dp[t][k] = best_val
                back[t][k] = best_i

        # Choose best t in [min_s..max_s] for k=m
        best_score = NEG
        best_t = -1
        for t in range(min_s, max_s + 1):
            if dp[t][m] > best_score:
                best_score = dp[t][m]
                best_t = t

        # Recover cuts between scenes at super-shot level.
        # We have back pointers to determine partition indices.
        cuts = []
        t = best_t
        k = m
        # Last scene ends at m-1.
        # For each scene, scene is [i..k-1], boundary between i-1 and i corresponds to cut at i-1.
        while t > 1:
            i = back[t][k]
            # cut after super shot i-1
            if i - 1 >= 0:
                cuts.append(i - 1)
            k = i
            t -= 1

        cuts = sorted(cuts)
        return cuts, best_t, float(best_score)


class LGSSModel(nn.Module):
    """End-to-end LGSS model.

    Training typically uses clip/segment-level cross entropy on coarse predictions.
    During inference, you can apply global optimal grouping using DP.
    """

    def __init__(self, cfg: LGSSConfig):
        super().__init__()
        self.cfg = cfg
        self.bnet = BNet(
            shot_feat_dim=cfg.shot_feat_dim,
            w_b=cfg.w_b,
            bnet_hidden_dim=cfg.bnet_hidden_dim,
            bnet_conv_channels=cfg.bnet_conv_channels,
        )
        self.segment_model = SegmentLevelModel(
            bnet_hidden_dim=cfg.bnet_hidden_dim,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        )
        self.global_grouping = GlobalOptimalGroupingDP(
            min_scenes=cfg.min_scenes,
            max_scenes=cfg.max_scenes,
            max_super_shots=cfg.max_super_shots,
            sigmoid_temp=cfg.sigmoid_temp,
        )

    def forward(
        self,
        shot_feats: torch.Tensor,
        apply_global_grouping: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            shot_feats: [n_shots, shot_feat_dim] or [B, n_shots, shot_feat_dim]
            apply_global_grouping: if True, refine boundaries using DP.

        Returns:
            coarse_boundary_probs: [B, n_shots-1] or [n_shots-1]
            final_boundary_probs:  [B, n_shots-1] or [n_shots-1]
              (when apply_global_grouping=False, final==coarse)

        Note:
            DP global grouping is non-differentiable, so gradients from final
            are not used.
        """
        if shot_feats.dim() == 2:
            shot_feats = shot_feats.unsqueeze(0)
            squeeze_b = True
        else:
            squeeze_b = False

        B, n, _ = shot_feats.shape

        # Clip-level boundary reps
        boundary_reps = self.bnet(shot_feats)  # [B, n-1, d]
        # Segment-level coarse probs
        coarse_probs = self.segment_model(boundary_reps)  # [B, n-1]

        if not apply_global_grouping:
            final_probs = coarse_probs
            return (coarse_probs.squeeze(0) if squeeze_b else coarse_probs,
                    final_probs.squeeze(0) if squeeze_b else final_probs)

        # Global optimal grouping per sample
        final_probs_list = []
        thr = self.cfg.coarse_threshold
        for b in range(B):
            probs = coarse_probs[b]  # [n-1]
            mask = probs > thr  # [n-1] bool

            super_shots = build_super_shots_from_boundaries(
                n_shots=n,
                coarse_boundary_mask=mask,
                shot_feat=shot_feats[b],
            )

            cuts_super, num_scenes, best_score = self.global_grouping(super_shots)
            # Convert super-shot cuts into shot-level boundaries.
            # A cut after super-shot k corresponds to a scene boundary between
            # (end of super-shot k) and the next super-shot.
            final_boundary = torch.zeros(n - 1, device=shot_feats.device, dtype=probs.dtype)

            # Reconstruct super-shot segments after possible downsampling
            super_shots_ds = limit_super_shots(super_shots, self.cfg.max_super_shots)
            # Need to apply the same representation of super shots used in DP.
            # Then cuts_super refer to indices in super_shots_ds.
            for k in cuts_super:
                # super_shots_ds[k] ends at some shot index end
                end_shot = super_shots_ds[k][1]
                if 0 <= end_shot <= n - 2:
                    final_boundary[end_shot] = 1.0

            # Optionally, you could set probabilities instead of hard 0/1.
            # Here we set hard labels.
            final_probs_list.append(final_boundary)

        final_probs = torch.stack(final_probs_list, dim=0)  # [B, n-1]

        if squeeze_b:
            return coarse_probs.squeeze(0), final_probs.squeeze(0)
        return coarse_probs, final_probs


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    """Weighted binary cross entropy on logits.

    paper uses class imbalance weight ~ 1:9.
    """
    # logits/targets: [*, n-1]
    targets = targets.float()
    # BCE with logits per element
    loss_per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = targets * pos_weight + (1.0 - targets) * neg_weight
    return (loss_per * weights).mean()


class LGSSTrainer:
    """A minimal trainer for clip/segment-level coarse predictions.

    This trainer computes loss only on coarse segment-level logits.
    Global optimal grouping is used only for inference.

    You provide:
      - shot_feats: [n_shots, shot_feat_dim]
      - boundary_targets: [n_shots-1] with 1 if boundary is a scene boundary.
    """

    def __init__(self, model: LGSSModel, lr: float = 1e-3, pos_weight: float = 1.0, neg_weight: float = 9.0):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def train_step(self, shot_feats: torch.Tensor, boundary_targets: torch.Tensor) -> float:
        self.model.train()
        self.optim.zero_grad(set_to_none=True)

        # Forward without DP; we need logits. Recompute segment logits by running BNet + LSTM.
        if shot_feats.dim() == 2:
            shot_feats_b = shot_feats.unsqueeze(0)
        else:
            shot_feats_b = shot_feats

        boundary_reps = self.model.bnet(shot_feats_b)
        # Recreate logits from segment model: it currently outputs probs.
        # We'll instead do a tiny trick: get probs then invert? not stable.
        # So we implement coarse logits by copying the head logic.
        # Better: expose logits head; but for simplicity, we take probs and use BCE on probs.
        # Users can swap this to BCEWithLogits if they modify SegmentLevelModel.
        coarse_probs = self.model.segment_model(boundary_reps)  # [B, n-1]

        targets = boundary_targets
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        # Weighted BCE on probabilities.
        # Convert to logits-like by clamping and using log-prob.
        eps = 1e-6
        p = coarse_probs.clamp(eps, 1 - eps)
        loss_per = -(targets.float() * torch.log(p) + (1 - targets.float()) * torch.log(1 - p))
        weights = targets.float() * self.pos_weight + (1.0 - targets.float()) * self.neg_weight
        loss = (loss_per * weights).mean()

        loss.backward()
        self.optim.step()
        return float(loss.item())


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example with dummy precomputed shot features.
    # Suppose each shot i already has concatenated feature:
    #   s_i = [place_feat, cast_feat, action_feat, audio_feat]
    # with total dimension = shot_feat_dim.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = LGSSConfig(
        shot_feat_dim=1024,
        w_b=4,
        bnet_hidden_dim=256,
        bnet_conv_channels=128,
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        lstm_dropout=0.2,
        coarse_threshold=0.5,
        max_super_shots=200,  # reduce for demo
        min_scenes=5,
        max_scenes=20,
    )

    model = LGSSModel(cfg).to(device)


    print("coarse_probs shape:", coarse_probs.shape)
    print("final_probs shape:", final_probs.shape)
    print("example final boundary indices:", torch.nonzero(final_probs > 0.5).view(-1)[:20].tolist())
