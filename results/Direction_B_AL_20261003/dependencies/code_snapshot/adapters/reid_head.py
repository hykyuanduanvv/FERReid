"""BNNeck + identity-classification head (BoT-style "strong baseline" loss for person ReID).

  raw CLS feature f
    |-- L2-normalize ------------------------> triplet (unchanged: hardest, margin --triplet_margin)
    |-- BatchNorm1d (no bias) = BNNeck ------> classifier (no bias) -> cross-entropy with label smoothing
    |-- BNNeck output, L2-normalized --------> retrieval feature (when --bnneck True)

Enabled by --ce_loss_weight > 0 and/or --bnneck True; with both off (the default) the model is
exactly the historical one (no head, triplet on the normalized raw CLS feature).
The head always runs in float32 (BatchNorm statistics are fragile in fp16).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def head_enabled(args):
    return getattr(args, "ce_loss_weight", 0.0) > 0 or getattr(args, "bnneck", False)


class ReIDHead(nn.Module):

    def __init__(self, dim, num_classes, bnneck=True, label_smoothing=0.1):
        super().__init__()
        self.use_bnneck = bnneck
        self.label_smoothing = label_smoothing
        self.bn = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad_(False)  # BoT: no shift, the classifier sees a centred feature
        self.classifier = nn.Linear(dim, num_classes, bias=False) if num_classes > 0 else None
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, raw, labels=None, compute_ce=False):
        # model.to(fp16) converts the BN buffers and the frozen bias; the trainer only moves *trainable*
        # parameters back to fp32, so check the buffers (checking bn.weight would miss them)
        if self.bn.running_mean.dtype != torch.float32 or self.bn.bias.dtype != torch.float32:
            self.float()
        with torch.autocast(device_type=raw.device.type, enabled=False):
            f = raw.float()
            neck = self.bn(f) if self.use_bnneck else f
            out = {"feat": F.normalize(neck, dim=-1)}
            if compute_ce:
                if self.classifier is None:
                    raise ValueError("ID loss requested but the head was built without classes "
                                     "(num_train_ids=0)")
                logits = self.classifier(neck)
                out["ce"] = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return out


def build_head(args, dim):
    if not head_enabled(args):
        return None
    return ReIDHead(dim, getattr(args, "num_train_ids", 0), bnneck=getattr(args, "bnneck", False),
                    label_smoothing=getattr(args, "label_smoothing", 0.1))
