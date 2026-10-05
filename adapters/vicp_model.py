"""VICP (Visual In-Context Prompting, ICCV 2025) adapted to person ReID -- kept as the in-context baseline.

Context pairs -> frozen encoder copy -> Q-Former (32 tokens per pair) -> yes/no questions for a frozen
Qwen3-0.6B -> hidden states of L*V query tokens -> linear map -> deep visual prompts shared by every
query / gallery image. Retrieval encoder: the pedestrian backbone (adapters/backbones.py) with LoRA.

This is the original VICP prompt path only. The experimental direction-A options (residual prompt, EMA
centring, episodic context, contrastive context loss, prompt distillation) live on the
Direction_A_contrast_distill branch.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapters.backbones import load_backbone, backbone_name
from adapters.reid_head import build_head
from ops.adapt import adapt_encoder
from ops.deep_prompt import insert_deep_prompts, sample_pair
from ops.losses import HardTripletLoss


class SimpleQFormer(nn.Module):
    """Learnable queries cross-attend to the two visual tokens of an image pair.

    Input (B, 2 * visual_token_dim) -> output (B, num_query_tokens, out_dim)."""

    def __init__(self, visual_token_dim, num_visual_tokens, num_query_tokens, qformer_hidden_dim,
                 num_layers=2, num_heads=8, dropout=0.1, out_dim=None):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.out_dim = out_dim if out_dim is not None else qformer_hidden_dim
        self.visual_proj = nn.Linear(visual_token_dim, qformer_hidden_dim)
        self.query_embeddings = nn.Parameter(torch.randn(num_query_tokens, qformer_hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=qformer_hidden_dim, nhead=num_heads,
                                           dim_feedforward=qformer_hidden_dim * 4, dropout=dropout,
                                           activation='gelu', batch_first=False, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(qformer_hidden_dim)
        self.output_proj = nn.Linear(qformer_hidden_dim, self.out_dim) if self.out_dim != qformer_hidden_dim else None

    def forward(self, pair_features):
        bsz = pair_features.size(0)
        memory = self.visual_proj(pair_features.view(bsz, self.num_visual_tokens, -1)).transpose(0, 1).contiguous()
        query = self.query_embeddings.unsqueeze(1).expand(-1, bsz, -1)
        out = self.norm(self.decoder(tgt=query, memory=memory)).transpose(0, 1).contiguous()
        return self.output_proj(out) if self.output_proj is not None else out


class _TrainedEncoderView(nn.Module):
    """--icl_feature trained: question features from the trained encoder (LoRA, no prompt). Held in a plain
    list so the encoder is not registered twice in the state_dict; called under torch.no_grad()."""

    def __init__(self, encoder):
        super().__init__()
        self._encoder = [encoder]

    def forward_features(self, x):
        return self._encoder[0].forward_features(x)


class VICPModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = adapt_encoder(load_backbone(backbone_name(args)).eval(), args)
        if getattr(args, "icl_feature", "frozen") == "trained":
            self.encoder_copy = _TrainedEncoderView(self.encoder)
        else:
            self.encoder_copy = load_backbone(backbone_name(args)).eval()
            self.encoder_copy.requires_grad_(False)
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        self.reid_head = build_head(args, self.encoder.embed_dim)

        self.num_layers = len(self.encoder.blocks)
        llm = self._resolve_llm_model(args.llm_model)
        self.lm = AutoModelForCausalLM.from_pretrained(llm)
        self.lm.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.hidden_size = self.encoder.embed_dim
        lm_dim = self.lm.config.hidden_size
        self.mm_projector = SimpleQFormer(visual_token_dim=self.hidden_size, num_visual_tokens=2,
                                          num_query_tokens=args.num_id_tokens, qformer_hidden_dim=lm_dim,
                                          num_layers=2, num_heads=8, dropout=0.1, out_dim=lm_dim)
        self.query_embeddings = nn.Parameter(torch.randn(args.num_vpt_tokens * self.num_layers, lm_dim) * 0.02)
        self.prompt_mlp = nn.Linear(lm_dim, self.hidden_size, bias=False)
        self.prompt_mlp.weight.data.zero_()
        print("[VICP] backbone:", backbone_name(args), "| ICL question features:", getattr(args, "icl_feature", "frozen"))

    @staticmethod
    def _resolve_llm_model(name):
        local = os.environ.get("QWEN3_06B_DIR", "/root/basic-models/Qwen3-0.6B")
        return local if name == "Qwen/Qwen3-0.6B" and os.path.isdir(local) else name

    def _icl_prompts(self, image_features, labels_e, device):
        """num_icl_bs sequences of num_icl_samples yes/no questions over the context features -> LLM ->
        prompts (num_icl_bs, L*V, D) and the answer-prediction loss (unchanged from VICP)."""
        tok = {1: self.tokenizer.convert_tokens_to_ids('yes'), 0: self.tokenizer.convert_tokens_to_ids('no')}
        input_ids, pair_feats = [], []
        for _ in range(self.args.num_icl_bs):
            s = []
            for _ in range(self.args.num_icl_samples):
                s.extend([-1] * self.args.num_id_tokens)
                if torch.randint(0, 2, (1,)).item() == 1:  # a positive pair of the context
                    idx = torch.randint(0, image_features.size(0) // 2, (1,)).item()
                    s.append(tok[1])
                    pair_feats.append(image_features.reshape(-1, 2, self.hidden_size)[idx])
                else:  # two random context images (may be the same person)
                    i1 = torch.randint(0, image_features.size(0), (1,)).item()
                    i2 = torch.randint(0, image_features.size(0), (1,)).item()
                    pair_feats.append(torch.stack([image_features[i1], image_features[i2]]))
                    s.append(tok[int(labels_e[i1] == labels_e[i2])])
            input_ids.append(s)
        input_ids = torch.tensor(input_ids, device=device).long()
        input_labels = input_ids.clone()
        input_labels[input_ids < 0] = -100
        selected = input_ids == -1
        input_ids[input_ids < 0] = 0

        feats = torch.cat(pair_feats).reshape(-1, self.hidden_size * 2)
        feats = self.mm_projector(feats.to(dtype=self.mm_projector.visual_proj.weight.dtype))
        emb = self.lm.get_input_embeddings()(input_ids).clone()
        emb[selected] = emb[selected] * 0 + feats.reshape(-1, self.lm.config.hidden_size).to(emb.dtype)
        icl_loss = self.lm(inputs_embeds=emb, labels=input_labels, use_cache=False).loss

        q = self.query_embeddings.to(dtype=emb.dtype).unsqueeze(0).expand(emb.size(0), -1, -1)
        out = self.lm(inputs_embeds=torch.cat([emb, q], dim=1), use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -self.args.num_vpt_tokens * self.num_layers:]
        return self.prompt_mlp(h.to(dtype=self.prompt_mlp.weight.dtype)), icl_loss

    def forward(self, image_crops, labels=None, prompts=None, domains=None):
        """Training / context forward (labels given, prompts None): the batch (pairs of one identity) is the
        context. Retrieval forward (prompts given, e.g. at test time): plain prompted encoding."""
        nview = 1
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size()
            image_crops = image_crops.reshape(-1, nc, h, w)
        image_crops = image_crops.to(dtype=self.encoder.patch_embed.proj.weight.dtype)
        zero = torch.tensor(0.0, device=image_crops.device)
        icl_loss = zero
        if labels is not None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
        if labels is not None and prompts is None:
            ctx, ctx_labels = image_crops[:256], labels[:256]
            with torch.no_grad():
                ctx_feats = self.encoder_copy.forward_features(ctx)['x_norm_clstoken']
            prompts, icl_loss = self._icl_prompts(ctx_feats, ctx_labels, image_crops.device)

        raw, patch = insert_deep_prompts(self.encoder, image_crops, prompts, self.num_layers)
        x = F.normalize(raw, dim=-1)
        std = x.std(dim=0).mean()
        id_loss = ot_loss = ce_loss = zero
        if labels is not None:
            id_loss = self.loss(x, labels)
            if self.args.ot_loss_weight != 0:
                from ops.wpa import compute_wpa
                pairs, pair_labels = sample_pair(labels)
                ot_loss = compute_wpa(patch[pairs[:, 0]], patch[pairs[:, 1]], pair_labels.to(x.device))
        features = x
        if self.reid_head is not None:
            head = self.reid_head(raw, labels, compute_ce=labels is not None and self.training
                                  and getattr(self.args, "ce_loss_weight", 0.0) > 0)
            features, ce_loss = head["feat"], head.get("ce", zero)
        loss = (icl_loss * self.args.icl_loss_weight + id_loss + ot_loss * self.args.ot_loss_weight
                + ce_loss * getattr(self.args, "ce_loss_weight", 0.0))
        return {"loss": loss, "id_loss": id_loss, "ce_loss": ce_loss, "ot_loss": ot_loss, "icl_loss": icl_loss,
                "features": features, "prompts": prompts, "std": std}
