import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM



class SimpleQFormer(nn.Module):
    """A minimal Q-Former: learnable queries cross-attend to visual tokens.

    Inputs are expected as concatenated pair features of shape (B, hidden_size * 2).
    The module reshapes to 2 visual tokens per sample, projects to the Q-Former
    hidden size, and runs a small Transformer decoder over learnable queries with
    cross-attention to the visual tokens, returning (B, num_query_tokens, out_dim).
    """

    def __init__(self,
                 visual_token_dim: int,
                 num_visual_tokens: int,
                 num_query_tokens: int,
                 qformer_hidden_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 out_dim = None):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_visual_tokens = num_visual_tokens
        self.hidden_dim = qformer_hidden_dim
        self.out_dim = out_dim if out_dim is not None else qformer_hidden_dim

        # Project visual tokens to Q-Former hidden size
        self.visual_proj = nn.Linear(visual_token_dim, qformer_hidden_dim)

        # Learnable query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(num_query_tokens, qformer_hidden_dim) * 0.02)

        # Transformer decoder layers with cross-attention over visual tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=qformer_hidden_dim,
            nhead=num_heads,
            dim_feedforward=qformer_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(qformer_hidden_dim)

        # Output projection if LM embedding dim differs
        self.output_proj = None
        if self.out_dim != qformer_hidden_dim:
            self.output_proj = nn.Linear(qformer_hidden_dim, self.out_dim)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """pair_features: (B, visual_token_dim * num_visual_tokens)
        Returns: (B, num_query_tokens, out_dim)
        """
        bsz = pair_features.size(0)

        # Recover visual tokens: (B, num_visual_tokens, visual_token_dim)
        visual_tokens = pair_features.view(bsz, self.num_visual_tokens, -1)
        # Project to hidden dim and switch to (S, B, D)
        visual_tokens = self.visual_proj(visual_tokens)  # (B, S, D)
        memory = visual_tokens.transpose(0, 1).contiguous()  # (S, B, D)

        # Prepare queries: (T, B, D)
        query = self.query_embeddings.unsqueeze(1).expand(-1, bsz, -1)  # (T, B, D)

        # Decoder with cross-attention
        out = self.decoder(tgt=query, memory=memory)  # (T, B, D)
        out = self.norm(out)  # (T, B, D)
        out = out.transpose(0, 1).contiguous()  # (B, T, D)

        if self.output_proj is not None:
            out = self.output_proj(out)  # (B, T, out_dim)

        return out

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        encoder = self._load_dinov2(args.vision_model).eval()
        encoder.requires_grad_(False)
        self.encoder = encoder

        encoder_copy = self._load_dinov2(args.vision_model).eval()
        encoder_copy.requires_grad_(False)
        self.encoder_copy = encoder_copy

        # FERReID: LoRA placement / rank or full fine-tuning follow the args; the defaults
        # (LoRA r=128 on the qkv of the last 4 blocks) are the original VICP configuration.
        from ops.adapt import adapt_encoder
        adapt_encoder(self.encoder, args)

        from ops.losses import HardTripletLoss
        self.loss = HardTripletLoss(margin=getattr(args, "triplet_margin", 0.1), hardest=True)
        # FERReID: optional BNNeck + ID classification head (None by default = original VICP)
        from adapters.reid_head import build_head
        self.reid_head = build_head(args, self.encoder.embed_dim)

        self.num_layers = len(encoder.blocks)
        llm_model = self._resolve_llm_model(args.llm_model)
        self.lm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.lm.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        self.hidden_size = self.encoder.embed_dim
        # Replace MLP projector with a learnable Q-Former
        self.mm_projector = SimpleQFormer(
            visual_token_dim=self.hidden_size,
            num_visual_tokens=2,
            num_query_tokens=self.args.num_id_tokens,
            qformer_hidden_dim=self.lm.config.hidden_size,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
            out_dim=self.lm.config.hidden_size,
        )
        self.query_embeddings = nn.Parameter(torch.randn(self.args.num_vpt_tokens * self.num_layers, self.lm.config.hidden_size) * 0.02)
        self.prompt_mlp = nn.Linear(self.lm.config.hidden_size, self.hidden_size, bias=False)
        self.prompt_mlp.weight.data.zero_()

        # FERReID direction A (--prompt_mode residual): prompt = base + gate * delta(context).
        #   base : one learnable prompt shared by every domain (a VPT prompt, zero-initialised)
        #   delta: prompt_mlp(LayerNorm(h - mean_h)), h = LLM hidden states at the query tokens;
        #          with --ctx_center ema, mean_h is a running mean of h, so the constant part of h is
        #          removed and delta can only carry what *changes* with the context
        #   gate : one learnable scalar per ViT layer (--ctx_gate_init)
        # prompt_mlp is initialised small instead of zero (a zero map with a zero-mean input has no
        # gradient path worth mentioning). The default --prompt_mode vicp keeps the original prompt.
        self.prompt_mode = getattr(args, "prompt_mode", "vicp")
        if self.prompt_mode == "residual":
            n_tok = self.args.num_vpt_tokens * self.num_layers
            self.base_prompt = nn.Parameter(torch.zeros(1, n_tok, self.hidden_size))
            self.delta_norm = nn.LayerNorm(self.lm.config.hidden_size)
            nn.init.normal_(self.prompt_mlp.weight, std=getattr(args, "delta_init_std", 0.02))
            self.ctx_gate = nn.Parameter(torch.full((self.num_layers, 1, 1), float(getattr(args, "ctx_gate_init", 0.1))))
            if getattr(args, "ctx_center", "none") == "ema":
                self.register_buffer("h_mean", torch.zeros(1, n_tok, self.lm.config.hidden_size))
                self.register_buffer("h_mean_count", torch.zeros(()))
        elif self.prompt_mode != "vicp":
            raise ValueError("--prompt_mode must be vicp or residual")

    @staticmethod
    def _load_dinov2(model_name):
        local_repo = os.environ.get("DINOV2_REPO", "/root/basic-models/dinov2")
        if os.path.isdir(local_repo):
            return torch.hub.load(local_repo, model_name, source="local")
        return torch.hub.load("facebookresearch/dinov2", model_name)

    @staticmethod
    def _resolve_llm_model(model_name):
        local_qwen = os.environ.get("QWEN3_06B_DIR", "/root/basic-models/Qwen3-0.6B")
        if model_name == "Qwen/Qwen3-0.6B" and os.path.isdir(local_qwen):
            return local_qwen
        return model_name

    def _prompts_from_hidden(self, h):
        """LLM hidden states at the query tokens (B, L*V, H_lm) -> visual prompts (B, L*V, D)."""
        if self.prompt_mode != "residual":  # original VICP
            return self.prompt_mlp(h.to(dtype=self.prompt_mlp.weight.dtype))
        h = h.float()
        if hasattr(self, "h_mean"):
            if self.training:
                with torch.no_grad():
                    mean = h.mean(0, keepdim=True)
                    if self.h_mean_count.item() == 0:
                        self.h_mean.copy_(mean)
                    else:
                        self.h_mean.mul_(0.99).add_(0.01 * mean)
                    self.h_mean_count.add_(1)
            h = h - self.h_mean.float()
        dtype = self.prompt_mlp.weight.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            delta = self.prompt_mlp(self.delta_norm.float()(h).to(dtype)).float()
        B, T, D = delta.shape
        delta = delta.reshape(B, self.num_layers, T // self.num_layers, D) * self.ctx_gate.float()
        return self.base_prompt.float() + delta.reshape(B, T, D)

    @torch.no_grad()
    def _select_context_ids(self, image_crops, nview, n_ctx, method):
        """Identity order (selected context identities first) chosen by a label-free selector over the
        batch: one feature per identity = mean of its images' CLS features (trained encoder, no prompt,
        no gradient, as at test time). No camera ids are
        available in a batch, so camera-based selectors fall back to their camera-free behaviour."""
        import numpy as np
        from adapters.context_selection import select_images

        class _BatchPool:  # the minimal ImagePool interface the selectors use
            def __init__(self, feats):
                self.paths, self.camids, self.has_cameras = [None] * len(feats), [0] * len(feats), False
                self.feats, self.style, self._sel_stats = feats, None, None

            def __len__(self):
                return len(self.paths)

        if method == "style_cover":
            raise ValueError("--train_context_selector style_cover is not supported (no style features in a batch)")
        f = self.encoder.forward_features(image_crops)["x_norm_clstoken"].float()  # trained encoder, no prompt
        f = F.normalize(F.normalize(f, dim=-1).reshape(-1, nview, f.size(-1)).mean(1), dim=-1)
        rng = np.random.RandomState(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        chosen = select_images(method, _BatchPool(f.cpu().numpy()), n_ctx, rng)
        rest = [i for i in range(f.size(0)) if i not in set(chosen)]
        return torch.tensor(chosen + rest)

    def base_prompts(self):
        """Context-free prompt of --prompt_mode residual (what the model does without any context), shaped
        like the prompts returned by forward(); None for the original VICP prompt."""
        if self.prompt_mode != "residual":
            return None
        return self.base_prompt.reshape(1, self.num_layers, self.args.num_vpt_tokens, -1)

    def sample_pair(self, labels):
                    # 获取样本数量和索引
        labels = labels.cpu()
        n_samples = len(labels)
        indices = torch.arange(n_samples)

        # 构建索引对 (i, j)，避免重复和自身配对
        i_idx, j_idx = torch.triu_indices(n_samples, n_samples, offset=1)

        # 比较标签，确定正负样本对
        is_positive = labels[i_idx] == labels[j_idx]

        # 分别获取正负样本对的索引
        positive_pairs = torch.stack((i_idx[is_positive], j_idx[is_positive]), dim=1)
        negative_pairs = torch.stack((i_idx[~is_positive], j_idx[~is_positive]), dim=1)

        # 确保正负样本对数量相等
        # min_pairs = min(len(positive_pairs), len(negative_pairs))
        min_pairs = min(128, len(positive_pairs), len(negative_pairs))

        # 随机采样（确保正负样本数量一致）
        positive_sampled = positive_pairs[torch.randperm(len(positive_pairs))[:min_pairs]]
        negative_sampled = negative_pairs[torch.randperm(len(negative_pairs))[:min_pairs]]

        # 为正负样本对分配标签（正样本对为1，负样本对为0）
        positive_labels = torch.ones(min_pairs, dtype=torch.long)
        negative_labels = torch.zeros(min_pairs, dtype=torch.long)

        # 合并正负样本对及其标签
        all_pairs = torch.cat((positive_sampled, negative_sampled), dim=0)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)
        return all_pairs, all_labels


    def forward(self,
                image_crops,
                labels=None,
                prompts=None,
                ):
        encoder_dtype = self.encoder.patch_embed.proj.weight.dtype
        if image_crops.ndim == 5:
            bs, nview, nc, h, w = image_crops.size()
            image_crops = image_crops.reshape(-1, nc, h, w)
        image_crops = image_crops.to(dtype=encoder_dtype)

        icl_loss = torch.tensor(0.0)

        if labels is not None and prompts is None:
            labels = labels.unsqueeze(1).expand(-1, nview).reshape(-1)
            # FERReID direction A (--episode_context_ids N): in training, the first N identities of the
            # batch are the context (questions for the LLM) and only the *other* identities are
            # retrieved with the resulting prompt, as at test time (context people != query people).
            # N is drawn uniformly from [--episode_context_ids_min, N] when the minimum is > 0.
            ctx_crops, ctx_labels = image_crops, labels
            n_ctx = getattr(self.args, "episode_context_ids", 0)
            if self.training and n_ctx > 0:
                n_min = getattr(self.args, "episode_context_ids_min", 0)
                if 0 < n_min < n_ctx:
                    n_ctx = int(torch.randint(n_min, n_ctx + 1, (1,)).item())
                # direction B phase 2 (--train_context_selector): a label-free selector, instead of the
                # batch order, decides which identities of the batch form the context
                sel = getattr(self.args, "train_context_selector", "random")
                if sel != "random":
                    order = self._select_context_ids(image_crops, nview, n_ctx, sel)
                    img_order = (order[:, None] * nview + torch.arange(nview)).reshape(-1).to(image_crops.device)
                    image_crops, labels = image_crops[img_order], labels[img_order]
                cut = n_ctx * nview
                if cut >= image_crops.size(0):
                    raise ValueError("episode_context_ids must leave identities to retrieve in the batch")
                ctx_crops, ctx_labels = image_crops[:cut], labels[:cut]
                image_crops, labels = image_crops[cut:], labels[cut:]
            # clip_image_crops = clip_image_crops.reshape(-1, nc, clip_image_crops.size(-2), clip_image_crops.size(-1))
            num_examples = 256
            clip_image_crops_e = ctx_crops[:num_examples]
            labels_e = ctx_labels[:num_examples]
            with torch.no_grad():
                image_features = self.encoder_copy.forward_features(clip_image_crops_e)['x_norm_clstoken']

            input_ids = []
            new_image_features = []
            yes_token_id = self.tokenizer.convert_tokens_to_ids('yes')
            no_token_id = self.tokenizer.convert_tokens_to_ids('no')
            tokenmaps = {1: yes_token_id, 0: no_token_id}
            for i in range(self.args.num_icl_bs):
                s = []
                for j in range(self.args.num_icl_samples):
                    s.extend([-1] * self.args.num_id_tokens)
                    x = torch.randint(0, 2, (1,)).item()
                    if x == 1:
                        idx = torch.randint(0, image_features.size(0) // 2, (1,)).item()
                        s.append(tokenmaps[x])
                        new_image_features.append(image_features.reshape(-1, 2, self.hidden_size)[idx])
                    elif x == 0:
                        i1 = torch.randint(0, image_features.size(0), (1,)).item()
                        i2 = torch.randint(0, image_features.size(0), (1,)).item()
                        new_image_features.append(torch.stack([image_features[i1], image_features[i2]]))
                        s.append(tokenmaps[int(labels_e[i1] == labels_e[i2])])
                        # s.append(int(labels_e[i1] == labels_e[i2]))
                    else:
                        raise ValueError
                input_ids.append(s)
            input_ids = torch.tensor(input_ids, device=image_crops.device).long()  # FERReID: was .cuda()
            input_labels = input_ids.clone()
            input_labels[input_ids < 0] = -100
        
            selected = input_ids == -1
            input_ids[input_ids < 0] = 0

            image_features = torch.cat(new_image_features)
            image_features = image_features.reshape(-1, self.hidden_size * 2)
            # Q-Former returns (B, num_id_tokens, lm_word_emb_dim)
            image_features = image_features.to(dtype=self.mm_projector.visual_proj.weight.dtype)
            image_features = self.mm_projector(image_features)
            input_embeddings = self.lm.get_input_embeddings()(input_ids).clone()

            image_features = image_features.reshape(-1, self.lm.config.hidden_size)
            input_embeddings[selected] = input_embeddings[selected] * 0 + image_features.to(input_embeddings.dtype)
            outputs = self.lm(inputs_embeds=input_embeddings, labels=input_labels, use_cache=False)

            icl_loss = outputs.loss

            query_embeddings = self.query_embeddings.to(dtype=input_embeddings.dtype)
            input_embeddings2 = torch.cat([input_embeddings, query_embeddings.unsqueeze(0).expand(input_embeddings.size(0), -1, -1)], dim=1)
            outputs2 = self.lm(inputs_embeds=input_embeddings2, use_cache=False, output_hidden_states=True)
            prompts = outputs2.hidden_states[-1][:, -self.args.num_vpt_tokens * self.num_layers:]
            prompts = self._prompts_from_hidden(prompts)

        ot_loss = torch.tensor(0.0)

        prompts = prompts.reshape(prompts.size(0), self.num_layers, self.args.num_vpt_tokens, -1)
        prompts = prompts[torch.randint(0, prompts.size(0), (image_crops.size(0),))]

        x = self.encoder.prepare_tokens_with_masks(image_crops, None)
        prompts = prompts.to(dtype=x.dtype)
        for blk in self.encoder.blocks[:-self.num_layers]:
            x = blk(x)
        for i, blk in enumerate(self.encoder.blocks[-self.num_layers:]):
            prompts_ = prompts[:, i]
            if i == 0:
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1:]], dim=1)
            else:
                x = torch.cat([x[:, 0].unsqueeze(1), prompts_, x[:, 1+prompts_.size(1):]], dim=1)
            x = blk(x)
        x = self.encoder.norm(x)
        patch_features = x[:, 1+prompts_.size(1):]
        # patch_features = x[:, 1:]
        x = x[:, 0]
        raw = x

        x = F.normalize(x, dim=-1)
        std = x.std(dim=0).mean()
        if labels is not None:
            id_loss = self.loss(x, labels)
            if self.args.ot_loss_weight != 0:  # FERReID: skip the (unused) WPA computation at weight 0
                from ops.wpa import compute_wpa
                all_pairs, all_labels = self.sample_pair(labels)
                ot_loss = compute_wpa(patch_features[all_pairs[:, 0]], patch_features[all_pairs[:, 1]], all_labels.to(x.device))

        else:
            id_loss = torch.tensor(0.0)

        # FERReID: BNNeck / ID classification head. The CE term is only computed in training mode:
        # in the context forward at test time the labels are context-pair indices, not source classes.
        ce_loss = torch.tensor(0.0)
        features = x
        if self.reid_head is not None:
            head = self.reid_head(raw, labels, compute_ce=labels is not None and self.training
                                  and getattr(self.args, "ce_loss_weight", 0.0) > 0)
            features = head["feat"]
            ce_loss = head.get("ce", ce_loss)

        loss = (icl_loss * self.args.icl_loss_weight + id_loss + ot_loss * self.args.ot_loss_weight
                + ce_loss * getattr(self.args, "ce_loss_weight", 0.0))
        x = features

        outputs = {
            'loss': loss,
            'id_loss': id_loss,
            'ce_loss': ce_loss,
            'ot_loss': ot_loss,
            'icl_loss': icl_loss,
            'features': x,
            'prompts': prompts,
            'std': std,
        }

        return outputs

        
if __name__ == '__main__':
    import easydict
    EasyDict = easydict.EasyDict
    args = EasyDict()
    # args.vision_model = 'dinov2_vits14'
    args.vision_model = 'dinov2_vitb14'
    args.llm_model = 'Qwen/Qwen3-0.6B'
    args.num_id_tokens = 4
    args.num_vpt_tokens = 2
    args.num_icl_samples = 64
    args.num_icl_bs = 8
    args.icl_loss_weight = 0.0
    args.ot_loss_weight = 0.0
    model = Model(args).cuda()
    image_crops = torch.randn(2, 2, 3, 224, 224).cuda()
    labels = torch.randint(0, 10, (2,)).cuda()
    prompts = None
    outputs = model(image_crops, labels, prompts)
    # print(outputs)
    exit(0)
