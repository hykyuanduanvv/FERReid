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
            # clip_image_crops = clip_image_crops.reshape(-1, nc, clip_image_crops.size(-2), clip_image_crops.size(-1))
            num_examples = 256
            clip_image_crops_e = image_crops[:num_examples]
            labels_e = labels[:num_examples]
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
            prompts = prompts.to(dtype=self.prompt_mlp.weight.dtype)
            prompts = self.prompt_mlp(prompts)

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
