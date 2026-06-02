import torch
import torch.nn as nn


class MiniLLMConfig:
    vocab_size = 4096
    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 8
    max_position_embeddings = 512
    pad_token_id = 0


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        residual = x
        norm_x = self.input_layernorm(x)
        attn_out, _ = self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=pad_token_id,
        )
        self.embed_positions = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)

        x = self.embed_tokens(input_ids) + self.embed_positions(position_ids)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask

        for layer in self.layers:
            x = layer(
                x,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss}
