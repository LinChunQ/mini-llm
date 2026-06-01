import torch
import torch.nn as nn

# 1. 核心配置文件（模拟 config.json）
class MiniLLMConfig:
    vocab_size = 4096       # 极小词表
    hidden_size = 256       # 隐藏层维度
    num_hidden_layers = 4   # 只堆叠 4 层 Transformer
    num_attention_heads = 8 # 8个注意力头
    max_position_embeddings = 512 # 最大上下文长度

# 2. 单个 Transformer 层（对应大模型的一层，内部包含注意力FNN和残差连接）
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 现代大模型标准的 Pre-Norm 结构
        self.input_layernorm = nn.RMSNorm(config.hidden_size) 
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        
        # 核心：自注意力层与前馈网络层
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(), # SwiGLU 的简化版
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def forward(self, x):
        # 残差连接 1：Attention
        x = x + self.self_attn(self.input_layernorm(x), self.input_layernorm(x), self.input_layernorm(x))[0]
        # 残差连接 2：MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

# 3. 完整的模型（对应整个 model.safetensors 映射出来的网络）
class MiniModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 词嵌入层：把 Token ID 变成稠密向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 堆叠多个 Transformer 块
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        # 最后的输出层：把向量再变回词表大小的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # 经典的 Causal LM 错位预测（自回归训练核心）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return {"logits": logits, "loss": loss}