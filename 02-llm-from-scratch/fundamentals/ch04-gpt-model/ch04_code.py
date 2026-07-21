"""
Chapter 4: GPT Model Architecture
Run this file to see each component in action.
"""

import torch
import torch.nn as nn
import tiktoken


# ===========================================
# 1. LayerNorm
# ===========================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


def demo_layernorm():
    print("\n" + "="*60)
    print("1. LayerNorm Demo")
    print("="*60)
    
    torch.manual_seed(123)
    batch = torch.randn(2, 5)
    
    print(f"Input:\n{batch}")
    print(f"\nMean per row: {batch.mean(dim=-1)}")
    print(f"Var per row:  {batch.var(dim=-1, unbiased=False)}")
    
    layer = LayerNorm(5)
    normalized = layer(batch)
    
    print(f"\nAfter LayerNorm:\n{normalized}")
    print(f"\nMean after: {normalized.mean(dim=-1)}")
    print(f"Var after:  {normalized.var(dim=-1, unbiased=False)}")
    print(f"\nLearned scale: {layer.scale}")
    print(f"Learned shift: {layer.shift}")


# ===========================================
# 2. GELU
# ===========================================
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


def demo_gelu():
    print("\n" + "="*60)
    print("2. GELU vs ReLU Demo")
    print("="*60)
    
    gelu = GELU()
    relu = nn.ReLU()
    
    x = torch.linspace(-3, 3, 7)
    
    print(f"Input:        {x.tolist()}")
    print(f"ReLU output:  {relu(x).tolist()}")
    print(f"GELU output:  {gelu(x).tolist()}")
    
    print("\nNote: GELU is smooth and allows small negative values through")


# ===========================================
# 3. FeedForward
# ===========================================
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


def demo_feedforward():
    print("\n" + "="*60)
    print("3. FeedForward Demo")
    print("="*60)
    
    GPT_CONFIG_124M = {
        "emb_dim": 768
    }
    
    torch.manual_seed(123)
    ff = FeedForward(GPT_CONFIG_124M)
    
    x = torch.randn(1, 4, 768)
    out = ff(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"After Linear(768 → 3072): (1, 4, 3072)")
    print(f"After GELU:   (1, 4, 3072)")
    print(f"After Linear(3072 → 768): {out.shape}")
    print(f"\nExpansion ratio: 4x (768 → 3072 → 768)")


# ===========================================
# 4. MultiHeadAttention (from Ch03)
# ===========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


# ===========================================
# 4. TransformerBlock
# ===========================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


def demo_transformer_block():
    print("\n" + "="*60)
    print("4. TransformerBlock Demo")
    print("="*60)
    
    GPT_CONFIG_124M = {
        "emb_dim": 768,
        "context_length": 1024,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    torch.manual_seed(123)
    block = TransformerBlock(GPT_CONFIG_124M)
    
    x = torch.randn(1, 4, 768)
    out = block(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("\nShape preserved! (batch, seq_len, emb_dim)")
    
    print("\nTransformerBlock components:")
    print(f"  - MultiHeadAttention: {sum(p.numel() for p in block.att.parameters()):,} params")
    print(f"  - FeedForward:        {sum(p.numel() for p in block.ff.parameters()):,} params")
    print(f"  - LayerNorm (x2):     {sum(p.numel() for p in block.norm1.parameters()) + sum(p.numel() for p in block.norm2.parameters()):,} params")
    print(f"  - Total:              {sum(p.numel() for p in block.parameters()):,} params")


# ===========================================
# 5. GPTModel
# ===========================================
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # 权重绑定（weight tying）：输出头与词元嵌入共享权重
        # 这样做的理由：
        # 1. 大幅减少参数量（vocab_size * emb_dim 这一大块只算一次）
        # 2. 输入端（把 token id 映射到向量）和输出端（把向量映射回 vocab 概率）
        #    本质是同一套语义空间的双向映射，共享权重让二者学得更一致
        # 3. GPT-2 / GPT-3 等模型都采用此做法
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def demo_gpt_model():
    print("\n" + "="*60)
    print("5. GPTModel Demo (Full Architecture)")
    print("="*60)
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    
    token_ids = torch.randint(0, 50257, (2, 4))
    
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Sample tokens: {token_ids[0].tolist()}")
    
    logits = model(token_ids)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  batch_size=2, seq_len=4, vocab_size=50257")
    
    print("\n" + "-"*60)
    print("Parameter breakdown:")
    print("-"*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print(f"\n  Token embedding:   {sum(p.numel() for p in model.tok_emb.parameters()):,}")
    print(f"  Position embedding: {sum(p.numel() for p in model.pos_emb.parameters()):,}")
    
    trf_params = sum(p.numel() for p in model.trf_blocks.parameters())
    print(f"  Transformer blocks: {trf_params:,} ({GPT_CONFIG_124M['n_layers']} blocks)")
    
    print(f"  Final LayerNorm:   {sum(p.numel() for p in model.final_norm.parameters()):,}")
    print(f"  Output head:       {sum(p.numel() for p in model.out_head.parameters()):,}")
    
    print("\nNote: Output head shares weights with token embedding (weight tying)")
    print("      So we don't count it twice in the official parameter count")


# ===========================================
# 6. Text Generation
# ===========================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def demo_text_generation():
    print("\n" + "="*60)
    print("6. Text Generation Demo")
    print("="*60)
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"Input text: '{start_context}'")
    print(f"Encoded: {encoded}")
    print(f"Tensor shape: {encoded_tensor.shape}")
    
    print("\nGenerating 10 new tokens...")
    output = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    decoded = tokenizer.decode(output.squeeze(0).tolist())
    
    print(f"\nOutput token IDs: {output.squeeze(0).tolist()}")
    print(f"Decoded text: '{decoded}'")
    
    print("\nNote: Output is random because model is not trained yet!")
    print("      After pretraining, it would generate coherent text.")


# ===========================================
# Main
# ===========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Chapter 4: GPT Model Architecture")
    print("="*60)
    
    demo_layernorm()
    demo_gelu()
    demo_feedforward()
    demo_transformer_block()
    demo_gpt_model()
    demo_text_generation()
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)
    print("\nKey takeaways:")
    print("1. LayerNorm normalizes across embedding dimension")
    print("2. GELU is a smooth activation function")
    print("3. FeedForward expands 4x then contracts")
    print("4. TransformerBlock = Attention + FFN + Residuals")
    print("5. GPTModel = Embeddings + 12 TransformerBlocks")
    print("6. Generation is autoregressive (one token at a time)")
