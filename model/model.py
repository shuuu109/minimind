from logging.config import stopListening
import math, torch, torch.nn.functional as F
from typing import Optional
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# MiniMind Config
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# RMSNorm
# 继承自torch.nn.Module, __init__初始化参数，_norm方法， forward方法定义前向传播逻辑
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


# YaRN
def precompute_freqs_cis(dim: int, end: int=32 * 1024, rope_base: float = 10000.0, rope_scaling: Optional[dict]=None):
    # 初始化RoPE频率 theta_i = base^(-2i/dim)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
        rope_scaling.get("original_max_position_embeddings", 2048),
        rope_scaling.get("factor", 16),
        rope_scaling.get("beta_fast", 32),
        rope_scaling.get("beta_slow", 1),
        rope_scaling.get("attention_factor", 1.0)
        )

        # 推断长度大于训练长度，使用缩放
        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_slow)), 0), min(math.ceil(inv_dim(beta_fast)), dim // 2)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 计算频率的余弦和正弦值
    t = torch.arange(end, device=freqs.device, dtype=freqs.dtype)
    freqs = torch.outer(t,freqs).float()  # [end, dim // 2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([-torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# 应用旋转位置编码到query和key张量
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

# 为 GQA 复制 KV 张量
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # batch_size, seq_len, num_key_value_heads, head_dim = x.shape
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

# Attention Module
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads  # query的head数
        self.n_local_kv_heads = self.num_key_value_heads # key和value的head数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个key/value head对应的query head数
        self.head_dim = config.head_dim # 通常为 hidden_size / num_attention_heads
        self.is_causal = True  # decoder-only，永远使用因果 mask
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # QK-norm 按 head 维度（head_dim）归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x: [batch_size, seq_len, hidden_size]
        # position_embeddings: (cos, sin)，各为 [seq_len, head_dim]（对应当前这批 token 的绝对位置）
        # past_key_value: (past_key, past_value)，各为 [batch_size, num_key_value_heads, past_seq_len, head_dim]
        # use_cache: 是否返回更新后的 key/value 用于缓存
        # attention_mask: [batch_size, 1, 1, kv_len]，1 表示保留，0 表示屏蔽（padding）

        bsz, seq_len, _ = x.size()

        # 1. 线性投影，把 hidden 映射到 Q/K/V 空间
        xq = self.q_proj(x)  # [batch_size, seq_len, num_attention_heads * head_dim]
        xk = self.k_proj(x)  # [batch_size, seq_len, num_key_value_heads * head_dim]
        xv = self.v_proj(x)  # [batch_size, seq_len, num_key_value_heads * head_dim]

        # 2. 拆出 head 维度：最后一维从 heads*head_dim 变成 (heads, head_dim)
        q = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        k = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        v = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. QK-norm：对每个 head 的 head_dim 维做 RMSNorm（Qwen3 风格）
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE 旋转位置编码
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 5. KV cache：把历史 K/V 拼到序列维度最前面
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present_key_value = (k, v) if use_cache else None

        # 6. 转成 [batch_size, heads, seq_len, head_dim]，并对 K/V 做 GQA 复制
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        # 7. 注意力计算
        # flash 路径要求 q/k 等长（is_causal 是方阵 mask），有 cache 时 q、k 长度不同，只能走手动路径
        if self.flash and past_key_value is None and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, heads, seq_len, kv_len]
            # 因果 mask：屏蔽“当前位置对未来位置”的注意力，只作用在最后 seq_len 列（兼容 cache）
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device, dtype=scores.dtype
            ).triu(1)
            if attention_mask is not None:
                scores = scores + (1.0 - attention_mask) * -1e9  # padding 掩码
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)
            attn_output = self.attn_dropout(attn_weights) @ v

        # 8. 合并 head 维度并做输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present_key_value

# SWIGLU Feedforward Module 
# FFN_SwiGLU(x) = down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) 

# MiniMind Transformer Layer 
class MiniMindBlock(nn.Module):
    def __init__(self, layerid: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # Self-Attention
        residual = hidden_states
        hidden_states, present_key_value= self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value    

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]，1 表示保留，0 表示屏蔽（padding）
        # past_key_values: list of (past_key, past_value)，各为 [batch_size, past_seq_len, num_key_value_heads, head_dim]
        # use_cache: 是否返回更新后的 key/value 用于缓存
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        if hasattr(past_key_values, 'layers'):past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 1. 词嵌入
        hidden_states = self.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)

        # 2. RoPE 旋转位置编码
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)    
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_len], self.freqs_sin[start_pos: start_pos + seq_len])

        # 3. 注意力 mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present_key_value)


        hidden_states = self.norm(hidden_states)
        # aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = sum([getattr(l.mlp, "aux_loss", 0.0) for l in self.layers], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=None, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    def generate(self, inputs = None, attention_mask=None, max_new_tokens = 8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], 
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(attention_mask.shape[0],1)], -1
            ) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty

            if top_k > 0: 
                    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
