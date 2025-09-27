from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the LayerNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
            bias (nn.Parameter): Learnable bias parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Compute layer normalization by subtracting the mean and dividing by 
        the standard deviation along the last dimension. Use the standard
        LayerNorm formula: (x - mean) / sqrt(variance + eps)

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # todo
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + self.eps)
        
        
    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying LayerNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight + self.bias
    

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor,
                                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). The query, key, and
        value tensors each have shape (bs, n_local_heads, seqlen, head_dim).
        An optimal implemention will jointly computing attention for multiple
        heads (n_local_heads of them) at once using matrix/tensor operations.

        Make sure to use attention_dropout (self.attn_dropout) on the computed
        attention matrix before applying it to the value tensor.
        '''
        # todo
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # attention_mask: (bs, 1, 1, seqlen) with 1 for real tokens, 0 for pads
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, value)
        return output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None:
            # broadcast to (bs, 1, 1, seqlen)
            attention_mask = attention_mask[:, None, None, :]

        output = self.compute_query_key_value_scores(query, key, value, attention_mask=attention_mask)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        '''
        This is the forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the
           output of the feed-forward network
        '''
        # todo
        # 1)
        x_norm1 = self.attention_norm(x)
        # 2)
        attn_out = self.attention(x_norm1, attention_mask=attention_mask)
        # 3) 3.1....
        x_res1 = x + attn_out
        # 3) 3.2....
        x_norm2 = self.ffn_norm(x_res1)
        # 4)
        ffn_out = self.feed_forward(x_norm2)
        # 5)
        output = x_res1 + ffn_out
        return output

class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature: float = 1.0,
        epsilon: float = 0.05,
        top_k: int = 0,
        top_p: float = 0.0,
        beam_size: int = 1,
        beam_alpha: float = 0.0,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using basic temperature sampling with epsilon sampling (i.e. 
        filtering out tokens with probability below the epsilon threshold at each timestep). 
        Most likely you'll want to make sure to be in model.eval() mode of operation for this. 
        Also note this is a super inefficient version of sampling with no key/value cache, 
        but you are free to add any optimizations on top of this.
        """
        # Helper: top-k filter on logits
        def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
            if k is None or k <= 0:
                return logits
            values, _ = torch.topk(logits, k, dim=-1)
            min_values = values[..., -1, None]
            return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

        # Helper: top-p (nucleus) filter on probabilities
        def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
            if p is None or p <= 0.0 or p >= 1.0:
                return probs
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # Keep tokens while cumulative prob <= p; always keep the first
            keep = cumsum <= p
            keep[..., 0] = True
            # Zero out the rest
            sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
            # Map back to original indices
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
            # Renormalize
            denom = new_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return new_probs / denom

        # Beam search path (batch size 1)
        if beam_size is not None and beam_size > 1:
            assert idx.size(0) == 1, "Beam search currently supports batch size 1"
            beams = [(idx, 0.0)]  # list of (seq_tensor, cumulative_log_prob)
            for step in range(max_new_tokens):
                candidates = []
                for seq, score in beams:
                    idx_cond = seq if seq.size(1) <= self.params.max_seq_len else seq[:, -self.params.max_seq_len:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :]
                    log_probs = F.log_softmax(logits / max(temperature, 1e-8), dim=-1)
                    # Expand top tokens for this beam
                    topv, topi = torch.topk(log_probs, beam_size, dim=-1)
                    for i in range(beam_size):
                        next_token = topi[:, i].view(1, 1)
                        next_seq = torch.cat([seq, next_token], dim=1)
                        next_len = next_seq.size(1)
                        new_score = score + topv[0, i].item()
                        if beam_alpha > 0.0:
                            norm_score = new_score / (next_len ** beam_alpha)
                        else:
                            norm_score = new_score
                        candidates.append((next_seq, new_score, norm_score))
                # Select top beams by normalized score
                candidates.sort(key=lambda x: x[2], reverse=True)
                beams = [(seq, score) for (seq, score, _) in candidates[:beam_size]]
            # Return the best beam (highest normalized score)
            best_seq = max(beams, key=lambda x: x[1] / (x[0].size(1) ** beam_alpha if beam_alpha > 0.0 else 1.0))[0]
            return best_seq

        # Sampling / greedy path
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            # todo
            if temperature == 0.0:
                # select the single most likely index
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                '''
                Perform temperature sampling with epsilon sampling:
                1) Scale the logits with the temperature followed by normalization using Softmax.
                2) Create a mask to filter out tokens with probability below epsilon threshold.
                3) Apply the mask to keep only tokens with probability >= epsilon.
                4) Renormalize the filtered probabilities so they sum to 1.
                5) Sample from this filtered probability distribution.
                (Optionally, also apply other sampling methods)
                '''
                logits_scaled = logits / temperature
                logits_scaled = apply_top_k(logits_scaled, top_k)
                probs = F.softmax(logits_scaled, dim=-1)
                if top_p and top_p > 0.0:
                    probs = apply_top_p(probs, top_p)
                elif epsilon and epsilon > 0.0:
                    mask = probs >= epsilon
                    filtered_probs = probs * mask
                    sums = filtered_probs.sum(dim=-1, keepdim=True)
                    need_fallback = sums == 0
                    if need_fallback.any():
                        # If no tokens pass the epsilon threshold, keep the argmax token
                        max_idx = torch.argmax(probs, dim=-1, keepdim=True)
                        fallback = torch.zeros_like(filtered_probs)
                        fallback.scatter_(1, max_idx, 1.0)
                        filtered_probs = torch.where(need_fallback, fallback, filtered_probs)
                        sums = filtered_probs.sum(dim=-1, keepdim=True)
                    probs = filtered_probs / sums.clamp_min(1e-12)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def load_pretrained(checkpoint):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
  dtype = "float32"

  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  # init from a model saved in a specific directory
  checkpoint_dict = torch.load(checkpoint, map_location=device)
  config = LlamaConfig(**checkpoint_dict['model_args'])
  model = Llama(config)
  state_dict = checkpoint_dict['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  return model
