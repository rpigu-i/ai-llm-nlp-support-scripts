from self_attention_v2 import SelfAttention_v2
import torch

torch.manual_seed(789)
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your    (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]] # step    (x^6)
)
d_in = inputs.shape[1] # input embedding soze
d_out = 2 # output embedding size 
sa_v2 = SelfAttention_v2(d_in, d_out)
print (sa_v2(inputs))

print ("Appying causal attention mask")

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

print ("Attention weights")
print (attn_weights)

print ("Apply mask")
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print (mask_simple) 
