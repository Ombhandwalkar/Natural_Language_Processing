# LLM Process of Token Generation-

    LLM generates the next token on the basis of probabilities calculated from previous tokens.

    Key and value states are created for each token, which is used in the scaled dot-product attention to compute the next token.

Problem-

        So the problem is we need to recalculate attention for all previous tokens for the new token.

        This becomes a very computationally expensive increase in the sequence tokens.




# KV Cache (Traditional Method)

    Instead of recalculating Keys and Values for old tokens, we cache the Keys and Values after first calculating them.

    Then for the next token, we only compute the attention for the next token using the cached values.




# Issues with KV Cache –

    It stores the cache in contiguous (continuous) memory blocks.

    So in parallel requests, it requires its individual KV cache.

    By allocating big contiguous blocks separately for each request, it becomes wasteful and impractical.

    This will increase internal fragmentation (unused memory inside blocks), external fragmentation (wasting free memory in small pieces), and out-of-memory errors.




# GPU memory-

    * Model weights themselves require 65% of GPU memory.

    * KV Cache dynamically requires up to 30% of storage as per the increase in requests.

    Ex- Suppose you have 100kb of GPU memory.

    65kb is used by model weights, and 35kb remains.

    For each parallel processing, we require a KV block. Due to this system, after certain requests, it will run out of memory errors.




# Solution – vLLM –

 Paged Attention

    Instead of storing all KV matrices in big contiguous blocks, we store blocks in pages.

    Each page holds a fixed number of tokens.

    Pages can be anywhere in GPU memory.

    We only maintain the mapping (indexing) of the metrics to know where each value is stored.

 

  How it helps –

    * It reduces memory waste.

    * supports more parallel requests

    * Avoid out-of-memory error issues.