# 1 MHA(multi head attention)
Google 的团队在 2017 年提出的一种 NLP 经典模型：Attention Is All You Need ，首次提出并使用了 Self-Attention 机制，也就是 Multi Head Attention。<br>
- [Attention is All You Need 论文](https://arxiv.org/pdf/1706.03762.pdf)

# 2 KV Cache
假设 K 和 V 能直接存在缓存中，模型规模小还好，一旦模型规模很大长度很长时，KV 根本就存不进缓存。<br>

# 3 MQA（Multi Query Attention）
MQA（Multi Query Attention）最早是出现在2019年谷歌的一篇论文 《Fast Transformer Decoding: One Write-Head is All You Need》，之所以没有被关注到，是因为文本生成类任务还没这么火热，解码序列长度也没有现阶段大模型的要求那么高。<br>
- [MQA 论文](https://arxiv.org/abs/1911.02150)

如上对比，在 Multi-Query Attention 方法中只会保留一个单独的key-value头，这样虽然可以提升推理的速度，但是会带来精度上的损失。《Multi-Head Attention:Collaborate Instead of Concatenate 》这篇论文的第一个思路是基于多个 MQA 的 checkpoint 进行 finetuning，来得到了一个质量更高的 MQA 模型。这个过程也被称为 Uptraining。

- [MQA update](https://arxiv.org/pdf/2006.16362.pdf)

# 4 GQA（Grouped Query Attention）
Google 在 2023 年发表的一篇 《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》的论文，整体论文写的清晰易读。

- [GQA 论文](https://arxiv.org/pdf/2305.13245.pdf)

# 5 FlashAttention: 
- [FlashAttention 论文链接](https://arxiv.org/abs/2205.14135)

# 6 FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- [FlashAttention2 论文链接](https://arxiv.org/pdf/2307.08691.pdf)

# 7 PagedAttention
- [参考链接](https://blog.vllm.ai/2023/06/20/vllm.html)
- [page attention 论文链接](https://arxiv.org/abs/2309.06180)

# 8 Grouped-query attention
- [GQA 论文](https://arxiv.org/pdf/2305.13245.pdf)

# 9 参考链接
- [参考链接1](https://zhuanlan.zhihu.com/p/647130255)
- [参考链接](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
