https://r.jina.ai/https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

Title: The Big LLM Architecture Comparison

URL Source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

Published Time: 2025-07-19T11:11:10+00:00

Markdown Content:
It has been seven years since the original GPT architecture was developed. At first glance, looking back at GPT-2 (2019) and forward to DeepSeek-V3 and Llama 4 (2024-2025), one might be surprised at how structurally similar these models still are.

Sure, positional embeddings have evolved from absolute to rotational (RoPE), Multi-Head Attention has largely given way to Grouped-Query Attention, and the more efficient SwiGLU has replaced activation functions like GELU. But beneath these minor refinements, have we truly seen groundbreaking changes, or are we simply polishing the same architectural foundations?

Comparing LLMs to determine the key ingredients that contribute to their good (or not-so-good) performance is notoriously challenging: datasets, training techniques, and hyperparameters vary widely and are often not well documented.

However, I think that there is still a lot of value in examining the structural changes of the architectures themselves to see what LLM developers are up to in 2025. (A subset of them are shown in Figure 1 below.)

[![Image 1](https://substackcdn.com/image/fetch/$s_!iCn-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4ae4aa85-3e22-486c-9bd9-27edc4acbf8b_3000x2093.png)](https://substackcdn.com/image/fetch/$s_!iCn-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4ae4aa85-3e22-486c-9bd9-27edc4acbf8b_3000x2093.png)

Figure 1: A subset of the architectures covered in this article.

So, in this article, rather than writing about benchmark performance or training algorithms, I will focus on the architectural developments that define today's flagship open models.

(As you may remember, [I wrote about multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) not too long ago; in this article, I will focus on the text capabilities of recent models and leave the discussion of multimodal capabilities for another time.)

**Tip:** This is a fairly comprehensive article, so I recommend using the navigation bar to access the table of contents (just hover over the left side of the Substack page).

**Optional:** The video below is a narrated and abridged version of this article.

As you have probably heard more than once by now, [DeepSeek R1](https://arxiv.org/abs/2501.12948) made a big impact when it was released in January 2025. DeepSeek R1 is a reasoning model built on top of the [DeepSeek V3 architecture](https://arxiv.org/abs/2412.19437), which was introduced in December 2024.

While my focus here is on architectures released in 2025, I think it’s reasonable to include DeepSeek V3, since it only gained widespread attention and adoption following the launch of DeepSeek R1 in 2025.

If you are interested in the training of DeepSeek R1 specifically, you may also find my article from earlier this year useful:

In this section, I’ll focus on two key architectural techniques introduced in DeepSeek V3 that improved its computational efficiency and distinguish it from many other LLMs:

*   Multi-Head Latent Attention (MLA)

*   Mixture-of-Experts (MoE)

Before discussing Multi-Head Latent Attention (MLA), let's briefly go over some background to motivate why it's used. For that, let's start with Grouped-Query Attention (GQA), which has become the new standard replacement for a more compute- and parameter-efficient alternative to Multi-Head Attention (MHA) in recent years.

So, here's a brief GQA summary. Unlike MHA, where each head also has its own set of keys and values, to reduce memory usage, GQA groups multiple heads to share the same key and value projections.

For example, as further illustrated in Figure 2 below, if there are 2 key-value groups and 4 attention heads, then heads 1 and 2 might share one set of keys and values, while heads 3 and 4 share another. This reduces the total number of key and value computations, which leads to lower memory usage and improved efficiency (without noticeably affecting the modeling performance, according to ablation studies).

[![Image 2](https://substackcdn.com/image/fetch/$s_!uVhV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F647caf83-cd3d-46f8-8bd0-0946bd896ea1_1023x474.png)](https://substackcdn.com/image/fetch/$s_!uVhV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F647caf83-cd3d-46f8-8bd0-0946bd896ea1_1023x474.png)

_Figure 2: A comparison between MHA and GQA. Here, the group size is 2, where a key and value pair is shared among 2 queries._

So, the core idea behind GQA is to reduce the number of key and value heads by sharing them across multiple query heads. This (1) lowers the model's parameter count and (2) reduces the memory bandwidth usage for key and value tensors during inference since fewer keys and values need to be stored and retrieved from the KV cache.

(If you are curious how GQA looks in code, see my[GPT-2 to Llama 3 conversion guide](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb) for a version without KV cache and my KV-cache variant [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/pkg/llms_from_scratch/llama3.py).)

While GQA is mainly a computational-efficiency workaround for MHA, ablation studies (such as those in the[original GQA paper](https://arxiv.org/abs/2305.13245) and the [Llama 2 paper](https://arxiv.org/abs/2307.09288)) show it performs comparably to standard MHA in terms of LLM modeling performance.

Now, Multi-Head Latent Attention (MLA) offers a different memory-saving strategy that also pairs particularly well with KV caching. Instead of sharing key and value heads like GQA, MLA compresses the key and value tensors into a lower-dimensional space before storing them in the KV cache.

At inference time, these compressed tensors are projected back to their original size before being used, as shown in the Figure 3 below. This adds an extra matrix multiplication but reduces memory usage.

[![Image 3](https://substackcdn.com/image/fetch/$s_!jagJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb9a75be-2848-4b99-af3d-4c48bdd0181a_1550x858.png)](https://substackcdn.com/image/fetch/$s_!jagJ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Feb9a75be-2848-4b99-af3d-4c48bdd0181a_1550x858.png)

_Figure 3: Comparison between MLA (used in DeepSeek V3 and R1) and regular MHA._

(As a side note, the queries are also compressed, but only during training, not inference.)

By the way, MLA is not new in DeepSeek V3, as its [DeepSeek-V2 predecessor](https://arxiv.org/abs/2405.04434) also used (and even introduced) it. Also, the V2 paper contains a few interesting ablation studies that may explain why the DeepSeek team chose MLA over GQA (see Figure 4 below).

[![Image 4](https://substackcdn.com/image/fetch/$s_!efDX!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b7e646a-16c1-4245-9a3f-55a41f3070c2_903x856.png)](https://substackcdn.com/image/fetch/$s_!efDX!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2b7e646a-16c1-4245-9a3f-55a41f3070c2_903x856.png)

Figure 4: Annotated tables from the DeepSeek-V2 paper, https://arxiv.org/abs/2405.04434

As shown in Figure 4 above, GQA appears to perform worse than MHA, whereas MLA offers better modeling performance than MHA, which is likely why the DeepSeek team chose MLA over GQA. (It would have been interesting to see the "KV Cache per Token" savings comparison between MLA and GQA as well!)

To summarize this section before we move on to the next architecture component, MLA is a clever trick to reduce KV cache memory use while even slightly outperforming MHA in terms of modeling performance.

The other major architectural component in DeepSeek worth highlighting is its use of Mixture-of-Experts (MoE) layers. While DeepSeek did not invent MoE, it has seen a resurgence this year, and many of the architectures we will cover later also adopt it.

You are likely already familiar with MoE, but a quick recap may be helpful.

The core idea in MoE is to replace each FeedForward module in a transformer block with multiple expert layers, where each of these expert layers is also a FeedForward module. This means that we swap a single FeedForward block for multiple FeedForward blocks, as illustrated in the Figure 5 below.

[![Image 5](https://substackcdn.com/image/fetch/$s_!d_xI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F632d3212-432a-4d43-b271-f2269be1d8ec_1304x822.png)](https://substackcdn.com/image/fetch/$s_!d_xI!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F632d3212-432a-4d43-b271-f2269be1d8ec_1304x822.png)

Figure 5: An illustration of the Mixture-of-Experts (MoE) module in DeepSeek V3/R1 (right) compared to an LLM with a standard FeedForward block (left).

The FeedForward block inside a transformer block (shown as the dark gray block in the figure above) typically contains a large number of the model's total parameters. (Note that the transformer block, and thereby the FeedForward block, is repeated many times in an LLM; in the case of DeepSeek-V3, 61 times.)

So, replacing _a single_ FeedForward block with _multiple_ FeedForward blocks (as done in a MoE setup) substantially increases the model's total parameter count. However, the key trick is that we don't use ("activate") all experts for every token. Instead, a router selects only a small subset of experts per token. (In the interest of time, or rather article space, I'll cover the router in more detail another time.)

Because only a few experts are active at a time, MoE modules are often referred to as _sparse_, in contrast to _dense_ modules that always use the full parameter set. However, the large total number of parameters via an MoE increases the capacity of the LLM, which means it can take up more knowledge during training. The sparsity keeps inference efficient, though, as we don't use all the parameters at the same time.

For example, DeepSeek-V3 has 256 experts per MoE module and a total of 671 billion parameters. Yet during inference, only 9 experts are active at a time (1 shared expert plus 8 selected by the router). This means just 37 billion parameters are used per inference step as opposed to all 671 billion.

One notable feature of DeepSeek-V3's MoE design is the use of a shared expert. This is an expert that is always active for every token. This idea is not new and was already introduced in the [DeepSeek 2024 MoE](https://arxiv.org/abs/2401.06066) and [2022 DeepSpeedMoE paper](https://arxiv.org/abs/2201.05596)s.

[![Image 6](https://substackcdn.com/image/fetch/$s_!i4ms!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d93c441-a6d2-4257-bd80-2d3590c4001c_1039x569.png)](https://substackcdn.com/image/fetch/$s_!i4ms!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d93c441-a6d2-4257-bd80-2d3590c4001c_1039x569.png)

_Figure 6: An annotated figure from "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", https://arxiv.org/abs/2401.06066_

The benefit of having a shared expert was first noted in the [DeepSpeedMoE paper](https://arxiv.org/abs/2201.05596), where they found that it boosts overall modeling performance compared to no shared experts. This is likely because common or repeated patterns don't have to be learned by multiple individual experts, which leaves them with more room for learning more specialized patterns.

To summarize, DeepSeek-V3 is a massive 671-billion-parameter model that, at launch, outperformed other open-weight models, including the 405B Llama 3. Despite being larger, it is much more efficient at inference time thanks to its Mixture-of-Experts (MoE) architecture, which activates only a small subset of (just 37B) parameters per token.

Another key distinguishing feature is DeepSeek-V3's use of Multi-Head Latent Attention (MLA) instead of Grouped-Query Attention (GQA). Both MLA and GQA are inference-efficient alternatives to standard Multi-Head Attention (MHA), particularly when using KV caching. While MLA is more complex to implement, a study in the DeepSeek-V2 paper has shown it delivers better modeling performance than GQA.

The OLMo series of models by the non-profit Allen Institute for AI is noteworthy due to its transparency in terms of training data and code, as well as the relatively detailed technical reports.

While you probably won’t find OLMo models at the top of any benchmark or leaderboard, they are pretty clean and, more importantly, a great blueprint for developing LLMs, thanks to their transparency.

And while OLMo models are popular because of their transparency, they are not that bad either. In fact, at the time of release in January (before Llama 4, Gemma 3, and Qwen 3), [OLMo 2](https://arxiv.org/abs/2501.00656) models were sitting at the Pareto frontier of compute to performance, as shown in Figure 7 below.

[![Image 7](https://substackcdn.com/image/fetch/$s_!7DYj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb5c7281-eded-4319-9ae2-7b07478a86b2_1027x823.png)](https://substackcdn.com/image/fetch/$s_!7DYj!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbb5c7281-eded-4319-9ae2-7b07478a86b2_1027x823.png)

Figure 7: Modeling benchmark performance (higher is better) vs pre-training cost (FLOPs; lower is better) for different LLMs. This is an annotated figure from the OLMo 2 paper, https://arxiv.org/abs/2501.00656

As mentioned earlier in this article, I aim to focus only on the LLM architecture details (not training or data) to keep it at a manageable length. So, what were the interesting architectural design choices in OLMo2 ? It mainly comes down to normalizations: the placement of RMSNorm layers as well as the addition of a QK-norm, which I will discuss below.

Another thing worth mentioning is that OLMo 2 still uses traditional Multi-Head Attention (MHA) instead of MLA or GQA.

Overall, OLMo 2 largely follows the architecture of the original GPT model, similar to other contemporary LLMs. However, there are some noteworthy deviations. Let's start with the normalization layers.

Similar to Llama, Gemma, and most other LLMs, OLMo 2 switched from LayerNorm to RMSNorm.

But since RMSNorm is old hat (it's basically a simplified version of LayerNorm with fewer trainable parameters), I will skip the discussion of RMSNorm vs LayerNorm. (Curious readers can find an RMSNorm code implementation in my [GPT-2 to Llama conversion guide](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb).)

However, it's worth discussing the placement of the RMSNorm layer. The original transformer (from the "[Attention is all you need](https://arxiv.org/abs/1706.03762)" paper) placed the two normalization layers in the transformer block _after_ the attention module and the FeedForward module, respectively.

This is also known as Post-LN or Post-Norm.

GPT and most other LLMs that came after placed the normalization layers _before_ the attention and FeedForward modules, which is known as Pre-LN or Pre-Norm. A comparison between Post- and Pre-Norm is shown in the figure below.

[![Image 8](https://substackcdn.com/image/fetch/$s_!wYj9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F61a4560f-d97f-4c78-a7a3-765babb45bec_1444x789.png)](https://substackcdn.com/image/fetch/$s_!wYj9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F61a4560f-d97f-4c78-a7a3-765babb45bec_1444x789.png)

_Figure 8: A comparison of Post-Norm, Pre-Norm, and OLMo 2's flavor of Post-Norm._

In [2020, Xiong et al.](https://arxiv.org/abs/2002.04745) showed that Pre-LN results in more well-behaved gradients at initialization. Furthermore, the researchers mentioned that Pre-LN even works well without careful learning rate warm-up, which is otherwise a crucial tool for Post-LN.

Now, the reason I am mentioning that is that OLMo 2 adopted a form of Post-LN (but with RMSNorm instead of LayerNorm, so I am calling it _Post-Norm_).

In OLMo 2, instead of placing the normalization layers before the attention and FeedForward layers, they place them after, as shown in the figure above. However, notice that in contrast to the original transformer architecture, the normalization layers are still inside the residual layers (skip connections).

So, why did they move the position of the normalization layers?The reason is that it helped with training stability, as shown in the figure below.

[![Image 9](https://substackcdn.com/image/fetch/$s_!ebW0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F186190ec-2ae5-430d-b0d4-a63486e0f3fb_1289x407.png)](https://substackcdn.com/image/fetch/$s_!ebW0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F186190ec-2ae5-430d-b0d4-a63486e0f3fb_1289x407.png)

_Figure 9: A plot showing the training stability for Pre-Norm (like in GPT-2, Llama 3, and many others) versus OLMo 2's flavor of Post-Norm. This is an annotated figure from the OLMo 2 paper, https://arxiv.org/abs/2501.00656_

Unfortunately this figure shows the results of the reordering together with QK-Norm, which is a separate concept. So, it’s hard to tell how much the normalization layer reordering contributed by itself.

Since the previous section already mentioned the QK-norm, and other LLMs we discuss later, such as Gemma 2 and Gemma 3, also use QK-norm, let's briefly discuss what this is.

QK-Norm is essentially yet another RMSNorm layer. It's placed inside the Multi-Head Attention (MHA) module and applied to the queries (q) and keys (k) before applying RoPE. To illustrate this, below is an excerpt of a Grouped-Query Attention (GQA) layer I wrote for my [Qwen3 from-scratch implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3) (the QK-norm application in GQA is similar to MHA in OLMo):

```
class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups,
        head_dim=None, qk_norm=False, dtype=None
    ):
        # ...

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # ...

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        # ...
```

As mentioned earlier, together with Post-Norm, QK-Norm stabilizes the training. Note that QK-Norm was not invented by OLMo 2 but goes back to the [2023 Scaling Vision Transformers paper](https://arxiv.org/abs/2302.05442).

In short, the noteworthy OLMo 2 architecture design decisions are primarily the RMSNorm placements: RMSNorm after instead of before the attention and FeedForward modules (a flavor of Post-Norm), as well as the addition of RMSNorm for the queries and keys inside the attention mechanism (QK-Norm), which both, together, help stabilize the training loss.

Below is a figure that further compares OLMo 2 to Llama 3 side by side; as one can see, the architectures are otherwise relatively similar except for the fact that OLMo 2 still uses the traditional MHA instead of GQA. (However, the [OLMo 2 team released a 32B variant](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) 3 months later that uses GQA.)

[![Image 10](https://substackcdn.com/image/fetch/$s_!S6Y9!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fa42974-cfae-45cc-9cd9-fc1d9607d386_1329x737.png)](https://substackcdn.com/image/fetch/$s_!S6Y9!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fa42974-cfae-45cc-9cd9-fc1d9607d386_1329x737.png)

Figure 10: An architecture comparison between Llama 3 and OLMo 2.

Google's Gemma models have always been really good, and I think they have always been a bit underhyped compared to other popular models, like the Llama series.

One of the distinguishing aspects of Gemma is the rather large vocabulary size (to support multiple languages better), and the stronger focus on the 27B size (versus 8B or 70B). But note that Gemma 2 also comes in smaller sizes: 1B, 4B, and 12B.

The 27B size hits a really nice sweet spot: it's much more capable than an 8B model but not as resource-intensive as a 70B model, and it runs just fine locally on my Mac Mini.

So, what else is interesting in [Gemma 3](https://arxiv.org/abs/2503.19786)? As discussed earlier, other models like Deepseek-V3/R1 use a Mixture-of-Experts (MoE) architecture to reduce memory requirements at inference, given a fixed model size. (The MoE approach is also used by several other models we will discuss later.)

Gemma 3 uses a different "trick" to reduce computational costs, namely sliding window attention.

With sliding window attention (originally introduced in the [LongFormer paper in 2020](https://arxiv.org/abs/2004.05150) and also already used by [Gemma 2](http://arxiv.org/abs/2408.00118)), the Gemma 3 team was able to reduce the memory requirements in the KV cache by a substantial amount, as shown in the figure below.

[![Image 11](https://substackcdn.com/image/fetch/$s_!LQA4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5363ce6-0ec8-49e6-b296-9836c248e159_665x302.png)](https://substackcdn.com/image/fetch/$s_!LQA4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb5363ce6-0ec8-49e6-b296-9836c248e159_665x302.png)

_Figure 11: An annotated figure from Gemma 3 paper (https://arxiv.org/abs/2503.19786) showing the KV cache memory savings via sliding window attention._

So, what is sliding window attention? If we think of regular self-attention as a _global_ attention mechanism, since each sequence element can access every other sequence element, then we can think of sliding window attention as _local_ attention, because here we restrict the context size around the current query position. This is illustrated in the figure below.

[![Image 12](https://substackcdn.com/image/fetch/$s_!tTJ5!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff32c2d74-ec34-43ef-86bc-bcce832426b3_1600x792.png)](https://substackcdn.com/image/fetch/$s_!tTJ5!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff32c2d74-ec34-43ef-86bc-bcce832426b3_1600x792.png)

_Figure 12: A comparison between regular attention (left) and sliding window attention (right)._

Please note that sliding window attention can be used with both Multi-Head Attention and Grouped-Query Attention; Gemma 3 uses grouped-query attention.

As mentioned above, sliding window attention is also referred to as _local_ attention because the local window surrounds and moves with the current query position. In contrast, regular attention is _global_ as each token can access all other tokens.

Now, as briefly mentioned above, the Gemma 2 predecessor architecture also used sliding window attention before. The difference in Gemma 3 is that they adjusted the ratio between global (regular) and local (sliding) attention.

For instance, Gemma 2 uses a hybrid attention mechanism that combines sliding window (local) and global attention in a 1:1 ratio. Each token can attend to a 4k-token window of nearby context.

Where Gemma 2 used sliding window attention in every other layer, Gemma 3 now has a 5:1 ratio, meaning there's only 1 full attention layer for every 5 sliding windows (local) attention layers; moreover, the sliding window size was reduced from 4096 (Gemma 2) to just 1024 (Gemma 3). This shifts the model's focus towards more efficient, localized computations.

According to their ablation study, the use of sliding window attention has minimal impact on modeling performance, as shown in the figure below.

[![Image 13](https://substackcdn.com/image/fetch/$s_!YSZb!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F788a5023-1ba3-4372-89cd-4ebeff132255_1600x477.png)](https://substackcdn.com/image/fetch/$s_!YSZb!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F788a5023-1ba3-4372-89cd-4ebeff132255_1600x477.png)

Figure 13: An annotated figure from Gemma 3 paper (https://arxiv.org/abs/2503.19786) showing that sliding window attention has little to no impact on the LLM-generated output perplexity.

While sliding window attention is the most notable architecture aspect of Gemma 3, I want to also briefly go over the placement of the normalization layers as a follow-up to the previous OLMo 2 section.

A small but interesting tidbit to highlight is that Gemma 3 uses RMSNorm in both a Pre-Norm and Post-Norm setting around its grouped-query attention module.

This is similar to Gemma 2 but still worth highlighting, as it differs from (1) the Post-Norm used in the original transformer (“Attention is all you need”), (2) the Pre-Norm, which was popularized by GPT-2 and used in many other architectures afterwards, and (3) the Post-Norm flavor in OLMo 2 that we saw earlier.

[![Image 14](https://substackcdn.com/image/fetch/$s_!A1BM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F752ec4b7-2720-44f8-958b-0ca85b3a96f1_1068x855.png)](https://substackcdn.com/image/fetch/$s_!A1BM!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F752ec4b7-2720-44f8-958b-0ca85b3a96f1_1068x855.png)

_Figure 14: An architecture comparison between OLMo2 and Gemma 3; note the additional normalization layers in Gemma 3._

I think this normalization layer placement is a relatively intuitive approach as it gets the best of both worlds: Pre-Norm and Post-Norm. In my opinion, a bit of extra normalization can't hurt. In the worst case, if the extra normalization is redundant, this adds a bit of inefficiency through redundancy. In practice, since RMSNorm is relatively cheap in the grand scheme of things, this shouldn't have any noticeable impact, though.

Gemma 3 is a well-performing open-weight LLM that, in my opinion, is a bit underappreciated in the open-source circles. The most interesting part is the use of sliding window attention to improve efficiency (it will be interesting to combine it with MoE in the future).

Also, Gemma 3 has a unique normalization layer placement, placing RMSNorm layers both before and after the attention and FeedForward modules.

A few months after the Gemma 3 release, Google shared [Gemma 3n](https://developers.googleblog.com/en/introducing-gemma-3n/), which is a Gemma 3 model that has been optimized for small-device efficiency with the goal of running on phones.

One of the changes in Gemma 3n to achieve better efficiency is the so-called Per-Layer Embedding (PLE) parameters layer. The key idea here is to keep only a subset of the model's parameters in GPU memory. Token-layer specific embeddings, such as those for text, audio, and vision modalities, are then streamed from the CPU or SSD on demand.

The figure below illustrates the PLE memory savings, listing 5.44 billion parameters for a standard Gemma 3 model. This likely refers to the Gemma 3 4-billion variant.

[![Image 15](https://substackcdn.com/image/fetch/$s_!Su7d!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb05999d6-88ca-4739-8b0b-266b48da288b_662x483.png)](https://substackcdn.com/image/fetch/$s_!Su7d!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb05999d6-88ca-4739-8b0b-266b48da288b_662x483.png)

Figure 15: An annotated figure from Google's Gemma 3n blog (https://developers.googleblog.com/en/introducing-gemma-3n/) illustrating the PLE memory savings.

The 5.44 vs. 4 billion parameter discrepancy is because Google has an interesting way of reporting parameter counts in LLMs. They often exclude embedding parameters to make the model appear smaller, except in cases like this, where it is convenient to include them to make the model appear larger. This is not unique to Google, as this approach has become a common practice across the field.

Another interesting trick is the[MatFormer](https://arxiv.org/abs/2310.07707) concept (short for Matryoshka Transformer). For instance, Gemma 3n uses a single shared LLM (transformer) architecture that can be sliced into smaller, independently usable models. Each slice is trained to function on its own, so at inference time, we can run just the part you need (instead of the large model).

[Mistral Small 3.1 24B](https://mistral.ai/news/mistral-small-3-1), which was released in March shortly after Gemma 3, is noteworthy for outperforming Gemma 3 27B on several benchmarks (except for math) while being faster.

The reasons for the lower inference latency of Mistral Small 3.1 over Gemma 3 are likely due to their custom tokenizer, as well as shrinking the KV cache and layer count. Otherwise, it's a standard architecture as shown in the figure below.

[![Image 16](https://substackcdn.com/image/fetch/$s_!SdCR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5140cd93-4e54-4a3d-af74-f8d1b8ea1647_1402x757.png)](https://substackcdn.com/image/fetch/$s_!SdCR!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5140cd93-4e54-4a3d-af74-f8d1b8ea1647_1402x757.png)

Figure 16: An architecture comparison between Gemma 3 27B and Mistral 3.1 Small 24B.

Interestingly, earlier Mistral models had utilized sliding window attention, but they appear to have abandoned it in Mistral Small 3.1. So, since Mistral uses regular Grouped-Query Attention instead of Grouped-Query Attention with a sliding window as in Gemma 3, maybe there are additional inference compute savings due to being able to use more optimized code (i.e., FlashAttention). For instance, I speculate that while sliding window attention reduces memory usage, it doesn't necessarily reduce inference latency, which is what Mistral Small 3.1 is focused on.

The extensive introductory discussion on Mixture-of-Experts (MoE) earlier in this article pays off again. [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) has also adopted an MoE approach and otherwise follows a relatively standard architecture that is very similar to DeepSeek-V3, as shown in the figure below. (Llama 4 includes native multimodal support, similar to models like Gemma and Mistral. However, since this article focuses on language modeling, we only focus on the text model.)

[![Image 17](https://substackcdn.com/image/fetch/$s_!ShdO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F17518ff9-1f60-4aca-b654-034dabe20626_1600x823.png)](https://substackcdn.com/image/fetch/$s_!ShdO!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F17518ff9-1f60-4aca-b654-034dabe20626_1600x823.png)

Figure 17: An architecture comparison between DeepSeek V3 (671-billion parameters) and Llama 4 Maverick (400-billion parameters).

While the Llama 4 Maverick architecture looks very similar to DeepSeek-V3 overall, there are some interesting differences worth highlighting.

First, Llama 4 uses Grouped-Query Attention similar to its predecessors, whereas DeepSeek-V3 uses Multi-Head Latent Attention, which we discussed at the beginning of this article. Now, both DeepSeek-V3 and Llama 4 Maverick are very large architectures, with DeepSeek-V3 being approximately 68% larger in its total parameter count. However, with 37 billion active parameters, DeepSeek-V3 has more than twice as many active parameters as Llama 4 Maverick (17B).

Llama 4 Maverick uses a more classic MoE setup with fewer but larger experts (2 active experts with 8,192 hidden size each) compared to DeepSeek-V3 (9 active experts with 2,048 hidden size each). Also, DeepSeek uses MoE layers in each transformer block (except the first 3), whereas Llama 4 alternates MoE and dense modules in every other transformer block.

Given the many small differences between architectures, it is difficult to determine their exact impact on final model performance. The main takeaway, however, is that MoE architectures have seen a significant rise in popularity in 2025.

The Qwen team consistently delivers high-quality open-weight LLMs. When I helped co-advising the LLM efficiency challenge at NeurIPS 2023, I remember that the top winning solutions were all Qwen2-based.

Now, Qwen3 is another hit model series at the top of the leaderboards for their size classes. There are 7 dense models: 0.6B, 1.7B, 4B, 8B, 14B, and 32B. And there are 2 MoE models: 30B-A3B, and 235B-A22B.

(By the way, note that the missing whitespace in "Qwen3" is not a typo; I simply try to preserve the original spelling the Qwen developers chose.)

Let's discuss the dense model architecture first. As of this writing, the 0.6B model may well be the smallest current-generation open-weight model out there. And based on my personal experience, it performs really well given its small size. It has great token/sec throughput and a low memory footprint if you are planning to run it locally. But what's more, it's also easy to train locally (for educational purposes) due to its small size.

So, Qwen3 0.6B has replaced Llama 3 1B for me for most purposes. A comparison between these two architectures is shown below.

[![Image 18](https://substackcdn.com/image/fetch/$s_!pu4b!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d542035-0713-4c71-8d6d-68e0e37bf0c7_1466x925.png)](https://substackcdn.com/image/fetch/$s_!pu4b!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9d542035-0713-4c71-8d6d-68e0e37bf0c7_1466x925.png)

Figure 18: An architecture comparison between Qwen3 0.6B and Llama 3 1B; notice that Qwen3 is a deeper architecture with more layers, whereas Llama 3 is a wider architecture with more attention heads.

If you are interested in a human-readable Qwen3 implementation without external third-party LLM library dependencies, I recently implemented [Qwen3 from scratch (in pure PyTorch)](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3).

The computational performance numbers in the figure above are based on my from-scratch PyTorch implementations when run on an A100 GPU. As one can see, Qwen3 has a smaller memory footprint as it is a smaller architecture overall, but also uses smaller hidden layers and fewer attention heads. However, it uses more transformer blocks than Llama 3, which leads to a slower runtime (lower tokens/sec generation speed).

As mentioned earlier, Qwen3 also comes in two MoE flavors: 30B-A3B and 235B-A22B. Why do some architectures, like Qwen3, come as regular (dense) and MoE (sparse) variants?

As mentioned at the beginning of this article, MoE variants help reduce inference costs for large base models. Offering both dense and MoE versions gives users flexibility depending on their goals and constraints.

Dense models are typically more straightforward to fine-tune, deploy, and optimize across various hardware.

On the other hand, MoE models are optimized for scaling inference. For instance, at a fixed inference budget, they can achieve a higher overall model capacity (i.e., knowledge uptake during training due to being larger) without proportionally increasing inference costs.

By releasing both types, the Qwen3 series can support a broader range of use cases: dense models for robustness, simplicity, and fine-tuning, and MoE models for efficient serving at scale.

To round up this section, let's look at Qwen3 235B-A22B (note that the A22B stands for "22B active parameters) to DeepSeek-V3, which has almost twice as many active parameters (37B).

[![Image 19](https://substackcdn.com/image/fetch/$s_!6Cx4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4627dac1-ced7-4e8d-8de4-d9238b1c427d_1632x810.png)](https://substackcdn.com/image/fetch/$s_!6Cx4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4627dac1-ced7-4e8d-8de4-d9238b1c427d_1632x810.png)

Figure 19: An architecture comparison between DeepSeek-V3 and Qwen3 235B-A22B.

As shown in the figure above, the DeepSeek-V3 and Qwen3 235B-A22B architectures are remarkably similar. What's noteworthy, though, is that the Qwen3 model moved away from using a shared expert (earlier Qwen models, such as [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-max/) did use a shared expert).

Unfortunately, the Qwen3 team did not disclose any reason as to why they moved away from shared experts. If I had to guess, it was perhaps simply not necessary for training stability for their setup when they increased the experts from 2 (in Qwen2.5-MoE) to 8 (in Qwen3). And then they were able to save the extra compute/memory cost by using only 8 instead of 8+1 experts. (However, this doesn't explain why DeepSeek-V3 is still keeping their shared expert.)

**Update.**[Junyang Lin](https://x.com/JustinLin610/status/1947364862184853626), one of the developers of Qwen3, responded as follows:

> At that moment we did not find significant enough improvement on shared expert and we were worrying about the optimization for inference caused by shared expert. No straight answer to this question honestly.

[SmolLM3](https://huggingface.co/blog/smollm3) is perhaps not as nearly as popular as the other LLMs covered in this article, but I thought it is still an interesting model to include as it offers really good modeling performance at a relatively small and convenient 3-billion parameter model size that sits between the 1.7B and 4B Qwen3 model, as shown in the figure below.

Moreover, it also shared a lot of the training details, similar to OLMo, which is rare and always appreciated!

[![Image 20](https://substackcdn.com/image/fetch/$s_!vPTQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Febfdfd6e-6aee-4894-8a57-307a25bc2c0a_743x519.png)](https://substackcdn.com/image/fetch/$s_!vPTQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Febfdfd6e-6aee-4894-8a57-307a25bc2c0a_743x519.png)

_Figure 20: An annotated figure from the SmolLM3 announcement post, https://huggingface.co/blog/smollm3, comparing the SmolLM3 win rate to Qwen3 1.7B and 4B as well as Llama 3 3B and Gemma 3 4B._

As shown in the architecture comparison figure below, the SmolLM3 architecture looks fairly standard. The perhaps most interesting aspect is its use of NoPE (No Positional Embeddings), though.

[![Image 21](https://substackcdn.com/image/fetch/$s_!oGOS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff761a811-e394-4ea1-bb80-fbed44f48d89_1447x770.png)](https://substackcdn.com/image/fetch/$s_!oGOS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff761a811-e394-4ea1-bb80-fbed44f48d89_1447x770.png)

Figure 21: A side-by-side architecture comparison between Qwen3 4B and SmolLM3 3B.

NoPE is, in LLM contexts, an older idea that goes back to a 2023 paper ([The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466)) to remove explicit positional information injection (like through classic absolute positional embedding layers in early GPT architectures or nowadays RoPE).

In transformer-based LLMs, positional encoding is typically necessary because self-attention treats tokens independently of order. Absolute position embeddings solve this by adding an additional embedding layer that adds information to the token embeddings.

[![Image 22](https://substackcdn.com/image/fetch/$s_!lLgK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd79f412c-269f-4236-9cbe-81f1a5944ae1_1190x548.png)](https://substackcdn.com/image/fetch/$s_!lLgK!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd79f412c-269f-4236-9cbe-81f1a5944ae1_1190x548.png)

Figure 22: A modified figure from my Build A Large Language Model (From Scratch) book (https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167) illustrating absolute positional embeddings.

RoPE, on the other hand, solves this by rotating the query and key vectors relative to their token position.

In NoPE layers, however, no such positional signal is added at all: not fixed, not learned, not relative. Nothing.

Even though there is no positional embedding, the model still knows which tokens come before, thanks to the causal attention mask. This mask prevents each token from attending to future ones. As a result, a token at position _t_ can only see tokens at positions _≤ t_, which preserves the autoregressive ordering.

So while there is no positional information that is explicitly added, there is still an implicit sense of direction baked into the model's structure, and the LLM, in the regular gradient-descent-based training, can learn to exploit it if it finds it beneficial for the optimization objective. (Check out the NoPE paper's theorems for more information.)

So, overall, the [NoPE paper](https://arxiv.org/abs/2305.19466) not only found that no positional information injection is necessary, but it also found that NoPE has better length generalization, which means that LLM answering performance deteriorates less with increased sequence length, as shown in the figure below.

[![Image 23](https://substackcdn.com/image/fetch/$s_!I9j6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7a57872-2af0-465b-ab5f-2caecc0aac8f_1364x800.png)](https://substackcdn.com/image/fetch/$s_!I9j6!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd7a57872-2af0-465b-ab5f-2caecc0aac8f_1364x800.png)

Figure 23: An annotated figure from the NoPE paper (https://arxiv.org/abs/2305.19466) showing better length generalization with NoPE.

Note that the experiments shown above were conducted with a relatively small GPT-style model of approximately 100 million parameters and relatively small context sizes. It is unclear how well these findings generalize to larger, contemporary LLMs.

For this reason, the SmolLM3 team likely only "applied" NoPE (or rather omitted RoPE) in every 4th layer.

[Kimi K2](https://moonshotai.github.io/Kimi-K2/) recently made big waves in the AI community due to being an open-weight model with an incredibly good performance. According to benchmarks, it's on par with the best proprietary models like Google's Gemini, Anthropic's Claude, and OpenAI's ChatGPT models.

A notable aspect is its use of a variant of the relatively new[Muon](https://github.com/KellerJordan/Muon) optimizer over AdamW. As far as I know, this is the first time Muon was used over AdamW for any production model of this size ([previously](https://arxiv.org/abs/2502.16982), it has only been shown to scale up to 16B). This resulted in very nice training loss curves, which probably helped catapult this model to the top of the aforementioned benchmarks.

While people commented that the loss was exceptionally smooth (due to the lack of spikes), I think it's not exceptionally smooth (e.g., see the OLMo 2 loss curve in the figure below; also, the L2 norm of the gradient would probably be a better metric to track training stability). However, what's remarkable is how well the loss curve decays.

However, as mentioned in the introduction of this article, training methodologies are a topic for another time.

[![Image 24](https://substackcdn.com/image/fetch/$s_!_Zh8!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F78d5d4e4-7dc4-49ab-87be-a885ba78447e_759x684.png)](https://substackcdn.com/image/fetch/$s_!_Zh8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F78d5d4e4-7dc4-49ab-87be-a885ba78447e_759x684.png)

Figure 24: Annotated figures from the Kimi K2 announcement blog article (https://moonshotai.github.io/Kimi-K2/) and the OLMo 2 paper (https://arxiv.org/abs/2305.19466).

The model itself is 1 trillion parameters large, which is truly impressive.

It may be the biggest LLM of this generation as of this writing (given the constraints that Llama 4 Behemoth is not released, proprietary LLMs don't count, and Google's 1.6 trillion [Switch Transformer](https://arxiv.org/abs/2101.03961) is an encoder-decoder architecture from a different generation).

It's also coming full circle as Kimi K2 uses the DeepSeek-V3 architecture we covered at the beginning of this article except they made it larger, as shown in the figure below.

[![Image 25](https://substackcdn.com/image/fetch/$s_!B3em!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb721c5ef-057b-405b-9293-f11e161d9230_1599x816.png)](https://substackcdn.com/image/fetch/$s_!B3em!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb721c5ef-057b-405b-9293-f11e161d9230_1599x816.png)

Figure 25: An architecture comparison between DeepSeek V3 and Kimi K2.

As shown in the figure above, Kimi K2 is basically the same as DeepSeek V3, except that it uses more experts in the MoE modules and fewer heads in the Multi-head Latent Attention (MLA) module.

Kimi K2 is not coming out of nowhere. The earlier Kimi 1.5 model discussed in the [Kimi k1.5: Scaling Reinforcement Learning with LLMs paper](https://arxiv.org/abs/2501.12599), was impressive as well. However, it had the bad luck that the DeepSeek R1 model paper was published on exactly the same date on January 22nd. Moreover, as far as I know, the Kimi 1.5 weights were never publicly shared.

So, most likely the Kimi K2 team took these lessons to heart and shared Kimi K2 as an open-weight model, before DeepSeek R2 was released. As of this writing, Kimi K2 is the most impressive open-weight model.

OpenAI’s [released](https://openai.com/index/introducing-gpt-oss/) gpt-oss-120b and gpt-oss-20b, their first open-weight models since GPT-2 in 2019, about one week after I wrote this article. Since OpenAI’s open-weight models have been so widely anticipated, I updated this article to include them. I will keep this section brief, but I have written another, much more detailed article dedicated to the gpt-oss models here:

Before summarizing the interesting tidbits, let's start with an overview of the two models, gpt-oss-20b and gpt-oss-120b, as shown in Figure 26 below.

[![Image 26](https://substackcdn.com/image/fetch/$s_!1PO4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2d438dde-c0f6-4e99-a0a8-b41e11366a27_1631x788.png)](https://substackcdn.com/image/fetch/$s_!1PO4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2d438dde-c0f6-4e99-a0a8-b41e11366a27_1631x788.png)

Figure 26: Architecture overview of the two gpt-oss models.

Looking at Figure 26, the architecture contains all the familiar components we have seen in other architectures discussed previously. For instance, Figure 27 puts the smaller gpt-oss architecture next to Qwen3 30B-A3B, which is also an MoE model with a similar number of active parameters (gpt-oss has 3.6B active parameters, and Qwen3 30B-A3B has 3.3B).

[![Image 27](https://substackcdn.com/image/fetch/$s_!eSQG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc79215d3-891e-431f-af20-8e2a3fa4be10_1552x799.png)](https://substackcdn.com/image/fetch/$s_!eSQG!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc79215d3-891e-431f-af20-8e2a3fa4be10_1552x799.png)

Figure 27: Architecture comparison between gpt-oss and Qwen3

One aspect not shown in Figure 27 is that gpt-oss uses sliding window attention (similar to Gemma 3, but in every other layer instead of using a 5:1 ratio).

Figure 27 shows that gpt-oss and Qwen3 use similar components. But if we look at the two models closely, we see that Qwen3 is a much deeper architecture with its 48 transformer blocks instead of 24.

On the other hand, gpt-oss is a much wider architecture:

*   An embedding dimension of 2880 instead of 2048

*   An intermediate expert (feed forward) projection dimension of also 2880 instead of 768

It's also worth noting that gpt-oss uses twice as many attention heads, but this doesn't directly increase the model's width. The width is determined by the embedding dimension.

Does one approach offer advantages over the other given a fixed number of parameters? As a rule of thumb, deeper models have more flexibility but can be harder to train due to instability issues, due to exploding and vanishing gradients (which RMSNorm and shortcut connections aim to mitigate).

Wider architectures have the advantage of being faster during inference (with a higher tokens/second throughput) due to better parallelization at a higher memory cost.

When it comes to modeling performance, there's unfortunately no good apples-to-apples comparison I am aware of (where parameter size and datasets are kept constant) except for an ablation study in the [Gemma 2 paper (Table 9)](https://arxiv.org/abs/2408.00118), which found that for a 9B parameter architecture, a wider setup is slightly better than a deeper setup. Across 4 benchmarks, the wider model achieved a 52.0 average score, and the deeper model achieved a 50.8 average score.

As shown in Figure 27 above, it's also noteworthy that gpt-oss has a surprisingly small number of experts (32 instead of 128), and only uses 4 instead of 8 active experts per token. However, each expert is much larger than the experts in Qwen3.

This is interesting because the recent trends and developments point towards more, smaller models as being beneficial. This change, at a constant total parameter size, is nicely illustrated in Figure 28 below from the DeepSeekMoE paper.

[![Image 28](https://substackcdn.com/image/fetch/$s_!kMwd!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F019ead67-6811-4c9d-af7b-4e6b2563ea68_1046x565.png)](https://substackcdn.com/image/fetch/$s_!kMwd!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F019ead67-6811-4c9d-af7b-4e6b2563ea68_1046x565.png)

Figure 28: An annotated figure from "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", https://arxiv.org/abs/2401.06066

Notably, unlike DeepSeek's models, neither gpt-oss nor Qwen3 uses shared experts, though.

Both gpt-oss and Qwen3 use grouped query attention. The main difference is that gpt-oss restricts the context size via sliding window attention in each second layer, as mentioned earlier.

However, there's one interesting detail that caught my eye. It seems that gpt-oss uses bias units for the attention weights, as shown in Figure 29 below.

[![Image 29](https://substackcdn.com/image/fetch/$s_!hfZq!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15ddb1b6-1540-4889-86c5-c54f5918562e_1221x352.png)](https://substackcdn.com/image/fetch/$s_!hfZq!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15ddb1b6-1540-4889-86c5-c54f5918562e_1221x352.png)

Figure 29: gpt-oss models use bias units in the attention layers. See code example here.

I haven't seen these bias units being used since the GPT-2 days, and they are commonly regarded as redundant. Indeed, I found a recent paper that shows mathematically that this is at least true for the key transformation (`k_proj`). Furthermore, the empirical results show that there is little difference between with and without bias units (see Figure 30 below).

[![Image 30](https://substackcdn.com/image/fetch/$s_!vQ5q!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fa2e161-ba61-43ab-a17b-5e6eab448699_307x151.png)](https://substackcdn.com/image/fetch/$s_!vQ5q!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fa2e161-ba61-43ab-a17b-5e6eab448699_307x151.png)

Figure 30: Table from https://arxiv.org/pdf/2302.08626 showing the average test loss when the models were trained from scratch with and without bias units.

Another detail you may have noticed is the definition of `sinks` in the code screenshot in Figure 30. In general models, attention sinks are special "always-attended" tokens placed at the start of the sequence to stabilize attention, which is especially useful in long-context scenarios. I.e., if the context gets very long, this special attended token at the beginning is still attended to, and it can learn to store some generally useful information about the entire sequence. (I think it was originally proposed in the [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) paper.)

In the gpt-oss implementation, _attention sinks_ are not actual tokens in the input sequence. Instead, they are learned per-head bias logits that are appended to the attention scores (Figure 31). The goal is the same as with the above-mentioned attention sinks, but without modifying the tokenized inputs.

[![Image 31](https://substackcdn.com/image/fetch/$s_!xuLx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F389b83da-f588-4e54-93ca-716d60d7f087_1064x754.png)](https://substackcdn.com/image/fetch/$s_!xuLx!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F389b83da-f588-4e54-93ca-716d60d7f087_1064x754.png)

Figure 31: The use of attention sinks in gpt-oss; based on the Hugging Face code [here](https://github.com/huggingface/transformers/blame/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/models/gpt_oss/modular_gpt_oss.py).

For more information about gpt-oss, and how it compares to GPT-2, please see my other gpt-oss article:

A few weeks after this article first went online, xAI released the weights of their 270B-parameter Grok 2.5 model.

I thought it would be worth including here, since Grok 2.5 was xAI's flagship production model last year. Up to this point, all models we discussed were released as open-weight models from the start. For example, gpt-oss is likely not an open-weight clone of GPT-4 but rather a custom model trained specifically for the open-source community.

With Grok 2.5, we get a rare look at a real production system, even if it is last year's.

Architecturally, Grok 2.5 looks fairly standard overall (Figure 32), but there are a few noteworthy details.

[![Image 32](https://substackcdn.com/image/fetch/$s_!3Zgv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F59c43b82-8214-4def-996f-f22a45fa085c_1683x919.png)](https://substackcdn.com/image/fetch/$s_!3Zgv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F59c43b82-8214-4def-996f-f22a45fa085c_1683x919.png)

Figure 32: Grok 2.5 next to a Qwen3 model of comparable size

For instance, Grok 2.5 uses a small number of large experts (eight), which reflects an older trend. As discussed earlier, more recent designs such as those in the DeepSeekMoE paper favor a larger number of smaller experts (this is also present in Qwen3).

Another interesting choice is the use of what amounts to a shared expert. The additional SwiGLU module shown on the left in Figure 32 functions as an always-on, shared expert. It is not identical to the classic shared-expert design since its intermediate dimension is doubled, but the idea is the same. (I still find it interesting that Qwen3 omitted shared experts, and it will be interesting to see if that changes with Qwen4 and later models.)

[GLM-4.5](https://arxiv.org/abs/2508.06471) is another major release this year.

It is an instruction/reasoning hybrid similar to Qwen3, but even better optimized for function calling and agent-style contexts.

[![Image 33](https://substackcdn.com/image/fetch/$s_!qmQC!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcf489564-f69c-4aa7-b335-0f7d172205c4_4349x3076.jpeg)](https://substackcdn.com/image/fetch/$s_!qmQC!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcf489564-f69c-4aa7-b335-0f7d172205c4_4349x3076.jpeg)

Figure 33: GLM-4.5 benchmark from the official GitHub repository at https://github.com/zai-org/GLM-4.5

As shown in Figure 34, GLM-4.5 comes in two variants. The flagship 355-billion-parameter model outperforms Claude 4 Opus on average across 12 benchmarks and trails only slightly behind OpenAI’s o3 and xAI’s Grok 4. There is also GLM-4.5-Air, a more compact 106-billion-parameter version that delivers performance only marginally below the 355-billion model.

Figure 35 compares the 355-billion architecture to Qwen3.

[![Image 34](https://substackcdn.com/image/fetch/$s_!ayu0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac10a372-7846-4baf-b3db-bc157fa9c7aa_1494x822.png)](https://substackcdn.com/image/fetch/$s_!ayu0!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac10a372-7846-4baf-b3db-bc157fa9c7aa_1494x822.png)

Figure 34: GLM-4.5 next to a similarly-sized Qwen3 model.

The designs are largely similar, but GLM-4.5 adopts a structural choice first introduced by DeepSeek V3: 3 dense layers precede the Mixture-of-Experts (MoE) blocks. Why? Starting with several dense layers improves convergence stability and overall performance in large MoE systems. If MoE routing is introduced immediately, the instability of sparse expert selection can interfere with early syntactic and semantic feature extraction. So, one might say that by keeping the initial layers dense ensures the model forms stable low-level representations before routing decisions begin to shape higher-level processing.

Also, GLM-4.5 uses a shared expert similar to DeepSeek-V3 (and unlike Qwen3).

(Interestingly, GLM-4.5 also retains the attention bias mechanism used in GPT-2 and gpt-oss.)

On 11 September 2025, the Qwen3 team released Qwen3 Next 80B-A3B (Figure 35), available in both Instruct and Thinking variants. While its design builds on the previously discussed Qwen3 architecture, I included it here as a separate entry to keep the figure numbering consistent and to draw attention to some of its design changes.

The new Qwen3 Next architecture stands out because, despite being 3× smaller than the previous 235B-A22B model (Figure 35), it introduces four times as many experts and even adds a shared expert. Both of these design choices (a high expert count and the inclusion of a shared expert) were future directions I had highlighted prior to this release, particularly in the video version of the article that I linked at the top.

[![Image 35](https://substackcdn.com/image/fetch/$s_!iUTU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce674219-3fb4-4c76-b397-408f477ce827_4111x2244.png)](https://substackcdn.com/image/fetch/$s_!iUTU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce674219-3fb4-4c76-b397-408f477ce827_4111x2244.png)

Figure 35: The original Qwen3 model released in May (left) next to the Qwen3 Next model released in September (right).

The other highlight is that they replace the regular attention mechanism by a [Gated DeltaNet](https://arxiv.org/abs/2412.06464) + [Gated Attention](https://arxiv.org/abs/2505.06708) hybrid, which helps enable the native 262k token context length in terms of memory usage (the previous 235B-A22B model model supported 32k natively, and 131k with [YaRN](https://arxiv.org/abs/2309.00071) scaling.)

So how does this new attention hybrid work? Compared to grouped‑query attention (GQA), which is still standard scaled dot‑product attention (sharing K/V across query‑head groups to cut KV‑cache size and memory bandwidth as discussed earlier but whose decode cost and cache still grow with sequence length), their hybrid mechanism mixes _Gated DeltaNet_ blocks with _Gated Attention_ blocks with in a 3:1 ratio as shown in Figure 36.

We can think of the gated attention block as standard scaled-dot-product attention that can be used in GQA, but it has a few tweaks on top. The main differences between _gated attention_ and plain GQA block are:

1.   an output gate (sigmoid-controlled, usually per-channel) that scales the attention result before it is added back to the residual;

2.   zero-centered RMSNorm for QKNorm, rather than a standard RMSNorm;

3.   partial RoPE (on a subset of dimensions).

Note that these are essentially just stability changes to GQA.

The Gated DeltaNet is a more significant change. In the DeltaNet block, q, k, v and two gates (α, β) are produced by linear and lightweight convolutional layers with normalization, and the layer replaces attention with a fast‑weight _[delta rule](https://arxiv.org/abs/2412.06464)_ update.

However, the tradeoff is that DeltaNet offers less precise content‑based retrieval than full attention, which is why one gated attention layer remains.

Given that attention grows quadratically, the DeltaNet component was added to help with memory efficiency. In the "linear-time, cache-free" family, the DeltaNet block is a essentially an alternative to Mamba. Mamba keeps a state with a learned state-space filter (essentially a dynamic convolution over time). DeltaNet keeps a tiny fast-weight memory updated with α and β and reads it with q, with small convolutions only used only to help form q, k, v, α, β.

The two subsections above describe two design decisions geared towards efficiency. Since all good things come in threes, the Qwen3 added another technique on top: [Multi-Token Prediction](https://arxiv.org/abs/2404.19737) (MTP).

Multi-token prediction trains the LLM to predict several future tokens, instead of a single one, at each step. Here, at each position _t_, small extra heads (linear layers) output logits for _t+1...t+k_, and we sum cross-entropy losses for these offsets (in the [MTP](https://arxiv.org/abs/2404.19737) paper the researchers recommended _k=4_). This additional signal speeds up training, and inference may remains one token at a time. However, the extra heads can be used in speculative multi-token decoding, which is what Qwen3-Next seems to do, however, the details are still a bit sparse:

> Qwen3-Next introduces a native Multi-Token Prediction (MTP) mechanism, which not only yields an MTP module with a high acceptance rate for Speculative Decoding but also enhances the overall performance.Additionally, Qwen3-Next specifically optimizes the multi-step inference performance of MTP, further improving the acceptance rate of Speculative Decoding in real scenarios through multi-step training that maintains consistency between training and inference. [Souce: Qwen3-Next blog post](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)

**After all these years, LLM releases remain exciting, and I am curious to see what's next!**

_This magazine is a personal passion project, and your support helps keep it alive._

_If you’d like to support my work, please consider my [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D) book or its follow-up, [Build a Reasoning Model (From Scratch)](https://mng.bz/Nwr7). (I’m confident you’ll get a lot out of these; they explain how LLMs work in depth you won’t find elsewhere.)_

_Thanks for reading, and for helping support independent research!_

If you read the book and have a few minutes to spare, I’d really appreciate a [brief review](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167). It helps us authors a lot!

**Your support means a great deal! Thank you!**
