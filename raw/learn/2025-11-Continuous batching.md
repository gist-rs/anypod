Continuous batching

TL;DR: in this blog post, starting from attention mechanisms and KV caching, we derive continuous batching by optimizing for throughput.

If you've ever used Qwen, Claude, or any other AI chatbot, you've probably noticed something: it takes a while for the first word of the response to appear, and then words appear one-by-one on your screen with (hopefully) a regular and fast-paced frequency. That's because at the heart of it, all LLMs are just fancy next token predictors. An LLM first processes your entire prompt to produce one new token. Then it keeps adding tokens one by one, each time reading everything that came before, until it decides generation is over.

This generation process is computationally expensive: it requires passing the input through billions of parameters for each token generated. To make these models practical for real-world applications, particularly when serving many users simultaneously, researchers and engineers have developed a range of efficient inference techniques.
One of the most impactful optimizations is continuous batching, which attempts to maximize performance by processing multiple conversations in parallel and swapping them out when they are done.

To understand how continuous batching works and why it's so effective in high-load serving scenarios, we'll build up from the fundamentals of how LLMs process tokens.
