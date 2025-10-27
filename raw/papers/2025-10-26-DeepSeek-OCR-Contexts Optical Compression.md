DeepSeek-OCR: Contexts Optical Compression
Haoran Wei, Yaofeng Sun, Yukun Li
DeepSeek-AI
Abstract
We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long
contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder
and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core
engine, designed to maintain low activations under high-resolution input while achieving high
compression ratios to ensure an optimal and manageable number of vision tokens. Experiments
show that when the number of text tokens is within 10 times that of vision tokens (i.e., a
compression ratio < 10×), the model can achieve decoding (OCR) precision of 97%. Even at a
compression ratio of 20×, the OCR accuracy still remains at about 60%. This shows considerable
promise for research areas such as historical long-context compression and memory forgetting
mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value.
On OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens,
and outperforms MinerU2.0 (6000+ tokens per page on average) while utilizing fewer than
800 vision tokens. In production, DeepSeek-OCR can generate training data for LLMs/VLMs
at a scale of 200k+ pages per day (a single A100-40G). Codes and model weights are publicly
accessible at http://github.com/deepseek-ai/DeepSeek-OCR.
64 vis toks(left) 100 vis toks(left) 64 vis toks(right) 100 vis toks(right)
DeepSeek-OCR
(Gundam-M 200dpi)
dots.ocr (200dpi)
DeepSeek-OCR (Gundam)
Precision (%)
100%
90%
80%
70%
60%
50%
40%
30%
20%
10%
0%
96.5%
98.5% 97.3% 96.8% 96.8%
19.7
MinerU2.0
93.8%
91.5% 89.8%
87.1%
85.8%
17.7
83.8%
16.5
79.3%
76.3%
15.1
13.2
12.6
59.1%
11.8
11.3
10.5
10.6
9.7
8.5
7.5
6.7
20x
15x
10x
Compression (×)
5x
0x
Overall Performance (Edit Distance)
InternVL3-78B
dots.ocr
Qwen2.5-VL-72B
OCRFlux-3B
Qwen2.5-VL-7B
OLMOCR
InternVL2-76B
600­700
700­800
800­900
900­1000
1000­1100
1100­1200
Text Tokens in Per Page (Ground­truth)
1200­1300
0.1
0.2
0.3
0.4
0.5
Vison Tokens > 1500 Average per image ( Encoder Series
DeepEncoder Series
QwenEncoder Series
InternVLEncoder Series
Other Encoders
7000
6000
5000
4000
3000
2000
1500
1000
800
600
500
400
Average Vision Tokens per Image
DeepSeek-OCR (Large)
DeepSeek-OCR (Base)
High Accuracy
ED < 0.25 ( better)
DeepSeek-OCR (Small)
GOT-OCR2.0
DeepSeek-OCR (Tiny)
Vision Tokens < 1000 Average per image ( More)
Fewer)
SmolDocling
300
250
200
150
100
(a) Compression on Fox benchmark
(b) Performance on Omnidocbench
Figure 1 |Figure (a) shows the compression ratio (number of text tokens in ground truth/number
of vision tokens model used) testing on Fox [21] benchmark; Figure (b) shows performance
comparisons on OmniDocBench [27]. DeepSeek-OCR can achieve state-of-the-art performance
among end-to-end models enjoying the fewest vision tokens.
Contents
1 Introduction 3
2 Related Works 4
2.1 2.2 Typical Vision Encoders in VLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
End-to-end OCR Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
3 Methodology 5
3.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2 3.3 DeepEncoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
3.2.1 3.2.2 Architecture of DeepEncoder . . . . . . . . . . . . . . . . . . . . . . . . . . 5
Multiple resolution support . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
The MoE Decoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.4 Data Engine . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.4.1 OCR 1.0 data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.4.2 OCR 2.0 data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.4.3 General vision data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.5 3.4.4 Text-only data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
Training Pipelines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.5.1 Training DeepEncoder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.5.2 Training DeepSeek-OCR . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4 Evaluation 10
4.1 4.2 Vision-text Compression Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . OCR Practical Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
12
4.3 Qualitative Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.3.1 4.3.2 4.3.3 Deep parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Multilingual recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . General vision understanding . . . . . . . . . . . . . . . . . . . . . . . . . . 12
12
16
17
5 Discussion 18
6 Conclusion 19
2
1. Introduction
Current Large Language Models (LLMs) face significant computational challenges when process-
ing long textual content due to quadratic scaling with sequence length. We explore a potential
solution: leveraging visual modality as an efficient compression medium for textual information.
A single image containing document text can represent rich information using substantially
fewer tokens than the equivalent digital text, suggesting that optical compression through vision
tokens could achieve much higher compression ratios.
This insight motivates us to reexamine vision-language models (VLMs) from an LLM-centric
perspective, focusing on how vision encoders can enhance LLMs’ efficiency in processing textual
information rather than basic VQA [12, 16, 24, 32, 41] what humans excel at. OCR tasks, as an
intermediate modality bridging vision and language, provide an ideal testbed for this vision-
text compression paradigm, as they establish a natural compression-decompression mapping
between visual and textual representations while offering quantitative evaluation metrics.
Accordingly, we present DeepSeek-OCR, a VLM designed as a preliminary proof-of-concept
for efficient vision-text compression. Our work makes three primary contributions:
First, we provide comprehensive quantitative analysis of vision-text token compression
ratios. Our method achieves 96%+ OCR decoding precision at 9-10×text compression,∼90% at
10-12×compression, and∼60% at 20×compression on Fox [21] benchmarks featuring diverse
document layouts (with actual accuracy being even higher when accounting for formatting
differences between output and ground truth), as shown in Figure 1(a). The results demonstrate
that compact language models can effectively learn to decode compressed visual representations,
suggesting that larger LLMs could readily acquire similar capabilities through appropriate
pretraining design.
Second, we introduce DeepEncoder, a novel architecture that maintains low activation mem-
ory and minimal vision tokens even with high-resolution inputs. It serially connects window
attention and global attention encoder components through a 16×convolutional compressor.
This design ensures that the window attention component processes a large number of vision
tokens, while the compressor reduces vision tokens before they enter the dense global attention
component, achieving effective memory and token compression.
Third, we develop DeepSeek-OCR based on DeepEncoder and DeepSeek3B-MoE [19, 20].
As shown in Figure 1(b), it achieves state-of-the-art performance within end-to-end models on
OmniDocBench while using the fewest vision tokens. Additionally, we equip the model with
capabilities for parsing charts, chemical formulas, simple geometric figures, and natural images
to enhance its practical utility further. In production, DeepSeek-OCR can generate 33 million
pages of data per day for LLMs or VLMs using 20 nodes (each with 8 A100-40G GPUs).
In summary, this work presents a preliminary exploration of using visual modality as an
efficient compression medium for textual information processing in LLMs. Through DeepSeek-
OCR, we demonstrate that vision-text compression can achieve significant token reduction
(7-20×) for different historical context stages, offering a promising direction for addressing
long-context challenges in large language models. Our quantitative analysis provides empirical
guidelines for VLM token allocation optimization, while the proposed DeepEncoder architecture
showcases practical feasibility with real-world deployment capabilities. Although focused on
OCR as a proof-of-concept, this paradigm opens new possibilities for rethinking how vision and
language modalities can be synergistically combined to enhance computational efficiency in
large-scale text processing and agent systems.
