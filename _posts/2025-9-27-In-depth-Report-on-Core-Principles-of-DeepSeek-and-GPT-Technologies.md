# DeepSeek and GPT Core Technology Principles In-Depth Research Report

## —Technical Innovations and Differentiation Analysis Based on the Transformer Architecture

### Executive Summary

This study provides an in-depth analysis of the core technological principles of DeepSeek and the GPT series models, with a focus on the key innovations DeepSeek introduces atop the Transformer architecture. The research finds that DeepSeek, through innovations such as the Multi-Head Latent Attention (MLA) mechanism, a Mixture of Experts (MoE) architecture, and multi-token prediction, achieves a significant breakthrough: reducing training costs by 90% and doubling inference efficiency while maintaining performance comparable to GPT. Unlike GPT’s emphasis on scale expansion and multimodal capabilities, DeepSeek places greater emphasis on efficiency and specialization, and its open-source strategy offers a new paradigm for AI research and industrialization. In short, GPT represents a “broad and comprehensive” technological path, whereas DeepSeek demonstrates a next-generation large-model development direction that is “efficient, low-cost, and deployable.”

### Keywords

Large language model, Transformer architecture, attention mechanism, mixture-of-experts model, sparse activation

* * *

## 1\. Research background and significance

### 1.1 Technological Development Trajectory

Since Google published the paper *“Attention is All You Need”* in 2017, the Transformer architecture has become the foundational framework for modern large language models. The core advantages of the Transformer are:

*   **Parallel computing capability**: freeing models from RNN sequence dependence, greatly improving training efficiency;
*   **Long-range dependency modeling**: the attention mechanism can capture both local and global relationships simultaneously;
*   **Flexible scalability**: performance can be continuously improved by increasing the number of layers and the size of parameters.

On this basis, OpenAI's GPT series has driven the scale evolution of large models:

*   **GPT-1** demonstrated the feasibility of the "pretraining + fine-tuning" paradigm;
*   **GPT-2/3** showcased the "emergent capabilities" brought by expansion of parameter scale;
*   **GPT-4** introduced multimodality and sparse architectures, achieving performance close to that of human experts.

However, this "scale-driven" path also brought significant challenges:

1.  **Training costs skyrocketed**: trillion-parameter models often require tens of millions of GPU hours, costing hundreds of millions of dollars;
2.  **Inference efficiency declined**: autoregressive single-token prediction causes high response latency, making it difficult to meet real-time interaction demands;
3.  **Energy consumption and sustainability**: Large-scale training consumes enormous amounts of electricity and generates significant carbon emissions, raising societal concerns.

Against this backdrop, the emergence of DeepSeek has both a retrospective and pioneering significance. It does not simply continue to scale up; instead, by **the MLA mechanism reducing memory usage, MoE enabling sparse activation, and multi-token prediction accelerating inference**, it provides a new sustainable technical path for large model development. This shift marks a move from "relying solely on scale" toward a development phase that "balances architecture and efficiency."

### 1.2 Research significance

The value of this research is reflected in three aspects:

1.  **Theoretical Value** This study not only systematically reviews the evolutionary history of the Transformer architecture since its introduction, but also reveals the computational and storage bottlenecks of traditional dense architectures. By analyzing DeepSeek's innovations (MLA, MoE, multi-token prediction), this study provides a theoretical basis for **possible future optimization directions for large models** and lays an academic foundation for exploring more efficient attention mechanisms and sparse activation patterns.
2.  **Technical Value** DeepSeek's innovations are highly engineering-practical:
    *   **MLA mechanism** significantly reduces KV cache overhead, making long-sequence modeling more feasible;
    *   **MoE sparse activation** improves compute utilization, addressing energy and latency issues under traditional full-parameter activation modes;
    *   **Multi-token prediction** achieves nearly doubled inference speed without significantly sacrificing generation quality. The combination of these techniques not only optimizes the Transformer but also represents an industry shift \*\*from "pursuing scale" to "pursuing efficiency."\*\*
3.  **Application value**
    *   **Research and Education**: DeepSeek's advantages in mathematical reasoning and code generation make it usable for assisting scientific computing, automatic problem solving, and programming education, lowering the barriers to professional education and research.
    *   **Enterprise Applications**: In scenarios that are **cost-sensitive and have high concurrency**, such as customer service, data analysis, and automated report generation, DeepSeek's high efficiency and low-cost advantages are even more pronounced.
    *   **Individuals and Small Teams**: Through an open-source strategy, researchers and developers can use large models with relatively low hardware requirements, promoting the **accessibility and democratization** of AI technology.

In summary, DeepSeek is not only a supplement and challenge to the GPT technological path, but also represents a **sustainable paradigm** for the development of large models: maintaining performance while achieving significant improvements in training and inference efficiency, providing new research and application ideas for academia, industry, and the developer community.

* * *

## 2\. Fundamental Theory of the Transformer Architecture

### 2.1 "Attention is All You Need"

#### 2.1.1 Mathematical Principles of the Attention Mechanism

The core idea of the attention mechanism is: **in a sequence, each position needs to dynamically decide, based on context, whom to “attend to” and by how much.**

**Mathematical formulas:**

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

*   **Q (Query)**: Query, denotes "what information I am currently looking for"
*   **K (Key)**: Key, denotes "what labels others have provided"
*   **V (Value)**: Value, denotes "the specific content others provided"
*   𝑑 k d k ​ : Scaling factor to prevent excessively large values that could make training unstable

**Understanding the computation process step by step**

1.  **Projecting Q/K/V** Input sequence $H \in \mathbb{R}^{L \times d_{\text{model}}}$
    
    $$
    Q = HW^Q, \quad K = HW^K, \quad V = HW^V
    $$
    
2.  **Compute similarity (scoring matrix)**
    
    $$
    S = \frac{QK^\top}{\sqrt{d_k}}
    $$
    
    Each element 𝑠 𝑖 𝑗 s ij ​ represents "the attention level of the $i$ st word to the $j$ nd word."
    
3.  **Add mask and normalize**
    
    *   **Causal mask**: ensures position $i$ can only see the tokens at $\le i$ .
    *   **Softmax**: converts the scores into a probability distribution.
4.  **Weighted aggregation to obtain the output**
    
    $$
    O = \text{softmax}(S)V
    $$
    

**A minimal numerical example**

*   Let $q_2=1, k_1=1, k_2=0, v_1=10, v_2=20$
*   Score: $s_{21}=1, s_{22}=0$
*   Softmax: $\alpha_{21}\approx 0.731, \alpha_{22}\approx 0.269$
*   Output: $o_2 = 0.731 \times 10 + 0.269 \times 20 \approx 12.69$

Intuitive explanation: Position 2 pays more attention to position 1, so the output is closer to v 1 v 1 ​ 。

**Visual Analogy**

It's like being in a meeting:

*   **Q** = your current focus
*   **K** = the topic tags of each colleague's remarks
*   **V** = the specific content of the remarks. You will prioritize listening to colleagues whose points are closer to your focus (higher weight), then synthesize these inputs to form new thoughts.

**Mini schematic: matrix shapes and steps**

| Steps | Matrix | Shapes (single-head) | Description |
| --- | --- | --- | --- |
| Input | $H$ | $(L, d_{\text{model}})$ | Sequence representation |
| Projection | $Q,K,V$ | $(L, d_{\text{k}})$ | Linear mapping |
| Scoring | $QK^\top$ | $(L, L)$ | Similarity matrix |
| Normalization | $softmax$ | $(L, L)$ | Attention weights |
| Output | $OV$ | $(L, d_{\text{v}})$ | Weighted sum |

#### 2.1.2 Parallel processing of multi-head attention

In single-head attention, all information interacts within the same representation space, which may overlook multidimensional semantic relationships. The core idea of multi-head attention (Multi-Head Attention, MHA) is to project the input vectors into multiple subspaces, compute attention in each subspace (called a "head"), and then concatenate the results. This allows the model to capture dependencies from different perspectives.

**Mathematical expressions:**

$$
\begin{aligned} \mathrm{MultiHead}(Q,K,V) &= \mathrm{Concat}\bigl(\text{head}_1,\dots,\text{head}_h\bigr) W^O, \\[6pt] \text{head}_i &= \mathrm{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V) \end{aligned}
$$

*   $h$ : number of attention heads
*   𝑊 i Q , W i K , W i V W i Q ​ ,W i K ​ ,W i V ​ : Each head's independent projection matrix
*   $W^O$ : The final concatenated output mapping

**Step-by-step understanding**

1.  **Partitioning the representation space**
    
    *   Input dimension d model d model ​ was evenly divided into $h$ parts, each head having dimension $d_k = d_{\text{model}}/h$ .
    *   Each head uses a different $W^Q, W^K, W^V$ , forming an independent attention perspective.
2.  **Parallel computation of attention**
    
    *   Each head independently computes $\mathrm{Attention}(Q_i, K_i, V_i)$ .
    *   The heads do not interfere with each other and can be computed in parallel simultaneously.
3.  **Concatenation and Linear Transformation**
    
    *   Concatenate the outputs of all heads: $\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)$ .
    *   Then multiply by $W^O$ to project back to d model d model ​ dimension.

**Illustrative example**

When translating the sentence *"The cat sat on the mat"*, different attention heads may learn:

*   **Head 1**: subject–predicate relationship (cat ⟶ sat)
*   **Head 2**: verb–preposition collocation (sat ⟶ on)
*   **Head 3**: Preposition–object relationship (on ⟶ mat)
*   **Head 4**: Long-distance dependency (The ⟶ mat, connection between sentence start and end)

The multi-head mechanism allows the model to "observe the sentence from multiple perspectives at once," rather than relying on a single semantic view.

**Illustrative analogy**

Think of multi-head attention as a team discussion:

*   Each expert (attention head) focuses on a different issue (syntax, semantics, contextual logic).
*   After the discussion, gather each expert’s opinion (head outputs) and form the final comprehensive conclusion (concatenate + projection).

**Small schematic: multi-head attention process**

| Steps                 | Operations                                           | Shapes                  |
| --------------------- | ---------------------------------------------------- | ----------------------- |
| Input                 | $Q,K,V$                                              | $(L, d_{\text{model}})$ |
| Projection            | $QW_i^Q, KW_i^K, VW_i^V$                             | $(L, d_k)$              |
| Attention computation | $\text{head}_i$                                      | $(L, d_k)$              |
| Concatenation         | $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$ | $(L, h \cdot d_k)$      |
| Output mapping        | Multiply $W^O$                                       | $(L, d_{\text{model}})$ |

### 2.2 Detailed Explanation of the Complete Transformer Architecture

The Transformer was originally composed of a **Encoder + Decoder** and was used for machine translation tasks.

*   **Encoder**: maps the source sequence (such as an English sentence) to a semantic representation.
*   **Decoder**: Based on the encoder outputs, progressively generates the target sequence (e.g., a French translation).

But in modern large language models (such as GPT), typically only the **decoder portion** is retained, because the task is primarily "autoregressive generation."

#### 2.2.1 Encoder Architecture

Each encoder layer consists of **two core components**:

1.  **Multi-Head Self-Attention Layer (Multi-Head Self-Attention)**
    
    *   Function: Captures dependencies between any two positions within a sequence (without relying on recurrence or convolution).
    *   Characteristic: Input and output dimensions are the same, typically $d_{\text{model}}=512$ or higher.
    *   Advantage: Can model long-range dependencies and local relationships simultaneously.
2.  **Feed-forward neural network layer (Feed Forward Network, FFN)**
    
    *   Structure: Two fully connected layers with a ReLU or GELU activation function in between.
    *   Dimensionality changes:
        
        $$
        d_{\text{model}} \;\;\longrightarrow\;\; 4 \times d_{\text{model}} \;\;\longrightarrow\;\; d_{\text{model}}
        $$
        
    *   Function: Applies a nonlinear transformation to each position independently to increase the model's expressive power.
3.  **Residual connection + Layer normalization (Residual + LayerNorm)**
    
    *   Ensure gradient stability and avoid degradation of deep networks.
    *   Mathematical form:
        
        $$
        \text{output} = \text{LayerNorm}\bigl(\,\text{input} + \text{SubLayer}(\text{input})\,\bigr)
        $$
        

**Analogy explanation**:

*   You can think of the **self-attention layer** as an “information exchange meeting,” letting each token converse with the others.
*   The **feed-forward layer** is like “internal reflection,” where each token processes once more based on what it has absorbed.
*   The **residual connection** is akin to “preserving the original memory,” preventing information from gradually fading across multiple layers.

#### 2.2.2 The importance of positional encoding

Transformer **has no recurrent structure**, so it cannot automatically perceive sequence positions like an RNN. Therefore, it is necessary to explicitly introduce **positional encoding (Positional Encoding, PE)** to add "order information" into the word vectors.

**Formula:**

$$
PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

 

$$
PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

*   $pos$ : the position of the word in the sentence (0,1,2,...)
*   $i$ : the dimension index (even dimensions use $\sin$ , odd dimensions use $\cos$ )

**Why use sine and cosine?**

*   **Relative positional relationships are stable**: Positional differences are converted into phase differences, making it easier for the model to learn relative order.
*   **Generalizable to longer sequences**: The formula can be extended to sequences longer than those seen during training.
*   **Numerically bounded**: Outputs always lie in \[-1,1\], avoiding training instability.

**Analogy**: Imagine you are in a line, and each person (word) is wearing a "tag" that has a set of $\sin$ / $\cos$ numbers. These numbers ensure that even if the model only looks at the tags, it can know your relative position in the queue.

**Small schematic: Transformer encoder layer structure**

| Module | Input | Operations | Output |
| --- | --- | --- | --- |
| Self-attention | HHH | Multi-head attention (capturing dependencies) | H′H'H′ |
| Feedforward network | H′H'H′ | Two fully connected layers + activation | H′′H''H′′ |
| Residual & normalization | H,H′,H′′H, H', H''H,H′,H′′ | Add residual, apply LayerNorm | New HHH |

* * *

## 3\. Evolution of the GPT Architecture

The development of the GPT series can almost be seen as a microcosm of the Transformer decoder architecture evolving under the drive of large-scale corpora and compute. From the initial proof of concept to the multimodal era, each iteration has pushed natural language processing from a "rules + feature engineering" approach toward a "unified large-model paradigm."

### 3.1 Technical Developments of the GPT Model Series

#### 3.1.1 GPT-1: Proof of Concept

*   **Parameter scale**: 117 million
*   **Training data**: BooksCorpus (about 5GB of text)
*   **Core innovation**: First proposed the "**pretraining + fine-tuning**" paradigm — pretrain with large-scale unsupervised corpora using language modeling, then fine-tune on downstream tasks.
*   **Architectural features**: 12-layer Transformer decoder structure.

Significance: It demonstrated that a single architecture can be transferred to different tasks through pretraining, eliminating the need to design separate models for each task.

#### 3.1.2 GPT-2: The scaling effects emerge

*   **Parameter scale**: 1.5 billion (largest version)
*   **Training data**: WebText (about 40GB of text)
*   **Major breakthrough**: First demonstrated "strong emergent abilities"—that is, some tasks can be performed in zero- or few-shot settings without dedicated fine-tuning.
*   **Social impact**: Because the generated text was so natural, OpenAI initially did not fully open-source it for safety reasons.

GPT-2 marked the **beginning of scale-driven nonlinear capability gains**, fueling the "bigger is better" research trend.

#### 3.1.3 GPT-3: A Milestone Breakthrough

*   **Parameter scale**: 175 billion
*   **Training data**: approximately 570 GB of high-quality text
*   **Technical features**: Possesses **few-shot learning capability** and **in-context learning**, allowing the model to learn tasks directly from prompts without updating its parameters.
*   **Application Impact**: Directly gave rise to products like ChatGPT, marking that large language models have truly reached the application layer.

The arrival of GPT-3 made large models a candidate path toward general intelligence and also spurred the rise of prompt engineering.

#### 3.1.4 GPT-4: The Multimodal Era

*   **Parameter Scale**: Rumored to be about 1.8 trillion (using a **MoE sparse activation** architecture)
*   **Capability expansion**: extended from plain text to **multimodal processing such as images, code, etc.**
*   **Performance improvements**: on benchmarks like MMLU, the Bar Exam, and medical exams, performance approaches or even surpasses human expert level.

GPT-4 marks **a shift of large models from a single-language tool to a general intelligent assistant**, and opens a new phase of multimodal integration of “language + vision + knowledge + code.”

### 3.2 Core features of the GPT architecture

#### 3.2.1 Autoregressive Generation Mode

GPT uses an **autoregressive** approach to generate text, predicting words from left to right. The mathematical form is:

$$
P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^n P\bigl(x_i \mid x_1, \ldots, x_{i-1}\bigr)
$$

*   **Clear training objective**: maximize the likelihood of the next word; the loss function is explicit and easy to optimize.
*   **High generation quality**: each step strictly depends on preceding text, ensuring contextual consistency.
*   **Flexible application**: As long as a task can be converted into a "sequence generation" problem (such as translation, summarization, dialogue), GPT can be applied directly.

**Plain explanation**: GPT is like a "text continuation" machine. It first looks at the preceding words and then predicts the most reasonable next word. By repeating this process, it can generate coherent paragraphs.

#### 3.2.2 Causal masking mechanism

To ensure that "during prediction you can only see the past and not peek at the future," GPT uses a **causal masking** in the attention matrix.

Mask matrix (4×4 example):

$$
Mask matrix（4×4）: [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
$$

*   The 1st token can only see itself
*   The 2nd token can see the 1st and itself
*   The 3rd token can see the previous 2 and itself
*   ...and so on

In this way, the prediction at each position is based only on previous information, ensuring that the training process and the inference process are consistent.

**Analogy explanation**:

*   It's like writing an essay: you can only decide the next sentence based on what you've already written, and you can't peek at the "answer key" in advance.
*   This is also the key mechanism by which GPT ensures **autoregressive consistency**.

**Summary**

The evolution of the GPT series reflects a clear path of **"scale → capability → application → multimodality"**:

*   **GPT-1**: demonstrated feasibility (methodology)
*   **GPT-2**: Demonstrated scale effects (emergent capabilities)
*   **GPT-3**: Laid the foundation for applications (birth of ChatGPT)
*   **GPT-4**: Moving toward multimodal general intelligence

This process is both an accumulation of technical breakthroughs and a testament to a paradigm shift.

* * *

## 4\. DeepSeek Core Technology Innovations: In-Depth Analysis

### 4.1 Multi-Head Latent Attention (MLA): A Revolutionary Improvement to the Attention Mechanism

#### 4.1.1 Memory Challenges of Conventional MHA

In standard Multi-Head Attention (MHA), each attention head must independently store the **key (K) and value (V) matrices**.

*   Memory requirement: $O(\text{sequence length} \times \text{number of heads} \times \text{head dimension} \times 2)$
*   As sequence length L and number of heads H increase, memory consumption grows **quadratically**.

**Specific example (traditional MHA KV cache size):**

*   Model dimension: 7168
*   Number of attention heads: 56
*   Dimension per head: 128
*   Sequence length: 4096
*   KV cache size: $4096 \times 56 \times 128 \times 2 \approx 59M$ parameters

This means that during inference, the KV cache alone can occupy several gigabytes of GPU memory, becoming a bottleneck for long-context processing.

#### 4.1.2 The Innovative Design Principle of MLA

MLA draws on the idea of **low-rank decomposition**, splitting the generation of K and V into two steps: **compression + decompression**.

1.  **Step one: compressing into the latent space**
    
    $$
    c_{KV} = h_t W_{DKV}
    $$
    
    *   Input dimension: 7168
    *   Latent dimension: 1024
    *   Compression ratio: approximately 7:1
    
    Intuitive understanding: "Packing" high-dimensional information into a smaller latent representation.
    
2.  **Step 2: Decompress from the latent space**
    
    $$
    \begin{aligned} K &= c_{KV} W_{UK}, \\ V &= c_{KV} W_{UV} \end{aligned}
    $$
    
    Decompress back to the K and V matrices when needed, restoring the original attention information with linear transformations.
    

**Key advantages:**

1.  **Memory savings**: Only need to cache a 1024-dimensional latent representation, not the full 7168-dimensional K and V.
2.  **Computational efficiency**: K and V can be quickly reconstructed when needed, without having to be fully stored.
3.  **Performance retention**: Through reasonable decomposition, performance hardly decreases (<1%).

#### 4.1.3 Comparative analysis of MLA and traditional MHA

| Comparison dimensions | Traditional MHA | DeepSeek MLA |
| --- | --- | --- |
| KV cache size | 7168 dims × number of heads | 1024 dimensions (shared) |
| Memory complexity | $ O(L×H×D) $ | $ O(L×Dlatent)$ |
| Reconstruction overhead | None | Minor (linear transformation) |
| Performance impact | Benchmark | <1% decrease |

**Analogy Explanation**

You can understand the difference between traditional MHA and MLA as:

*   **Traditional MHA**: Each head keeps an entire "book" (the full K/V matrices), causing the library (VRAM) to fill up quickly.
*   **MLA**: First compress the book into a "summary card" (latent space representation), and when needed use the card to quickly reconstruct the main content of the book. This saves space while preserving content integrity.

### 4.2 Mixture of Experts (MoE): An Efficient Architecture with Sparse Activation

#### 4.2.1 The Basic Idea of MoE

In traditional dense models, every forward pass activates **all parameters**, regardless of whether the input requires them. While this yields strong expressive power, it results in huge computational and memory costs.

The core idea of MoE is:

*   **Prepare multiple "expert" subnetworks for the model**, each expert focusing on different feature patterns.
*   **Gating Network** dynamically selects a small subset of experts to participate in computation based on the input.
*   The final output is a weighted combination of the results from these experts.

The essence is **"activate on demand"**: by activating only about 5% of the parameters, you can achieve performance comparable to or even better than the full model.

**Analogy explanation**：

*   It's like the diagnostic process in a hospital: the patient doesn't need all the doctors to consult together; they just need to see the appropriate specialist. This both reduces resource waste and improves diagnostic efficiency.

#### 4.2.2 MoE architecture design of DeepSeek V3

**Overall configuration：**

*   Total number of experts: 257 (256 routing experts + 1 shared expert)
*   Experts activated each time: 9 experts (8 routing + 1 shared)
*   Expert size: about 2,048 neurons per expert
*   Activation ratio: 37B / 671B ≈ **5.5%**

**The working process of the gating network:**

1.  **Input processing** Project the input vector $x$ into the gating network to obtain a score for each expert:
    
    $$
    s = W_{\text{gate}} x
    $$
    
2.  **Expert selection (Top-K strategy)** Select the top $K$ experts with the highest scores from $s$ .
    
3.  **Weight normalization** Apply softmax to the selected scores so that they sum to 1:
    
    $$
    g = \text{softmax}(\text{Top-}K(s))
    $$
    
4.  **Expert computation and output** Feed the input into the selected experts to obtain a weighted composite result:
    
    $$
    y = \sum_i g_i \cdot \text{Expert}_i(x) \;+\; \text{Shared\_Expert}(x)
    $$
    

The shared expert ensures every input has a "common channel," preventing model failure in extreme cases. It also guarantees that every input has at least one stable path, improving robustness.

#### 4.2.3 Technical breakthroughs in load balancing

A common problem with MoE is **load imbalance**: some experts are used frequently while others remain almost idle.
Previous solutions typically relied on adding auxiliary terms to the loss function, but this could affect the performance of the main task.

**Improvements in DeepSeek V3:**

1.  **Improved routing algorithm**: Introduces a dynamic balancing mechanism during expert selection so that different experts receive more even traffic.
2.  **Expert diversity design**: Prevents all experts from learning similar features through structural differences and parameter initialization strategies.
3.  **Dynamic Weight Adjustment**: Dynamically correct routing probabilities based on experts' historical usage frequency to prevent "hot experts" from being over-allocated.

The result is: Even without auxiliary losses, DeepSeek can achieve efficient and stable load balancing.

#### 4.2.4 Comparison of MoE Working Mechanisms

| Features | Traditional dense model | MoE (using DeepSeek V3 as an example) |
| --- | --- | --- |
| Activation method | Full-parameter activation | Sparse activation (5.5%) |
| Computational overhead | $O(\text{total number of parameters})$ | $O(\text{number of activation parameters})$ |
| Number of experts | None | 257 |
| Number of experts activated each time | \- | 9 |
| Load balancing | Not applicable | Improved routing + dynamic adjustment |

**Analogy explanation**

*   **Traditional models**: It's like making everyone in the company attend every meeting, regardless of whether the topic is relevant to them, which is an extreme waste of resources.
*   **MoE**: It selectively invites only the departments most relevant to the topic, resulting in more efficient discussions and more professional outcomes.

### 4.3 Multi-token prediction: a key breakthrough in inference efficiency

#### 4.3.1 Limitations of Traditional Single-Token Prediction

In standard autoregressive language models, text generation is carried out **word by word**:

$$
t_1 \;\;\rightarrow\;\; t_2 \;\;\rightarrow\;\; t_3 \;\;\rightarrow\;\; t_4 \;\;\dots
$$

Each step predicts only the next token, then appends it to the input sequence before performing the next forward pass.

**Main issues:**

1.  **Slow inference speed**: Generating $n$ tokens requires $n$ forward passes.
2.  **Low parallelism**: Each token must wait for the previous token to be fully generated.
3.  **Computational redundancy**: Intermediate layer representations are recomputed at every step, resulting in low efficiency.

Analogy: It's like a teacher assigning homework where each question must wait until the previous one is completely finished before being handed out, so students cannot work in parallel.

#### 4.3.2 DeepSeek’s Multi-Token Prediction Innovation

DeepSeek V3 introduces **Multi-Token Prediction**, predicting multiple future tokens in a single forward pass instead of predicting just one.

Mathematical expressions:

$$
\text{input}: [t_1, t_2, \ldots, t_n] \;\;\;\longmapsto\;\;\; \text{output}: [t_{n+1}, t_{n+2}, t_{n+3}, t_{n+4}]
$$

**Implementation mechanism:**

1.  **Multi-Head Prediction** Add multiple prediction heads in the output layer, each predicting the token at different positions.
    
2.  **Speculative Decoding** Generate multiple candidate tokens in parallel, then verify them one by one; if correct, accept directly, thereby reducing inference steps. In practice this is often combined with a "small-model prediction + large-model verification" strategy to further speed up.
    
3.  **Probability Calibration** Ensure the predicted multi-step distributions are reasonable and that parallel generation does not cause contextual logic confusion.
    

#### 4.3.3 Performance Improvement (Experimental Results)

*   First-token acceptance rate: ≈ **99%**
*   Second-token acceptance rate: ≈ **85–90%**
*   Overall inference speed: increased by **1.8–2.0×**

Significantly reduced inference latency while maintaining generation quality, making it especially suitable for latency-sensitive applications (such as conversational systems and real-time interaction).

This involves a trade-off between acceptance rate and speed: generating more tokens can further accelerate inference, but the acceptance rate may decrease, requiring a balance between the two.

#### 4.3.4 Single-Token Prediction vs Multi-Token Prediction

| Features | Single-Token Prediction (Traditional) | Multi-Token Prediction (DeepSeek) |
| --- | --- | --- |
| Output per step | 1 token | Multiple tokens |
| Parallelism | None (strictly serial) | High (can be generated in parallel) |
| Inference speed | Slow | Increase by 1.8–2.0 times |
| Acceptance rate | 100% (progressive) | 85–99% (multi-step verification) |
| Application Scenarios | General Text Generation | Real-time Dialogue, Low-latency Applications |

**Analogy Explanation**

*   **Traditional single-token prediction**: Like drawing an animation frame by frame, you can only draw one frame at a time and must proceed in order.
*   **Multi-token prediction**: Like batch rendering, it can generate multiple frames at once and then proof them frame by frame, greatly improving efficiency.

* * *

## 5\. Key differences between DeepSeek and GPT technologies

### 5.1 Fundamental architectural differences

The difference between DeepSeek and GPT essentially lies in the **way computational resources are used**. GPT follows a "dense compute" path, while DeepSeek employs a "sparse activation" and "efficient storage" strategy. These two design philosophies directly affect **model inference speed, training cost, and deployment feasibility**.

#### 5.1.1 Comparison of parameter activation patterns

**GPT series (dense architecture)**

*   **Activation pattern**: every forward pass activates **all parameters**.
*   **Computational complexity**: $O(\text{total number of parameters})$.
*   **Advantages**: Strong expressive capability, suitable for general-purpose tasks.
*   **Disadvantages**: High computational and energy overhead, slow inference speed.

**DeepSeek V3 (sparse architecture)**

*   **Activation mode**: Only about **5.5% of parameters** are activated each time (MoE sparse experts).
*   **Computational complexity**: $O(\text{number of activation parameters})$.
*   **Advantages**: Computation is more efficient, and inference costs are greatly reduced.
*   **Disadvantages**: Requires complex expert routing and load-balancing mechanisms.

**Analogy explanation：**

*   GPT is like a "full-staff participation" factory that mobilizes all workers for every task, ensuring maximum productivity but consuming a great deal of energy.
*   DeepSeek is more like an "on-demand division of labor" team of experts that dispatches only the most suitable small group for each task, achieving higher efficiency but with a more complex scheduling system.

#### 5.1.2 Comparison of memory usage patterns

During large model inference, GPU memory consumption mainly comes from three parts: **parameter storage, KV cache, activation parameters**.

| Components | GPT-4o estimate | DeepSeek V3 | Efficiency improvement |
| --- | --- | --- | --- |
| Model parameters | ~1.8T | 671B | About 2.7x |
| KV cache | Full storage | MLA compression 7:1 | About 7x |
| Activation parameters | All activations | Sparse Activation 37B | Approximately 48x |

**Comparative Analysis:**

*   GPT's KV cache grows linearly with sequence length and number of attention heads, becoming a bottleneck for long-context tasks.
*   DeepSeek uses MLA to compress the KV cache dimension from 7168 to 1024, causing a sharp reduction in VRAM usage.
*   Moreover, with MoE sparse activation, the actual computation during inference is an order of magnitude lower than GPT.

**Analogy:**

*   GPT is like moving all the books (parameters + cache) intact onto the shelves, filling up the entire library.
*   DeepSeek stores "summary cards" (latent representations) and restores full information when needed, greatly saving space.

#### 5.1.3 Summary

*   **GPT approach**: Pursuing global consistency and strong expressive power → **full-parameter activation + full cache storage**.
*   **DeepSeek approach**: Pursuing efficiency and low cost → **sparse activation + latent compressed storage**.

They represent two different philosophies in the development of large models:

*   GPT: closer to "general intelligence," but expensive.
*   DeepSeek: closer to "efficient deployment," suitable for practical large-scale deployment.

### 5.2 Significant differences in training efficiency

In large model training, cost and efficiency often determine whether iteration and deployment are possible. Compared with GPT's "high-cost, compute-intensive" approach, DeepSeek achieves lower training overhead at the same scale through **accuracy optimization + efficient parallel strategies**.

#### 5.2.1 Training Cost Comparison

**Training data for DeepSeek V3:**

*   **Training duration**: approximately 2.788M H800 GPU hours
*   **Training corpus**: 14.8T tokens
*   **Cost estimate**: approximately $3–5 million

**Comparisons with models of the same class:**

*   **Llama 3.1 405B**: about 15–20M GPU hours (estimated)
*   **GPT-4**: Training is rumored to have cost over $100 million
*   **Efficiency gains**: DeepSeek saves about **80–90% of costs** compared to models of the same class

**Intuitive analogy**: If training GPT-4 is like "building an aircraft carrier"—expensive and time-consuming; DeepSeek is more like "assembling an efficient fleet," able to be deployed into action faster with the same resources.

#### 5.2.2 Training technical innovations

DeepSeek can significantly reduce training overhead, mainly relying on the following two types of technological innovations:

1.  **FP8 mixed-precision training (Mixed-Precision Training)**
    
    *   **Forward propagation**: FP8 (8-bit floating point) → Greatly reduces computation and storage overhead
    *   **Gradient computation**: BF16 (16-bit floating point) → Ensures backward propagation precision
    *   **Weight updates**: FP32 (32-bit floating point) → ensures stability of parameter updates
    
    This triple-precision strategy both ensures numerical stability and maximizes throughput efficiency at the hardware level.
    
2.  **Hybrid Parallelism with Pipelining**
    
    *   **Model parallelism**: Split model layers across different GPUs to address memory limitations of extremely large models.
    *   **Data parallelism**: Distribute training samples across multiple cards to increase throughput.
    *   **Expert Parallelism**: In the MoE architecture, experts are assigned to different GPUs to reduce the load on a single card.
    *   **Communication Optimization**: Through compression and scheduling, cross-node transmission bottlenecks are reduced.
    
    Analogy: It's like an assembly-line factory with different production lines operating simultaneously: one dedicated to assembling parts (model parallelism), one for mass production (data parallelism), another handling special tasks (expert parallelism), all coordinated through an efficient logistics system (communication optimization).
    

#### 5.2.3 Training Efficiency Comparison

| Features | GPT-4 (Dense) | DeepSeek V3 (Sparse + Optimized) |
| --- | --- | --- |
| GPU hours consumption | Tens of millions | About 2.8M |
| Cost estimate | \>>> 100 million USD | 3–5 million USD |
| Precision strategy | FP16 / BF16 | FP8 + BF16 + FP32 |
| Parallel strategy | Primarily data parallelism | Model + data + experts: triple parallelism |
| Communication optimization | standard All-Reduce | Cross-node optimization to reduce bottlenecks |

### 5.3 Specialized characteristics of performance

DeepSeek V3 and the GPT series exhibit different biases in performance. GPT emphasizes generality and multimodal capabilities, while DeepSeek has advantages in highly structured tasks such as mathematics, logic, and programming.

#### 5.3.1 DeepSeek’s Outstanding Mathematical and Reasoning Abilities

**Benchmark comparison results:**

| Test set | DeepSeek V3 | GPT-4o | Advantage Areas |
| --- | --- | --- | --- |
| MATH-500 | 90.2% | 74.6% | Mathematical Reasoning |
| MMLU-Pro | 75.9% | 73.3% | Expert Knowledge |
| HumanEval | 88.5% | 90.2% | Code generation |
| GPQA | 65.2% | 69.7% | Scientific Q&A |

**Advantages analysis:**

*   **Mathematical reasoning**: DeepSeek greatly outperforms on MATH-500, indicating it is better at tasks in algebra, geometry, probability, and the like.
*   **Logical reasoning**: Performs consistently on tasks that require multi-step chains of thought.
*   **Chinese processing**: Because it was trained on a larger-scale Chinese corpus, DeepSeek has a clear advantage in Chinese understanding and generation tasks.

**Analogy explanation:**

*   GPT is like a "well-rounded polymath" who can write stories, compose poems, and chat.
*   DeepSeek is more like a "logically meticulous" engineer or mathematician, especially skilled in computation, proofs, and code generation.

#### 5.3.2 Trade-off between generality and domain expertise

**Areas where GPT-4o excels:**

*   **Creative writing**: Better at generating natural, highly creative text.
*   **Multi-turn dialogue**: Possesses strong context retention and emotional understanding capabilities.
*   **Multimodal processing**: Supports text + image + voice input and output, enabling a wider range of application scenarios.
*   **User experience optimization**: Interaction is closer to human expression habits.

**Areas of strength for DeepSeek V3:**

*   **Mathematics and Scientific Computing**: More sensitive to formulas, derivations, and precise calculations.
*   **Code Generation and Debugging**: Approaches GPT-4o level on HumanEval, especially strong on engineering tasks.
*   **Logical Reasoning and Analysis**: Skilled at multi-step reasoning chains with good stability.
*   **Cost-Sensitive Scenarios**: Delivers near-SOTA performance under limited budgets.

#### 5.3.3 Performance Bias Comparison

| Characteristics | GPT-4o | DeepSeek V3 |
| --- | --- | --- |
| Mathematical Reasoning | Moderately Weak | Strong |
| Logical multi-step reasoning | Good | Outstanding |
| Code generation | Leading | Near-leading level |
| Chinese processing | General | Clear advantages |
| Creative writing | Clear advantages | General |
| Multimodal tasks | Strong (text + image + audio) | Weak (text-dominant) |
| Cost Efficiency | Low | Efficient |

**Comparative Analysis**

*   **Task selection**: Prefer DeepSeek for tasks requiring rigorous logic and a unique answer (mathematics, code, science).
*   **Creative scenarios**: GPT-4o is more suitable for tasks requiring creativity and interactivity (writing, dialogue, multimodal applications).
*   **Hybrid deployment**: In real systems, a **Task Routing** strategy can be used to assign different tasks to the most appropriate model, balancing cost and performance.

* * *

## 6\. Technical evolution history and development trends

From the proposal of the Transformer in 2017 to DeepSeek's breakthrough in 2024, the development of large language models has followed a clear trajectory of **architecture exploration → scale expansion → efficiency optimization → democratization**. This process not only reflects the internal logic of technological iteration but also mirrors shifts in research paradigms, industry strategies, and societal acceptance.

### 6.1 Technical inheritance from Transformer to DeepSeek

#### 6.1.1 Key technical milestones

*   **2017: The foundation laid by the Transformer**
    
    *   Core contribution: proposed a pure attention mechanism architecture, completely freeing models from the constraints of recurrence and convolution.
    *   Technical breakthrough: enabled modeling of long-range dependencies and parallel computation.
    *   Impact: became the foundational universal architecture for modern LLMs.
*   **2018–2020: Rise of the GPT series**
    
    *   GPT-1: validated the "pretraining + fine-tuning" paradigm.
    *   GPT-2: demonstrated the "scaling effect," with emergent capabilities.
    *   GPT-3: language generation abilities approached human level, laying the foundation for applications.
*   **2021: the emergence of efficiency optimizations**
    
    *   Switch Transformer: Initial application of MoE sparse architecture.
    *   PaLM: Exploring the training limits of ultra-large-scale models.
    *   Technical trend: Beginning to shift from simply expanding scale to **focusing on training and inference efficiency**.
*   **2024: DeepSeek technical breakthroughs**
    
    *   MLA: Attention mechanism compression and efficient reconstruction.
    *   Auxiliary-loss-free MoE: Solves the expert load-balancing problem.
    *   Multi-token prediction: Inference speed increased to 1.8–2.0× of the original.

**Summary**: DeepSeek no longer relies on mere parameter scaling; instead, through innovative architectural design, it achieves a balance between efficiency and performance.

#### 6.1.2 The Intrinsic Logic of Technological Evolution

*   **Phase One: Architecture Exploration (2017–2018)**
    
    *   Goal: Validate the feasibility of the Transformer.
    *   Technical Focus: Attention mechanism, positional encoding, residual connections.
    *   Representative works: the original Transformer, GPT-1.
*   **Second stage: scale expansion (2019–2022)**
    
    *   Goal: improve performance by increasing parameter scale.
    *   Technical focus: data engineering, distributed parallel training, hardware stacking.
    *   Sample：GPT-2/3、PaLM、Switch Transformer。
*   **Phase 3: Efficiency Optimization (2023–Present)**
    
    *   Objective: Reduce training and inference costs while maintaining performance.
    *   Technical focus: Sparse activation (MoE), latent space compression (MLA), multi-token prediction.
    *   Representative works: DeepSeek series, Mixtral, Llama 2/3.

**Trend summary**: The evolution logic is reflected as **from "can it be achieved" → "can it be scaled" → "can it be efficient"**. The significance of DeepSeek lies in proving that the development of large models is not limited to a single path of "scaling up."

### 6.2 Strategic significance of the open-source strategy

#### 6.2.1 The role in promoting technological democratization

DeepSeek chooses to be **fully open-source**, and its open materials include:

*   **Model weights**: The full 671B parameters are available for direct download.
*   **Training code**: Reproducible training scripts and configuration files are provided.
*   **Technical report**: Detailed explanations of core principles such as MLA, MoE, and MTP.
*   **Inference framework**: Optimized deployment tools to lower the barrier to implementation.

**Impact:**

1.  **Lowering the barrier to research**: Enables researchers worldwide to directly access cutting-edge technologies.
2.  **Accelerating technological innovation**: The community can perform secondary development based on open-source成果.
3.  **Promoting ecosystem prosperity**: Establishes a complete toolchain and industry ecosystem from models to applications.

Analogy: If closed-source models are "black-box prescriptions of private labs," open-source models are "public recipes" that allow researchers worldwide to innovate on them.

#### 6.2.2 Competitive and cooperative relationship with closed-source models

**Advantages of open-source models:**

*   High transparency and strong customizability.
*   Localized deployment, with data security and control.
*   Cost-controllable, no ongoing API call fees required.

**Advantages of closed-source models:**

*   Continuous optimization and iteration, with faster version updates.
*   Higher user experience and service quality.
*   Stronger multimodal capabilities and more complete ecosystem support.

**Future development trends:**

*   Open-source and closed-source may **coexist for the long term**, forming a complementary pattern.
*   Open-source models will continue to take an increasing share in B-side (enterprise-level) applications.
*   Technological innovation will increasingly come from open-source communities, while closed-source models will focus on commercial deployment and user experience optimization.

### 6.3 Outlook on Future Development Trends

Based on current evolution paths, several directions for the future development of large models can be predicted:

1.  **Balancing Scale and Efficiency**
    
    *   No longer solely pursuing parameter count, but exploring the optimal trade-off between performance and efficiency.
    *   Sparsification, compression, and quantization will become mainstream optimization methods.
2.  **Multimodality and cross-modal integration**
    
    *   Unified modeling of text, images, speech, and video will become a trend.
    *   More natural human-computer interaction experiences will drive industry upgrades.
3.  **Personalized and specialized models**
    
    *   There is growing demand for specialized models targeted at specific domains (medical, financial, scientific research).
    *   Through controlled fine-tuning, the model can better adapt to specialized scenarios.
4.  **Widespread accessibility and edge deployment**
    
    *   Open-source and efficient models lower the deployment threshold, enabling individuals and small to medium enterprises to use advanced AI.
    *   Inference frameworks will support deployment on mobile and edge devices.

Overall trend: The development of large models is gradually shifting from a "parameter-scale race" to a new stage that emphasizes "efficiency, accessibility, and specialization" equally.

* * *

## 7\. References and technical materials

**Core academic papers:**

1.  Vaswani, A., et al. (2017). "Attention is all you need." Advances in neural information processing systems.
    
2.  DeepSeek-AI Team. (2024). "DeepSeek-V3 Technical Report." arXiv preprint arXiv:2412.19437.
    
3.  Radford, A., et al. (2018). "Improving language understanding by generative pre-training." OpenAI Blog.
    
4.  Brown, T., et al. (2020). "Language models are few-shot learners." Advances in neural information processing systems.
    

**Technical reports and white papers:**

5.  DeepSeek Technical Documentation: [https://github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
6.  OpenAI GPT-4 Technical Report (2023)
