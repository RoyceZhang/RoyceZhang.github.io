# Deep Research Report on Large Model Training Processes and Core Fine-Tuning Methods

This study delves deeply into the technical details and mathematical principles of the complete training pipeline for large language models, covering the full-stack technologies from pretraining data preparation to human preference alignment. Through a systematic analysis of current mainstream fine-tuning techniques — including full-parameter fine-tuning, parameter-efficient fine-tuning (PEFT), and reinforcement learning optimization methods — it provides comprehensive technical guidance and theoretical support for large model training practices. The research focuses on mathematical derivations, technical principles, and engineering implementations of core algorithms, furnishing a solid theoretical foundation to drive the development and industrial application of large model technologies.

* * *

## 1\. Research Background and Significance

### 1.1 Technological Development Trajectory

Since Google published "Attention Is All You Need" in 2017, the Transformer architecture has become the unified framework for modern large language models. Its core lies in multi-head self-attention (MHA), residual connections, and layer normalization; these innovations enable parallel training and modeling of long-range dependencies.

On this basis, large model training techniques have evolved through the following stages:

1.  Scale-driven stage: By exponentially increasing parameter counts and data scale (GPT-2/3, PaLM), models demonstrated "emergent capabilities."
2.  Paradigm formation stage: Gradually forming a four-stage training paradigm:
    *   Pre-training (Pre-training): Large-scale unsupervised learning to acquire general language understanding.
    *   Supervised Fine-Tuning (SFT): Training on labeled data to inject task instructions and formatting capabilities.
    *   Reward Modeling (RM): Constructing a reward model to learn the human preference function.
    *   Reinforcement Learning (RLHF): Aligning values through human feedback so that the model's outputs better match expectations.
3.  Efficiency optimization phase: Exploring methods such as sparse activation, mixed precision, MoE architectures, and multi-token prediction to reduce costs while maintaining capabilities.

👉 Intrinsic logic: The development trajectory of large models shows a clear trend from "can it be achieved" → "can it be scaled up" → "can it be efficient" → "can it be widely adopted."

### 1.2 Research value and challenges

**Challenges:**

*   Soaring training costs: Trillion-parameter models typically require tens of millions of GPU hours, with a single training run costing up to hundreds of millions of dollars.
*   Inference efficiency limitations: Autoregressive single-token prediction causes response latency, making it hard to meet real-time application demands.
*   Complex data governance: Requires large-scale data cleaning, deduplication, and quality assessment to prevent low-quality/harmful/copyrighted data from contaminating training.
*   The alignment challenge: human values and preferences are highly diverse, and finding a balance between safety and openness remains a research difficulty.

**Value aspects:**

1.  Theoretical value: systematically sort out the mathematical principles and optimization algorithms of large model training, clarifying the mechanisms at different stages.
2.  Technical value: compare full-parameter fine-tuning, PEFT methods such as LoRA/QLoRA, and alignment strategies like RLHF and DPO, analyzing their applicable scenarios and pros and cons.
3.  Practical value: Provide actionable solutions for data engineering, distributed training, mixed precision, and evaluation regression, offering implementation references for researchers and industry applications.

👉 This research aims to answer: How can efficient and controllable training and inference be achieved while maintaining performance? How can large models be made more aligned with human needs at the algorithmic and engineering levels? This will provide theoretical foundations and engineering guidance for the future development of large models.

* * *

## 2\. Preparation and processing techniques for pretraining data

### 2.1 Data collection and source analysis

#### 2.1.1 Data Source Composition

Pretraining data for large models mainly comes from text corpora across multiple dimensions. According to the data mix of current mainstream models, web text typically accounts for over 80% of the total data, including publicly available web content such as Common Crawl and Wikipedia; code repository data accounts for about 6.5%, mainly from open-source platforms like GitHub; academic literature accounts for about 2.5%, covering academic databases such as arXiv and PubMed; and books and documents account for about 4.5%, including various e-books and technical documents.

This diversified data combination ensures the model can acquire comprehensive language understanding and generation capabilities. Large-scale general text data is mainly used to enhance the model’s foundational language modeling ability; specialized text data is used to strengthen specific capabilities; multilingual data helps build cross-lingual semantic associations; scientific text enhances the model’s understanding of professional knowledge; and code data improves the model’s logical reasoning and structured thinking abilities.

#### 2.1.2 Data Quality Evaluation Criteria

Data quality assessment is a key step to ensure training effectiveness. The main criteria include:

*   Content completeness: ensuring the text's semantics are intact, with no truncation or missing parts;
*   Language fluency: evaluating the naturalness of the text using metrics such as perplexity;
*   Information density: filtering out redundant and low-quality content to ensure knowledge value;
*   Safety evaluation: identify and remove harmful, biased, or sensitive content.

Perplexity evaluation is usually computed by pretrained language models to calculate the probability distribution of text; its formula is:

$$
PPL(X) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right)
$$

where $X = \{x_1, x_2, ..., x_N\}$ is the input text sequence. The lower the perplexity, the higher the text quality.

### 2.2 Data cleaning and preprocessing workflow

#### 2.2.1 Basic Text Normalization

Text normalization is the first step in data preprocessing, aiming to ensure consistency of texts from different sources at the character level. It mainly includes:

*   Unicode normalization: Use NFKC (Normalization Form Canonical Composition) to ensure semantically identical characters are represented uniformly, for example normalizing full-width/half-width characters and composed characters with diacritics.
*   Character encoding unification: Convert all texts to UTF-8 encoding to resolve encoding inconsistencies across different corpora (such as web pages, e-books, databases).
*   Case handling: Generally retain the original case to preserve proper nouns and formatting features; for specific tasks (such as retrieval or matching), you may choose to normalize to lowercase.

👉 This stage is essentially about laying a "clean and consistent character foundation" for subsequent cleaning and modeling.

#### 2.2.2 Content Filtering and Safety Controls

After text normalization, it is necessary to remove invalid information and potential risky content. The main items include:

1.  **Non-text content cleaning**
    
    *   URLs and HTML tags: use regular expressions or parsing tools to remove web links, HTML tags, and script fragments.
    *   Special characters: remove non-semantic or noisy symbols, such as repeated emojis and garbled characters.
2.  **Language identification**
    
    *   Extract target-language text using tools such as fastText and langdetect.
    *   Filter out samples that are not in the target language to avoid contaminating the model's language distribution.
3.  **Safety content filtering**
    
    *   Keyword matching: identify harmful content such as violence, pornography, and discrimination based on a sensitive-words database.
    *   Classifier discrimination: Use pretrained text classification models to automatically detect risky content (such as hate speech and misinformation).

👉 This step ensures that training data is controllable in terms of compliance and safety, reducing the risk of the model producing inappropriate content.

#### 2.2.3 Deduplication techniques

Deduplication is a core step to ensure data diversity and training stability, usually divided into three levels: exact deduplication, fuzzy deduplication, and semantic deduplication:

1.  **Exact Deduplication**
    
    *   Generate a hash signature for documents and remove samples that are completely identical:
        
        $$
        Hash(d) = SHA\text{-}256(\text{content}(d))
        $$
    
2.  **Fuzzy Deduplication**
    
    *   Use MinHash and Locality-Sensitive Hashing (LSH) to detect near-duplicates:
        
        $$
        MinHash(S) = \min_{x \in S} h(x)
        $$
        
        Where $S$ is the set of n-grams of the document, and $h$ is the hash function.
    *   Suitable for detecting web passages that are relatively long and have high similarity.
3.  **Semantic deduplication**
    
    *   Use an embedding model to compute the semantic similarity of documents:
        
        $$
        Similarity(d_1, d_2) = \frac{Embed(d_1) \cdot Embed(d_2)}{\|Embed(d_1)\| \cdot \|Embed(d_2)\|}
        $$
        
    *   Use clustering algorithms to identify semantically similar samples and retain only representative documents.

👉 Layered deduplication strategy: first perform fast hash-based deduplication (efficient, coarse-grained), then use LSH and semantic embeddings for fine-grained filtering to balance efficiency and quality. The data cleaning and preprocessing pipeline is the necessary stage from "raw web data" to "high-quality training corpus." The following objectives must be achieved:

*   Normalization: ensure character-level consistency;
*   Content filtering: remove noise and potential risks;
*   Multi-layer deduplication ensures data diversity and information richness.

The resulting corpus is not only clean and compliant but can also effectively support the training objectives of large models.

### 2.3 Data scheduling and ratio optimization

#### 2.3.1 Data mixing strategy

In large-scale pretraining, different data sources play different roles in shaping a model's capabilities.

*   General data (such as web pages and encyclopedias): enhance language modeling and basic semantic abilities.
*   Code data: improve logical reasoning and structured generation capabilities.
*   Academic texts: strengthen the model's domain knowledge and precision of expression.
*   Multilingual data: helps the model establish cross-language semantic mappings.

Research and practice show that even when training models for specialized tasks, it is necessary to mix in a certain proportion of general-purpose data to maintain foundational language fluency and generalization ability.

The optimal mixing ratio is usually determined through multi-objective optimization:

$$
\max_{\{w_i\}} \sum_{i=1}^{k} w_i \cdot Score_i(M)
$$

The constraint is:

$$
\sum_{i=1}^{k} w_i = 1, \quad w_i \geq 0
$$

Among them:

*   $w_i$ : the mixing weight of data class $i$
*   $Score_i(M)$ : the model's performance score on that class of tasks

👉 Intuitive understanding: This is a typical weight-allocation problem that requires finding a balance among multiple dimensions such as "language fluency, domain expertise, and reasoning ability."

#### 2.3.2 Curriculum Learning Design

Curriculum learning (Data Curriculum) focuses on the order in which data is presented, not just on proportion allocation. Its core ideas are:

*   From simple to complex: have the model first learn data with clear structure and concise semantics, then gradually introduce more difficult samples.
*   From general to specialized: first build a language foundation using large-scale general data, then introduce code, academic, and domain-specific data for reinforcement.

A common progressive training pathway is:

**General text → Code data → Domain-specific corpora**

This staged design helps the model gradually build up combined abilities in "language + reasoning + domain expertise."

Course difficulty can be quantified using the following metrics:

$$
Difficulty(d) = \alpha \cdot PPL(d) + \beta \cdot Length(d) + \gamma \cdot Complexity(d)
$$

Where:

*   $PPL(d)$ : Perplexity, reflecting how challenging the corpus is for the model's language understanding
*   $Length(d)$ : Document length, representing the input processing burden
*   $Complexity(d)$ : Structural or grammatical complexity, measuring the difficulty of comprehension
*   $\alpha, \beta, \gamma$ : weight coefficient, tunable via the validation set

👉 Intuitive understanding: the course difficulty formula is like a "test difficulty scorecard" that, combining language complexity, length, and structural difficulty, dynamically schedules the model's learning pace.

Overall, data scheduling is not just about "how much data to feed," but also about "how to allocate" and "when to present" it.

*   Hybrid strategy: ensures balanced development of different capabilities.
*   Course design: follow the principle of progressive learning so the model can absorb knowledge more efficiently.

This dual regulation of "scale + sequence" is an important method in modern large-model training for improving performance and reducing costs.

* * *

## 3\. Core workflow for large-model training

### 3.1 Technical principles of the pretraining phase

#### 3.1.1 Autoregressive Language Modeling

The pretraining of large models mainly uses the Autoregressive Language Modeling (Autoregressive LM) task, whose objective is to train the model to predict the next token based on the historical context. The mathematical form is:

$$
\mathcal{L}_{LM}(\boldsymbol{u}) = \sum_{t=1}^{T} \log P(u_t \mid \boldsymbol{u}_{<t}; \theta)
$$

Where:

*   $\boldsymbol{u} = \{u_1, u_2, ..., u_T\}$ denotes the input sequence
*   $u_t$ is the $t$ th token in the sequence
*   $\theta$ are model parameters

Optimize by minimizing the loss function via gradient descent:

$$
\nabla_\theta \mathcal{L} = \sum_{t=1}^{T} \nabla_\theta \log P(u_t \mid \boldsymbol{u}_{<t}; \theta)
$$

In practice, the loss function is often implemented using cross-entropy loss:

$$
\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{j=1}^{V} y_{ij} \log(\hat{y}_{ij})
$$

Among them:

*   $N$ : batch size
*   $V$ : vocabulary size
*   $y_{ij}$ : the true label of the $i$ th sample (one-hot encoded)
*   $\hat{y}_{ij}$ : The probability distribution predicted by the model

👉 Intuitive understanding: It's like a "fill-in-the-blank" game—when the model sees the beginning of a sentence, it must predict the most plausible next word; with repeated training, it gradually learns the statistical patterns of language and semantic logic.

#### 3.1.2 Training optimization strategies

During the pretraining phase, the AdamW optimizer (Adam with added weight decay) is typically used, with common parameters:

*   $\beta_1 = 0.9$
*   $\beta_2 = 0.95$
*   $\epsilon = 10^{-8}$

This configuration has been repeatedly validated in large model training and can strike a balance between convergence speed and stability.

The learning rate schedule uses a "linear warm-up + cosine decay" strategy:

1.  Warm-up phase: gradually increase the learning rate to avoid large initial updates that could destabilize training.

$$
\eta_t = \eta_{\max} \cdot \frac{t}{t_{warmup}}
$$

2.  Cosine decay phase: as training progresses, gradually reduce the learning rate to help the model find better solutions during convergence.

$$
\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{1 + \cos\!\Bigl(\pi \cdot \frac{t - t_{warmup}}{t_{\max} - t_{warmup}}\Bigr)}{2}
$$

Among them:

*   $t$ : current step
*   $t_{warmup}$ : warm-up steps
*   $t_{\max}$ : total training steps

👉 Analogy: Learning rate scheduling is like an athlete's training rhythm:

*   Warm-up phase: gradually speed up first to avoid strains;
*   High-efficiency phase: go all out to learn;
*   Cool-down phase: gradually reduce intensity to avoid overfitting.

Overall, the pretraining stage is the "foundation work" of a large model:

*   Mastering language patterns through autoregressive modeling;
*   Precisely optimizing prediction probabilities using cross-entropy loss;
*   Improving training stability and efficiency through optimizers and learning rate scheduling.

This step determines the model's foundational language comprehension ability, laying a solid capability framework for subsequent fine-tuning and alignment.

### 3.2 Supervised Fine-Tuning Phase

#### 3.2.1 Mathematical Principles of Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) builds on the pretrained model by further training it on instruction-response pairs so that the model can better perform specific tasks and follow human instructions.

Given a labeled dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N
$$

where $x_i$ denotes the instruction and $y_i$ denotes the desired response. The optimization objective is to minimize the supervised loss function:

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \right]
$$

Gradient updates use the backpropagation algorithm:

$$
\nabla_\theta \mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \nabla_\theta \log P_\theta(y_t \mid x, y_{<t}) \right]
$$

👉 Intuitive understanding: If the pretrained model is like a "scholar well-versed in language," then supervised fine-tuning is like training it with exam questions (instruction-answer pairs) so it can answer in real applications instead of only "freewheeling."

#### 3.2.2 Data Construction and Quality Control

The effectiveness of SFT highly depends on the quality and diversity of the dataset. Construction methods include:

1.  **Human Annotation**
    
    *   High-quality instructions and reference answers are written by professional annotators.
    *   Advantages: High accuracy; capable of covering complex tasks.
    *   Disadvantages: High cost; limited in scale.
2.  **Model-generated + human filtering**
    
    *   Use a powerful teacher model (such as GPT-4) to generate large-scale instruction–response pairs.
    *   Then manually review and filter, retaining high-quality samples.
    *   This method is commonly used to rapidly expand datasets.
3.  **Data augmentation**
    
    *   Use automated methods to expand existing data, such as:
        *   Synonym replacement
        *   Sentence structure variation
        *   Context expansion
    *   Ensure task diversity and reduce model overfitting.

Data quality evaluation metrics:

*   Task relevance:
    
    $$
    Relevance = \frac{\text{相关样本数}}{\text{总样本数}}
    $$
    
    Measures whether the instruction and the response closely correspond to the task objective.
    
*   Language fluency:
    Use a pretrained language model to compute perplexity (PPL); a lower perplexity indicates the text is natural and fluent.
    
*   Logical consistency:
    Check whether the response maintains a reasonable logical correspondence with the instruction, such as correctness of answers and completeness of steps.
    

👉 Intuitively: if pretraining is “laying the foundation,” then supervised fine-tuning is “furnishing the house” — only high-quality furnishing materials (instruction data) can ensure the final house’s usefulness and comfort.

Overall, the goal of the supervised fine-tuning stage is to transform the model from a "language generalist" into a "task specialist." It learns the mapping between human instructions and desired responses by minimizing cross-entropy loss. Using high-quality instruction data is the core determinant of the model's final performance. Currently, a mixed human + automatic construction approach is commonly used, with strict quality control to ensure training effectiveness.

### 3.3 Reward Modeling Stage

#### 3.3.1 Bradley-Terry

Reward modeling (RM) is the key step that teaches the model to distinguish "better" from "worse" answers. A commonly used method is the Bradley-Terry model, which learns human preferences through pairwise comparisons.

Given a preference dataset:

$$
\mathcal{D} = \{(x, y_w, y_l)\}
$$

Among them:

*   $x$ denotes the input instruction
*   $y_w$ denotes the human-annotated preferred reply (winner)
*   $y_l$ denotes the human-annotated non-preferred reply (loser)

Preference probability is defined as:

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r_\theta(x, y_w))}{\exp(r_\theta(x, y_w)) + \exp(r_\theta(x, y_l))}= \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

where:

*   $r_\theta(x,y)$ is the score of the response given by the reward model
*   $\sigma$ is the sigmoid function

👉 Intuitive understanding: It's like a judge comparing two essays; the one with the higher score is considered the "better answer." The reward model learns to mimic human preferences through this kind of comparison.

#### 3.3.2 Reward Model Training

The objective of the reward model is to maximize the likelihood of the preference data, i.e., to make the model more likely to judge the answers preferred by humans as superior:

$$
\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \Big[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \Big]
$$

Its gradient update formula is:

$$
\nabla_\theta \mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \frac{\nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l)}{1 + \exp(r_\theta(x, y_w) - r_\theta(x, y_l))} \right]
$$

Training results:

*   Preferred replies ( $y_w$ ) will receive higher scores
*   Non-preferred replies ( $y_l$ ) will receive lower scores

Thus allowing the reward model to gradually learn human value judgments.

#### 3.3.3 Engineering Practice Key Points

In practical training, reward modeling requires attention to:

1.  Data scale: Compared to pretraining and SFT, reward modeling requires a smaller amount of data but demands higher quality.
2.  Annotation methods: Human preference ranking is commonly used; combining "humans + strong models" for joint annotation can also be employed to improve efficiency.
3.  Model architecture: Reward models often share most parameters with the original language model, adding only a scalar head at the output layer to predict scores.
4.  Overfitting issue: Reward models may learn surface features (such as response length), so regularization and diverse samples are needed to avoid bias.

👉 Analogy: If the SFT stage is “the teacher teaching the student problem-solving methods,” then reward modeling is “the teacher giving two homework assignments and telling the student which is better,” through which the student (model) learns the teacher’s preference criteria.

Overall, reward modeling is the crucial linking step in RLHF, with its mathematical basis in pairwise comparison via the Bradley-Terry model. The goal is to make the model’s scoring mechanism align as closely as possible with human preferences, providing the “reward signal” for subsequent reinforcement learning and determining the model’s final alignment.

### 3.4 Reinforcement Learning Optimization Phase

Reinforcement Learning Optimization (RLHF, Reinforcement Learning with Human Feedback) is the key step to further align the model with human values under the guidance of a reward model or preference signals. The core goal of this phase is: to maximize human satisfaction while maintaining language fluency.

#### 3.4.1 PPO Algorithm Principles

PPO (Proximal Policy Optimization) is the most commonly used optimization algorithm in RLHF. Its core idea is:

*   Update the policy guided by the reward model;
*   Constrain the update magnitude through "clipping" to prevent training instability or policy collapse.

Its optimization objective is:

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\Big(r_t(\theta) A_t,\; clip(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\Big) \right]
$$

Where:

*   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policies;
*   $A_t$ is the advantage function, measuring how good an action is relative to the average;
*   $\epsilon$ is the clipping parameter (typically 0.1–0.2), preventing excessively large updates.

Computation of the advantage function:

1.  Basic form:
    
    $$
    A_t = Q(s_t, a_t) - V(s_t)
    $$
    
2.  Use Generalized Advantage Estimation (GAE) to improve stability:
    
    $$
    A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
    $$
    
    Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the temporal-difference error.
    

👉 Intuitive understanding: PPO is like "reinforcement learning with a seatbelt." The model will try to explore new responses, but each time it can only deviate from the original policy within a limited range, ensuring it can improve without going off track.

#### 3.4.2 DPO Direct Preference Optimization

DPO (Direct Preference Optimization) is a more concise alignment method proposed in recent years. It skips the step of "training an independent reward model" and directly optimizes the policy using human preference data.

Basic idea:
Starting from the reward maximization problem with a KL constraint:

$$
\pi^* = \arg\max_\pi \mathbb{E}_{x \sim \rho, y \sim \pi(y|x)} [r(x, y)] - \beta \, KL\big(\pi(y|x) \,\|\, \pi_{ref}(y|x)\big)
$$

Its analytical solution is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)
$$

where $Z(x)$ is a normalization constant.

By reparameterization, the DPO loss function can be written as:

$$
\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma\!\Big(\beta \log \tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big) \right]
$$

👉 Intuitive understanding: PPO is "indirect training" — first learn a reward model, then learn the policy; while DPO is "direct training" — it no longer explicitly builds a reward model, but directly updates the policy using human preferences, making the process more efficient.

#### 3.4.3 Comparative analysis of PPO vs DPO

| Characteristics | PPO (traditional RLHF) | DPO (direct optimization) |
| --- | --- | --- |
| Training process | Need to train a reward model + optimize the policy | Directly optimize the policy based on preference data |
| Stability | Has a pruning mechanism and is relatively stable | Stability depends on the KL constraint parameter |
| Data requirements | Requires preference pairs + reward model data | Only preference over data |
| Computational cost | Higher (two-stage) | Lower (single-stage) |
| Practical applicability | More mature, widely used | Emerging approaches, gradually being adopted by industry |

Overall, the reinforcement learning optimization stage is the "last mile" of aligning large models:

*   PPO: A classic and stable method, suitable for large-scale RLHF training.
*   DPO: More efficient, gradually becoming the new trend.

Their shared goal is: to make the model produce outputs that better align with human preferences while ensuring fluency.

* * *

## 4\. Core fine-tuning techniques and methods

### 4.1 Full-parameter Fine-tuning Techniques

#### 4.1.1 Mathematical Principles of Full-parameter Fine-tuning

Full-parameter fine-tuning (Full Fine-Tuning) is the most traditional and straightforward fine-tuning method. The idea is to update all parameters of the pretrained model on the target task dataset, allowing the model to fully adapt to the new task.

Given pretrained parameters $\theta_{pre}$ and the target task dataset $\mathcal{D}_{task}$ , the optimization objective of full-parameter fine-tuning is:

$$
\theta_{fine} = \arg\min_\theta \mathcal{L}_{task}(\theta) + \lambda \, \Omega(\theta - \theta_{pre})
$$

Where:

*   $\mathcal{L}_{task}(\theta)$ : Task-specific loss function (such as classification cross-entropy, generative negative log-likelihood)
*   $\Omega(\cdot)$ : Regularization term, used to constrain the model from deviating excessively from the pretrained parameters
*   $\lambda$ : Regularization strength

Parameter updates use gradient descent with a small learning rate:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{task}(\theta_t)
$$

where the learning rate $\eta$ is usually 1–2 orders of magnitude smaller than in the pretraining phase to avoid disrupting the general knowledge acquired during pretraining.

👉 Intuitive understanding: Full-parameter fine-tuning is like “renovating an entire building.” While it ensures the structure fully adapts to new uses, it is costly and may damage the original foundational structure.

#### 4.1.2 Mathematical modeling of catastrophic forgetting

A common problem with full-parameter fine-tuning is catastrophic forgetting:
That is, performing well on the new task but forgetting the general knowledge acquired during pretraining.

A mathematical modeling approach is as follows:

Let the pretraining loss be $\mathcal{L}_{pre}(\theta)$ and the fine-tuning loss be $\mathcal{L}_{fine}(\theta)$ , then the degree of forgetting can be expressed as:

$$
\mathcal{L}_{forget}(\theta) = \mathcal{L}_{pre}(\theta) - \mathcal{L}_{pre}(\theta_{pre})
$$

That is: the extent to which performance on the original pretraining tasks declines after fine-tuning.

Regularization methods for mitigating forgetting:

1.  **L2 regularization**
    
    $$
    \Omega(\theta) = \frac{\lambda}{2} \|\theta - \theta_{pre}\|_2^2 
    $$
    
    Force the parameters not to deviate too far from their pretrained values.
    
2.  **Elastic Weight Consolidation, EWC**
    
    $$
    \Omega(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{pre,i})^2
    $$
    
    Where $F_i$ is the diagonal element of the Fisher information matrix, representing the importance of the parameter.
    Intuitively: impose stronger constraints on changes to critical parameters while allowing minor parameters to adjust freely.
    

👉 Intuitive understanding: the forgetting problem in full-parameter fine-tuning is like learning a new skill that "pushes out" an old one. L2 regularization is like "gently holding onto all the old knowledge," whereas EWC is like "gripping the most crucial old knowledge," reducing forgetting.

#### 4.1.3 Engineering practical considerations

*   Advantages:
    *   The model adapts best to the target task and achieves the best results
    *   Suitable for scenarios with large amounts of data and high task independence
*   Disadvantages:
    *   Extremely high computation and storage costs (requires updating all parameters)
    *   Prone to catastrophic forgetting
    *   Difficult to share across multiple tasks, poor transferability
*   Practical application:
    *   When data is abundant and the target task is extremely important (such as medical image analysis or financial risk prediction), full-parameter fine-tuning may still be used.
    *   But in most real-world scenarios, parameter-efficient fine-tuning (PEFT) is preferred instead.

Full-parameter fine-tuning is the "complete" approach to fine-tuning large models because it is the most straightforward mathematically and yields the strongest results, while being the most costly in engineering and suffering severely from forgetting. This sets the context and motivation for the subsequent proposal of parameter-efficient fine-tuning (PEFT).

### 4.2 Parameter-Efficient Fine-Tuning (PEFT) Techniques

The goal of Parameter-Efficient Fine-Tuning (PEFT) is to keep the main pretrained model parameters frozen while updating only a small number of additional parameters, thereby significantly reducing computational and storage costs. Such methods include LoRA, QLoRA, Prompt Tuning, Prefix Tuning, and others.

#### 4.2.1 LoRA (Low-Rank Adaptation) method

The core idea of LoRA (Low-Rank Adaptation) is to approximate parameter updates using a low-rank matrix decomposition, thereby reducing the number of parameters that need to be trained.

Given a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ , LoRA introduces an update matrix $\Delta W$ :

$$
W = W_0 + \Delta W = W_0 + BA
$$

Among them:

*   $B \in \mathbb{R}^{d \times r}$
*   $A \in \mathbb{R}^{r \times k}$
*   $r \ll \min(d, k)$ (low-rank constraint)

The rank constraint of the update matrix is:

$$
rank(\Delta W) = rank(BA) \leq r
$$

Comparison of parameter counts:

*   Original parameters: $d \times k$
*   LoRA parameters: $d \times r + r \times k = r(d+k)$
*   Parameter reduction ratio: $\frac{r}{k} + \frac{r}{d}$

When $r \ll \min(d,k)$ , the amount of training parameters can be significantly reduced.

👉 Intuitively: LoRA is like "adding a small adapter plug-in to a large model" — only the adapter is trained while most parameters remain frozen, allowing fast adaptation to new tasks.

#### 4.2.2 Gradient analysis of LoRA training

LoRA training only updates $A$ and $B$ , while keeping $W_0$ unchanged.

Forward propagation:

$$
y = (W_0 + BA)x = W_0x + BAx
$$

Let $z = Ax$ , then:

$$
y = W_0x + Bz
$$

The backpropagation gradient is:

$$
\frac{\partial \mathcal{L}}{\partial A} = X^T \left(\frac{\partial \mathcal{L}}{\partial y} B^T\right)
$$

 

$$
\frac{\partial \mathcal{L}}{\partial B} = (Ax)^T \frac{\partial \mathcal{L}}{\partial y}
$$

👉 Intuitive understanding: During training, the model only needs to adjust the "low-rank updater" ( $A$ and $B$ ), while the original parameters $W_0$ act as a "frozen background", thereby greatly reducing computational cost.

#### 4.2.3 QLoRA quantized low-rank adaptation

QLoRA builds on LoRA by incorporating 4-bit quantization, enabling large-model fine-tuning to be completed on a single GPU.

Weight quantization formula:

$$
Q(w) = Round\!\left(\frac{w - \min(w)}{\max(w) - \min(w)} \times (2^4 - 1)\right)
$$

Dequantization recovery:

$$
\tilde{w} = \frac{Q(w)}{2^4 - 1} \times (\max(w) - \min(w)) + \min(w)
$$

QLoRA uses a twofold quantization strategy: it quantizes not only the weights but also the quantization constants themselves, further saving memory.

👉 Engineering value: QLoRA can fine-tune 65B-parameter models on consumer-grade GPUs (e.g., 24GB VRAM), greatly lowering hardware barriers.

#### 4.2.4 Prompt-based fine-tuning methods

Prompt-based methods achieve efficient fine-tuning by inserting trainable "soft prompts" into the input or within model layers.

1.  Prompt Tuning adds a learnable prompt vector $P \in \mathbb{R}^{m \times d}$ of length $m$ to the input sequence $X \in \mathbb{R}^{n \times d}$ :
    
    $$
    \tilde{X} = Concat(P, X) \in \mathbb{R}^{(m+n) \times d}
    $$
    
    Training objective:
    
    $$
    P^* = \arg\min_P \mathcal{L}_{task}(f(Concat(P, X)))
    $$
    
    👉 Only the prompt vectors $P$ are updated; the main model parameters are completely frozen.
    
2.  Prefix Tuning adds trainable "prefix key-value pairs" to the attention mechanism of each Transformer layer:
    
    $$
    h_l = Attention(Concat(P_l^K, K_l), \, Concat(P_l^V, V_l), \, Q_l)
    $$
    
    Where $P_l^K, P_l^V$ is the learnable prefix parameter of layer $l$ .
    

👉 Intuitively: Prompt Tuning is like writing a "cheat sheet" for the model—by injecting extra prompts, it becomes easier for the model to adapt to tasks without modifying the main model.

Overall, PEFT techniques turn large-model fine-tuning from a "complete overhaul" into "lightweight customization."

*   LoRA: Low-Rank Adaptation, reduces training parameters;
*   QLoRA: combines quantization, making it feasible on consumer-grade hardware;
*   Prompt/Prefix Tuning: Quickly adapt to tasks using soft prompts or prefixes.

👉 In practical engineering, LoRA and QLoRA have become mainstream methods, widely used for fine-tuning and deploying open-source large models (such as LLaMA, Mistral).

### 4.3 Comparison of Reinforcement Learning Fine-tuning Methods

#### 4.3.1 Mathematical Details of the PPO Algorithm

In RLHF, PPO (Proximal Policy Optimization) is the most common optimization method. Its total loss function combines three parts: policy loss, value function loss, and entropy regularization:

$$
\mathcal{L}_{total} = \mathcal{L}^{CLIP} - c_1 \mathcal{L}^{VF} + c_2 S[\pi_\theta]
$$

Where:

*   Clipped policy loss $\mathcal{L}^{CLIP}$ : constrains the magnitude of policy updates to ensure stability;
*   Value function loss:
    
    $$
    \mathcal{L}^{VF} = \mathbb{E}_t \Big[ (V_\theta(s_t) - \hat{R}_t)^2 \Big]
    $$
    
    where $V_\theta(s_t)$ is the value function estimate and $\hat{R}_t$ is the target return;
*   Policy entropy:
    
    $$
    S[\pi_\theta] = \mathbb{E}_t \Big[ \pi_\theta(a_t|s_t) \log \pi_\theta(a_t|s_t) \Big]
    $$
    
    Entropy regularization encourages policy diversity and prevents the model from collapsing into a single mode.

Coefficient $c_1, c_2$ is used to balance the importance of the three types of loss.

👉 Intuitive understanding: PPO is like a "cautious student":

*   It is not allowed to deviate too far from the original answer at once (clipping constraint);
*   It checks the gap between its answer and the teacher's standard answer (value function loss);
*   It maintains a degree of creativity, avoiding being monotonous (entropy regularization).

#### 4.3.2 Mathematical relationship between DPO and PPO

DPO (Direct Preference Optimization) can be seen as a simplified version of PPO under certain assumptions.

In the PPO framework, the optimization objective is:

$$
\mathcal{J}_{PPO} = \mathbb{E}_{x,y} \big[ r(x, y) \big] - \beta \, KL(\pi \,\|\, \pi_{ref})
$$

where $r(x,y)$ is the score output by the reward model, and $\pi_{ref}$ is the reference policy.

In DPO, the reward function is implicitly defined as the logarithm of the policy ratio:

$$
r_{DPO}(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

This indicates that DPO essentially assumes a logarithmic relationship between the reward function and the policy ratio, thereby eliminating the step of training a separate reward model.

👉 Intuitive understanding:

*   PPO: It’s like “first hiring a referee to give scores (a reward model), then improving the policy based on those scores.”
*   DPO: Equivalent to "directly comparing the answer with the reference answer without needing a judge," more efficient but relies on stronger assumptions.

#### 4.3.3 Engineering Application Comparison

| Characteristics | PPO (traditional RLHF) | DPO (Direct Preference Optimization) |
| --- | --- | --- |
| Is a reward model needed? | ✅ Yes | ❌ No |
| Stability | ✅ Mature and stable, suitable for industrial use | ⚠️ Relatively new, depends on hyperparameter tuning |
| Data requirements | Preference data + reward model training | Only preference data |
| Computational cost | High (two-stage) | Low (single-stage) |
| Practical applicability | Widely applied (ChatGPT) | Increasingly popular (LLaMA family) |

#### 4.3.3 Summary

*   PPO: More robust, suitable for large-scale RLHF, and more mature and reliable for industry;
*   DPO: More efficient, eliminates the need for a reward model, but depends on assumptions and is still rapidly evolving.

👉 Analogous summary:

*   PPO is like the traditional approach of "hiring expert judges + multi-round scoring": stable but expensive;
*   DPO is like "directly comparing answers to reference answers": fast but may overlook some details.

* * *

## 5\. Technical solution selection and optimization strategies

### 5.1 Fine-tuning method selection guide

#### 5.1.1 Resource Requirements Analysis

Different fine-tuning methods differ significantly in computational complexity and memory usage:

*   Computational complexity:
    
    *   Full-parameter fine-tuning: $O(N \times P)$ , where $N$ is the amount of training data and $P$ is the number of model parameters. 👉 The overhead is enormous for large-scale models.
    *   LoRA: $O(N \times r \times (d+k))$ , where $r \ll \min(d,k)$ . 👉 Significantly reduces computation using low-rank decomposition.
    *   Prompt Tuning: $O(N \times m \times d)$ , where $m$ is the prompt length. 👉 Trains only a small number of input prompt vectors, with the lowest overhead.
*   Memory requirements:
    
    *   Full-parameter fine-tuning: requires storing the full parameters, gradients, and optimizer state, typically about 3–4 times the parameter size.
    *   LoRA: Only needs to store gradients of low-rank matrices, significantly reducing memory consumption.
    *   QLoRA: Through 4-bit quantization, further reduces memory usage to about one-quarter of the original, and can run on consumer-grade GPUs.

👉 Intuitive understanding: Full-parameter fine-tuning is like "rebuilding the entire building," which is costly; LoRA/QLoRA are "plug-in lightweight modules," low-cost modifications; Prompt Tuning is like "sticking labels or writing prompts," hardly altering the original structure.

#### 5.1.2 Performance Evaluation

There are trade-offs in performance among different methods:

*   **Task adaptability**
    
    *   Full-parameter fine-tuning: delivers near-optimal performance on almost all tasks but requires the most resources.
    *   LoRA: achieves results close to full-parameter fine-tuning on most tasks, offering excellent cost-effectiveness.
    *   Prompt-based methods: perform well on certain specific tasks (such as text classification and style control), but have limited generalization.
*   **Convergence speed**
    Convergence speed is proportional to the effective learning rate and the effective amount of data:
    
    $$
    t_{convergence} \propto \frac{1}{\eta \times |\mathcal{D}_{eff}|}
    $$
    
    *   The higher the learning rate $\eta$ , the faster the convergence, but if it is too large it may cause oscillation;
    *   The larger the effective data volume $|\mathcal{D}_{eff}|$ , the faster the model can learn stable patterns.

👉 Engineering practice: LoRA is often regarded as the "optimal compromise" — achieving a good balance between resources, performance, and convergence speed.

### 5.2 Training Stability Optimization

#### 5.2.1 Gradient Control Techniques

*   Gradient clipping prevents gradient explosion:
    
    $$
    \tilde{g} =\begin{cases}g & \text{if } ||g|| \leq \tau \\\frac{\tau}{||g||} g & \text{otherwise}\end{cases}
    $$
    
    👉 Analogy: It's like installing brakes on training to prevent parameters from updating too much at once.
    
*   Adaptive learning rate adjustment using cosine annealing schedule:
    
    $$
    \eta_t = \eta_0 \times \sqrt{\frac{1 + \cos(\pi t / T)}{2}}
    $$
    
    where $T$ is the total number of training steps. 👉 Analogy: Start fast, then gradually slow down, and finally stabilize and converge.
    

#### 5.2.2 Regularization Strategies

*   Weight decay adds a parameter penalty to the loss function:
    
    $$
    \mathcal{L}_{reg} = \mathcal{L}_{task} + \lambda \sum_i \theta_i^2
    $$
    
    👉 Prevents weights from growing without bound and improves generalization.
    
*   Dropout randomly zeros out some neurons during training:
    
    $$
    \tilde{h}_i =\begin{cases}0 & \text{with probability } p \\\frac{h_i}{1-p} & \text{otherwise}\end{cases}
    $$
    
    👉 Analogy: forcing the network to learn even when "some neurons are missing," preventing overfitting.
    
*   Label Smoothing softens the label distribution:
    
    $$
    \tilde{y}_i = (1-\alpha) y_i + \frac{\alpha}{K}
    $$
    
    where $\alpha$ is the smoothing factor and $K$ is the number of classes. 👉 Prevents the model from being overly confident and improves robustness.
    

### 5.3 Summary

*   Method selection: Full-parameter fine-tuning delivers the best performance but is costly; LoRA/QLoRA are the mainstream choices, and prompt-based methods are suitable for small-scale customization.
*   Training stability: Gradient control + regularization are the two main pillars to ensure the model does not "run away" during training.
*   Engineering value: Choosing appropriate fine-tuning methods and optimization strategies can significantly reduce resource consumption and improve model convergence stability and generalization.

* * *

## 6\. Development trends and technical outlook

### 6.1 Efficiency Optimization Trends

#### 6.1.1 Model Compression Techniques

Knowledge distillation transfers knowledge from a large model to a smaller one via a "teacher-student" network structure. Its objective function is:

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha) \mathcal{L}_{KL}(\sigma(z_t/\tau), \sigma(z_s/\tau))
$$

Where:

*   $z_s, z_t$ are the outputs of the student model and the teacher model, respectively;
*   $\tau$ is the temperature parameter, which adjusts the smoothness of the soft label distribution.

👉 Intuitive understanding: the large model is the "teacher" and the small model is the "student"; distillation is the teacher imparting knowledge to the student through answers and reasoning processes.

Model pruning (Pruning) reduces model size by removing redundant parameters:

*   Structured pruning: Remove entire channels/layers, formula:
    
    $$
    \mathcal{L}_{prune} = \mathcal{L}_{task} + \lambda \sum_i ||W_i||_1
    $$
    
*   Unstructured pruning: Remove individual parameters based on weight importance.

👉 Analogy: Structured pruning is "cutting off whole branches," while unstructured pruning is "trimming small twigs and leaves."

#### 6.1.2 Mixed-precision training

Mixed-precision training balances performance and efficiency by using different numerical precisions in different computational stages:

*   FP16 forward pass: reduces memory usage and speeds up computation;
*   FP32 gradient accumulation: ensures numerical stability;
*   Dynamic loss scaling: prevents gradient underflow.

Loss scaling formula:

$$
scaled\_loss = loss \times scale\_factor
$$

where $scale\_factor$ is dynamically adjusted to keep the gradients within an appropriate range.

👉 Analogy: It's like using "rough notes" for draft calculations (FP16), while the final results are still proofread with "fine tools" (FP32).

### 6.2 Development of multimodal fusion

#### 6.2.1 Cross-modal Alignment Techniques

Achieving vision-language alignment through contrastive learning:

$$
\mathcal{L}_{align} = -\log \frac{\exp(sim(v_i, t_i)/\tau)}{\sum_{j=1}^N \exp(sim(v_i, t_j)/\tau)}
$$

Where $v_i, t_i$ are the paired visual and text representations, and $sim(\cdot)$ is the similarity function.

👉 Analogy: Train the model to do an "image-caption matching" task, choosing the correct matching pair.

#### 6.2.2 Unified Multimodal Architecture

Different modalities are uniformly mapped to a shared representation space:

$$
h_{unified} = Transformer(Embed_{vision}(I) \oplus Embed_{text}(T))
$$

Where $\oplus$ denotes the sequence concatenation operation.

👉 Significance: The model no longer distinguishes between text and images, but processes all modalities with the same mechanisms to achieve unified reasoning.

### 6.3 Personalization and Specialization Trends

#### 6.3.1 Domain Adaptation Techniques

Knowledge transfer via distribution alignment:

$$
\mathcal{L}_{domain} = \mathcal{L}_{task} + \lambda \, MMD(f(X_s), f(X_t))
$$

Where $MMD$ (Maximum Mean Discrepancy) measures the distribution discrepancy between the source domain and the target domain.

👉 Application: model customization for specialized fields such as medicine, finance, and law.

#### 6.3.2 Personalized Modeling

Achieve rapid user customization through meta-learning:

$$
\theta_{user} = \theta_{base} - \alpha \nabla_\theta \mathcal{L}_{user}(\theta_{base})
$$

With only a small amount of user data, parameters can be quickly adjusted to accommodate individual preferences.

👉 Analogy: A pretrained model is like a "general translator," while personalized modeling is like applying a "custom filter" for a user, making responses align more closely with their style.

* * *

## Conclusions and Prospects

Large-model training is a complex technical system involving multiple dimensions such as data engineering, algorithmic optimization, and system architecture. From the fine-grained handling of pretraining data, to the mathematical principles of the four-stage training process, to the selection and application of diverse fine-tuning techniques, each link has its specific technical challenges and solutions.

In the data preparation stage, high-quality data cleaning, deduplication, and mixing strategies are the foundation for ensuring model performance. Mathematical modeling shows that optimal data ratios and curriculum design can significantly improve training efficiency and model capabilities. In terms of the training process, autoregressive modeling for pretraining, supervised optimization in SFT, preference learning in reward modeling, and policy optimization in RLHF form a complete technical chain, with each stage having its specific mathematical principles and optimization objectives.

At the fine-tuning technique level, parameter-efficient fine-tuning has become the mainstream choice. LoRA achieves a balance between efficiency and performance through low-rank decomposition, and its mathematical principles reveal the theoretical relationship between parameter reduction and performance retention. Reinforcement learning methods have evolved from the complex PPO to the simplified DPO, reflecting a trend in algorithm design from complexity toward simplicity.

Looking ahead, large-model training technology will evolve toward greater efficiency, increased specialization, and broader accessibility. Techniques such as mixed-precision training, model compression, and multimodal fusion will further lower the barriers to training. Personalization and domain adaptation techniques will meet the specific needs of specialized scenarios. These technological developments will help move large models from the laboratory into wider practical applications, truly realizing the democratization of artificial intelligence.

In summary, large-model technology is evolving along the path of "can it be trained?" → "how to do it efficiently?" → "how to make it widely accessible?". By deeply understanding the core mathematical principles and engineering implementations, researchers and engineers can make more informed choices and drive continuous innovation and real-world deployment of large models.