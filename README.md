![header](header.png)

# **Language Tokenization Strategies and Embedding Vector Conversions**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AvK7xT9Y9EJD45A6XF-TfaJPfAVZKv_Y?usp=sharing)

## **Summary**
Language models, both traditional vs modern and small vs large, always split up a collection of texts into tokens (small, manageable chunks of texts) before they are converted to a list of numerical values, namely vector. This repository is intended to document my personal journey of learning different tokenization strategies from traditional text split by white space to byte level split. In addition to tokenization strategies, this repository also covers the extent to which I understand embedding vectors, starting from the sparsity problem with traditional vectorization techniques, particularly one-hot encoding, to contextual embeddings which already become the common practice in today's NLP applications in LLMs. 

## **1 One-Hot Encoding**
One-hot encoding is a common technique in traditional NLP by converting tokens into vectors so that machine learning algorithms can model the patterns (e.g., negative sentiment characterized by the presence of a specific set of tokens associated with less desirable qualities or negative perceptions). While this tokenization technique is useful, this technique introduces the sparsity problem. One-hot encoding basically converts a token into either 1 for present and 0 for absent in a document. This eiher 1 or 0 is where the sparsity happens. Sparsity occurs when a matrix which represents documents by rows and features (tokens) by columns contains a large number of zeros. When a matrix is largely populated by zeros, computation memory will be wasted to process the absence of a token. 

<img src="img/one-hot-encoding.png" alt="one-hot encoding" width="400"/></br>
**Image 1**: A dataframe of one-hot encoded sentence. Let us assume this is a matrix of one-hot encoded sentence.

As shown, sentence, "Colorless green ideas sleep furiously.", is represented by the presence or absence of the word by 1 (present) and 0 (absent). This encoding can lead to vocabulary explosion in morphologically rich languages like Indonesian. Each variant will be treated as a separate column in matrix, increasing dimensionality and sparsity at the same time. This will be a worse problem when the dataset is a set of samples from social media where users use a wide variety of spellings. The dimensionality and sparsity can even be more problematic. Moreover, as this simple tokenization and vectorization (categorical to numerical values) on the absence or presence of a token, the encoding does not include semantic information too, making us telling token *ideas* and *concepts* are semantically similar become impossible. The limitation on semanticity of tokens does not enable meaning-related NLP tasks such as document clustering by meaning, semantic search, and keyphrase extraction theoretically less possible.


## **2 Static Dense Embeddings**
Static embeddings such as GloVe, Word2Vec, and fastText can resolve the semantic problems in one-hot encoding by instead of representing tokens with either 1 or 1, they use dense vectors, i.e., compact numerical representations which involves non-zero floating points, containing both semantic and syntactic information of the tokens. But it is worth to note that unlike one-hot encoding which directly makes sense to humans, dense embeddings do not since they encode meanings geometrically, not linguistically. 

<img src="img/glove-embeddings.png" alt="static embeddings" width="400"/></br>
**Image 2**: Each token is represented in a long vector containing numbers with floating points.

While static embeddings can solve the semantic problem in one-hot encoding (as well as frequency-based encoding such as term frequency and term frequency-inverse document frequency), static embeddings treat the same tokens the same regardless their syntactic positions. For example, the vector representing token *sleep* as verb as in sentence, "Colorless green ideas sleep furiously.", will be the same as to the same token as in "I sleep at night.". This happens because the values in static embeddings remains the same regardless the context, especially related to in what position the token *sleep* is used and what relation it bears with other tokens in the same sentence.

<img src="img/token-sleep-glove.png" alt="comparison between token sleep in two sentences with static embeddings" width="250"/></br>
**Image 3**: A comparison between token sleep in two sentences with static embeddings

To check if the same words in different uses have the same embedding vectors, we can both compare the embedding vectors of both tokens side by side. Alternatively, it is also possible to use cosine similarity to examine the semantic similarity between the vectors. If the value is closer to 1, both vectors share similar semantic information.

**Step 1**: Calculate the dot product between vector A (from token *sleep* in sentence A, $A=[a_1, a_2, a_3, ... a_n]$) and B (from the same token in sentence B, $B=[b_1, b_2, b_3 ..., b_n]$). Dot product seeks to measure similarity between vectors in terms of directions (positive for same direction, negative for opposite direction). It works by multiplying each component (e.g., $a_i \times b_i$) and then summing them up.

$$
\begin{align}
A\cdot B &=\sum_i A_i \times B_i = 0.846890^2 + 0.588220^2 + (-0.617240)^2 + \: ... \: + (-0.115990)^2 = 1.7331
\end{align}
$$

**Step 2**: Compute the magnitude (norm, $\|A\|$ and $\|B\|$). Norm or magnitude denotes the length of a vector in $n$-dimensional space. It is called normalization because it uses the Euclidean normalization (L2 norm) formula where $\|A\|=\sqrt{a_1^2, a_2^2, a_3^2 + ... + a_n^2}$. 

$$
\begin{align}
\|A\| &=\sqrt{\sum_i A_i^2}=\sqrt{1.7331} = 1.3165\\
\|B\| &=\sqrt{\sum_i B_i^2} = \sqrt{1.7331}= 1.3165
\end{align}
$$

**Step 3**: Calculate cosine similarity. Cosine similarity is the angle between two vectors, computed by dividing the dot product (similarity in direction and magnitude) by the product of norms. The division is done to remove the effect of vector length so the directional similarity can be defined. In its output interpretation, 1.0 ($\theta=0 \degree$) means identical direction and therefore highly similar meaning while -1.0 ($\theta=180\degree$) means opposite meaning. Value 0.0 ($\theta=90\degree$) also possible, meaning completely unrelated.

$$
\text{Cosine Similarity}=\frac{A \cdot B}{\|A\|\cdot \|B\|} = \frac{1.7331}{1.3165 \times 1.3165} =\frac{1.7331}{1.7331}=1.00
$$


## **3 Contextual Dense Embeddings**
On the contrary to static embeddings, contextual embeddings such as BERT and RoBERTa will encode both semantic and syntactic information as well. The embedding vectors even for exactcly the same word with the same word class (e.g., *sleep* (verb) in sentence 1 vs *sleep* (verb) in sentence 2) will be different depending on the context of use (see Image XXX).

But prior to explaining their differences, let us see the embedding process works, starting from how contextual embeddings are built on tokenized texts in the form of (sub)words and sentences to how embedding vectors get updated in every hidden layer in the transformer blocks.

### **3.1 Tokenization**
Prior to converting tokens into vectors (or *vectorization* for short), a piece of text should be tokenized. By breaking text into smaller and reusable pieces (or *tokens*), we can build a language model from an algorithm that learns from patterns of the small discrete units. But what do small discrete units have something to do with a machine learning algorithm? By breaking text into smaller pieces, a model can recognize morphological structures of words and syntactical relations between words. A model can also assign meaning to each token or subword. Splitting texts into pieces allows a model to learn patterns from repeated use of words. 

Language models have various tokenization techniques such as whitespace-based approach, byte-pair encoding (BPE), WordPiece encoding, SentencePiece encoding, and byte-level BPE, to name a few. In short, (1) the core idea of the first approach is to split text by spaces or punctuation marks, resulting in one token per word. This works for languages whose writing systems utilize whitespace but may not be ideal for Chinese or Japanese. (2) Dissimilar to per-word tokenization technique, BPE works in subword levels by merging <u>frequent character pairs</u>. This tokenization technique is based on Gage's ([1994](https://dl.acm.org/doi/10.5555/177910.177914)) idea for data compression algorithm which was later adopted in machine translation to address out-of-vocabulary issue (Sennrich et al, [2015](https://doi.org/10.48550/arXiv.1508.07909)). (3) WordPiece is very much similar to BPE, except for utilizing likelihood of character pairs  ( $P(\text{char01} \mid \text{char02})$ ). The goal of using likelihood instead of raw frequency is to make the tokenization more probabilistic. Frequency alone as in BPE can be misleading as BPE decides a token by considering how many times letter-01 co-occur with letter-02. Likelihod, on the other hand, decides a token based on how much two letters improve model's ability to predict next token. (3) SentencePiece treats text as a stream of bytes. (4) Byte-level BPE is fundameentally the BPE algorithm which operates on the byte-level so that all writing systems, including emojis, can be accommodated. I will explain some of these later.

## **3.2 Embeddings**
Now let us take a look at the tokenization process in Image 3. The raw sentence (text input) is split up into smaller pieces with byte-pair BPE. As previously mentioned, BPE splits text into sub-word units in RoBERTa vocabulary. To note, symbol `Ġ` marks where a whitespace occurs before the token inside a sentence boundary. This symbol is a way to tell a model about word boundaries. After the tokenization, each token will be mapped to a unique identifier (integer) from RoBERTa's vocabulary. The IDs are necessary because the next step is to retrieve dense embedding vectors based on the token IDs from RoBERTa. But these vectors are just input token embeddings. To get final input embedding vectors, positional embedding must be summed with positional embeddings ($X_t = E_t+P_t$). Without positional embeddings, the model cannot understand the positions of the tokens (*Colorless green ideas sleep furiously* $\ne$ *Green sleep colorless furiously\**). The element-wise matrix addition adds more information without changing the dimensionality of the embeddings.

<img src="img/roberta-tokenization-flow.png" width=800></br>
**Image 3**: Tokenization process in RoBERTa

The final input embeddings are then passed through 12 transformer encoder block in which the token representation is refined (updated) in each layer based on context (word order, meaning relations, and discourse dependencies such as referential dependencies, logical relationships). The refinement process is done by the multi-head self-attention along with feed-forward neural network. In multi-head self-attention step, each token embedding $x_i$ (e.g., *sleep*) goes through 3 linear projects, namely query vector ($Q_i=x_iW_Q$), key vector ($K_i=x_iW_K$), and value vector ($V_i=x_iW_V$). These are elements used to compute attention scores (how much each token should attend to every token): 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$ 

The $\sqrt{d_k}$ denotes the dimension of key vector. The $d_k$ is square rooted for normalizing the dot product between $Q$ and transposed version of $K$. Softmax here is used to convert the scores into attention weights (1.00 in total per token). The attention weights are then multiplied by $V$.


>[!note]
>**Sentence 1**: *Colorless green ideas **sleep** furiously.*</br>
>**Sentence 2**: *I **sleep** peacefully at night.*

<img src="img/token-sleep-roberta.png" alt="comparison between token sleep in two sentences with contextual embeddings" width="250"/></br>
