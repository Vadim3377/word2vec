# Word2Vec Implementation in Pure NumPy

This project implements the **core training loop of Word2Vec** using **pure NumPy**, without relying on machine learning frameworks such as PyTorch or TensorFlow.

The goal of this implementation is educational: to explicitly implement the optimization procedure used in Word2Vec, including:

* forward pass
* loss computation
* gradient calculation
* parameter updates

The model implemented is **Skip-Gram with Negative Sampling (SGNS)**.

---

# Project Goals

The objective of this project is to demonstrate a full understanding of how Word2Vec works internally by implementing:

* vocabulary construction
* skip-gram training pair generation
* negative sampling
* embedding training with stochastic gradient descent
* cosine similarity evaluation

All computations are implemented manually using **NumPy vector operations**.

---

# Word2Vec Overview

Word2Vec learns **dense vector representations of words** such that words appearing in similar contexts have similar embeddings.

This implementation uses the **Skip-Gram architecture**, which predicts surrounding context words given a center word.

Example training pair:

```
Sentence: "IDE improves developer productivity"

Center word: developer  
Context words: improves, productivity
```

The model learns embeddings that maximize the probability of observing context words given the center word.

---

# Negative Sampling

Computing a full softmax over the entire vocabulary is expensive.

Instead, **negative sampling** approximates the objective by training the model to:

* increase similarity between real word-context pairs
* decrease similarity between randomly sampled word pairs

The loss for one training pair is:

```
L = -log f(v_c * v_o) - Sigma( log( f(-v_c · v_k)))
```

Where:

* (v_c) = center word embedding
* (v_o) = true context embedding
* (v_k) = negative sample embeddings
* f = sigmoid function

This implementation uses the **standard unigram distribution raised to the power of 0.75** for sampling negative examples.

---

# Training Procedure

The training loop performs the following steps:

1. Build vocabulary from the corpus
2. Generate skip-gram pairs using a sliding window
3. Sample negative examples
4. Compute the loss
5. Compute gradients manually
6. Update parameters using stochastic gradient descent

Two embedding matrices are trained:

```
W_in   : center word embeddings
W_out  : context word embeddings
```

After training, `W_in` contains the final word vectors.

---
# How to Run

Install dependencies:

```
pip install numpy
```

Run training:

```
python main.py
```

The script will:

1. Load the corpus
2. Build the vocabulary
3. Train the Word2Vec model
4. Print example similarity queries

Example output:

```
Training started...
Epoch 1 | Loss: 3.42
Epoch 2 | Loss: 2.91
Epoch 3 | Loss: 2.37

Nearest neighbors for "python":
developer
language
programming
numpy
```

---

# Example Corpus

The repository includes a small sample corpus (`sample_corpus.txt`) used for demonstration purposes.

The corpus is intentionally small to make training fast and the code easy to understand.

For more meaningful embeddings, the model can be trained on larger datasets such as:

* Wikipedia text
* OpenSubtitles
* news datasets
* large book corpora

---

# Limitations

This implementation prioritizes **clarity over performance**.

It intentionally omits several optimizations used in production Word2Vec implementations:

* mini-batch training
* subsampling of frequent words
* learning rate decay
* hierarchical softmax
* vectorized negative sampling
* multi-threaded training

These features could significantly improve training speed and embedding quality.

---

# Possible Improvements

Potential extensions include:

* implementing CBOW architecture
* adding subsampling of frequent words
* implementing hierarchical softmax
* supporting mini-batch training
* training on a large corpus
* visualizing embeddings with t-SNE

