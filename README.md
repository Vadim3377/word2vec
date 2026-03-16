
# Word2Vec Implementation in Pure NumPy

This project implements the **core training loop of Word2Vec** using **pure NumPy**, without relying on machine learning frameworks such as PyTorch or TensorFlow. The goal is to explicitly implement the optimization procedure behind Word2Vec, including the **forward pass, loss computation, gradient calculation, and parameter updates**.

Word2Vec is a neural embedding method that learns **dense vector representations of words** from text corpora. The central idea is that **words appearing in similar contexts tend to have similar meanings**. During training, the model learns embeddings where semantically related words become close in vector space.

This implementation uses the **Skip-Gram architecture with Negative Sampling**, a commonly used Word2Vec variant. In the skip-gram model, the network learns to predict surrounding context words given a center word. Instead of computing a full softmax over the entire vocabulary, **negative sampling** is used to efficiently approximate the training objective by distinguishing real word-context pairs from randomly sampled negative examples.

---

# Implementation Overview

The implementation follows the standard Word2Vec pipeline:

### 1. Text preprocessing

The input corpus is tokenized and normalized. A vocabulary of unique words is constructed, and each word is mapped to an integer ID.

### 2. Training pair generation

Using a configurable context window, the corpus is converted into **skip-gram training pairs** of the form:

```
(center_word, context_word)
```

These pairs represent local contextual relationships in the text.

### 3. Model initialization

Two embedding matrices are initialized:

* **W_in** — input embeddings for center words
* **W_out** — output embeddings for context words

Each row corresponds to the vector representation of a word.

### 4. Core training loop

For each training pair:

1. A **positive pair** (real context) is used.
2. Several **negative samples** are drawn from the vocabulary.
3. The model computes similarity scores between word embeddings.
4. A **negative sampling loss** is calculated.
5. Gradients are derived manually.
6. Parameters are updated using **stochastic gradient descent (SGD)**.

The loss decreases during training, indicating that the embeddings improve.

---

# Evaluation

After training, the learned embeddings are evaluated qualitatively using:

### Nearest neighbour queries

Words with similar embeddings are retrieved using cosine similarity.

Example:

```
pycharm => intellij, webstorm, clion
```

### Analogy tests

Vector arithmetic demonstrates semantic relationships:

```
pycharm - python + java = intellij
```

These tests show that the model captures contextual relationships between words.


---

# Project Structure

```
src/
    dataset.py    # Text preprocessing and training pair generation
    model.py      # Word2Vec model and training logic
    utils.py      # Tokenization and mathematical utilities

train.py          # Training script and evaluation
data/
    sample_corpus.txt
```

=


