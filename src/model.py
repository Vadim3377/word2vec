import numpy as np
from src.utils import sigmoid


class Word2Vec:
    """
    Skip-gram Word2Vec model with Negative Sampling implemented in pure NumPy.

    This class stores two embedding matrices:

    - W_in:  input embeddings for center words
    - W_out: output embeddings for context words

    During training, the model learns to assign high similarity to real
    (center, context) pairs and low similarity to randomly sampled
    negative pairs.

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary.

    embedding_dim : int, default=50
        Dimensionality of the word embeddings.

    neg_samples : int, default=5
        Number of negative samples to draw for each positive pair.

    lr : float, default=0.025
        Learning rate for stochastic gradient descent.
    """

    def __init__(self, vocab_size, embedding_dim=50, neg_samples=5, lr=0.025):
        """
        Initialize model parameters.

        The input embedding matrix is initialized randomly with small values.
        The output embedding matrix is initialized to zeros, which is a common
        simple initialization for Word2Vec implementations.

        Parameters
        ----------
        vocab_size : int
            Number of words in the vocabulary.

        embedding_dim : int, optional
            Size of each word vector.

        neg_samples : int, optional
            Number of negative examples per training step.

        lr : float, optional
            Learning rate used for SGD updates.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_samples = neg_samples
        self.lr = lr

        # Small random initialization for input embeddings
        scale = 0.5 / embedding_dim
        self.W_in = np.random.uniform(-scale, scale, (vocab_size, embedding_dim))

        # Output embeddings start from zero
        self.W_out = np.zeros((vocab_size, embedding_dim))

    def sample_negatives(self, probs, positive_id):
        """
        Sample negative word IDs for negative sampling.

        Negative examples are drawn according to the provided probability
        distribution, typically based on unigram frequencies raised to
        the power of 0.75. The true positive context word is excluded.

        Parameters
        ----------
        probs : numpy.ndarray
            Probability distribution over vocabulary words used for
            negative sampling.

        positive_id : int
            Vocabulary ID of the true context word, which must not be
            included among the negative samples.

        Returns
        -------
        list[int]
            List of sampled negative word IDs.
        """
        negatives = []

        # Keep sampling until the desired number of negative examples is reached
        while len(negatives) < self.neg_samples:
            idx = np.random.choice(len(probs), p=probs)

            # Exclude the true context word from the negative set
            if idx != positive_id:
                negatives.append(idx)

        return negatives

    def train_step(self, center, context, neg_probs):
        """
        Perform one SGD training step for a single skip-gram pair.

        The model receives one positive pair:
            (center word, true context word)

        It also samples several negative words and optimizes the
        negative sampling objective:

            log delta(v * u_pos) + Sigma( log (delta(-v · u_neg)))

        where:
        - v is the input embedding of the center word
        - u_pos is the output embedding of the true context word
        - u_neg are output embeddings of sampled negative words

        This method computes:
        - forward pass
        - loss
        - gradients
        - parameter updates

        Parameters
        ----------
        center : int
            Vocabulary ID of the center word.

        context : int
            Vocabulary ID of the true context word.

        neg_probs : numpy.ndarray
            Probability distribution used for negative sampling.

        Returns
        -------
        float
            Scalar loss value for this training example.
        """
        # Input embedding of the center word
        v = self.W_in[center]

        # Output embedding of the positive context word
        u_pos = self.W_out[context]

        # Sample negative context words
        negatives = self.sample_negatives(neg_probs, context)
        u_neg = self.W_out[negatives]


        # Forward pass

        # Score for the positive pair
        pos_score = np.dot(v, u_pos)
        pos_sig = sigmoid(pos_score)

        # Scores for the negative pairs
        neg_scores = u_neg @ v
        neg_sig = sigmoid(neg_scores)

        # Negative sampling loss:
        # -log σ(v·u_pos) - Σ log(1 - σ(v·u_neg))
        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(1.0 - neg_sig + 1e-10))


        # Gradient computation

        # Gradient of the positive term
        grad_pos = pos_sig - 1.0

        # Gradient of the negative terms
        grad_neg = neg_sig

        # Gradient with respect to center embedding v
        grad_v = grad_pos * u_pos + np.sum(grad_neg[:, None] * u_neg, axis=0)

        # Gradient with respect to positive output embedding
        grad_u_pos = grad_pos * v

        # Gradient with respect to negative output embeddings
        grad_u_neg = grad_neg[:, None] * v


        # SGD parameter updates

        # Update center word embedding
        self.W_in[center] -= self.lr * grad_v

        # Update true context embedding
        self.W_out[context] -= self.lr * grad_u_pos

        # Update sampled negative embeddings
        for i, neg_id in enumerate(negatives):
            self.W_out[neg_id] -= self.lr * grad_u_neg[i]

        return loss

    def most_similar(self, word_id, top_k=5):
        """
        Find the most similar words to a given word using cosine similarity.

        Similarity is computed between the input embedding of the query word
        and all other input embeddings in the vocabulary.

        Parameters
        ----------
        word_id : int
            Vocabulary ID of the query word.

        top_k : int, default=5
            Number of most similar words to return.

        Returns
        -------
        list[tuple[int, float]]
            List of (word_id, cosine_similarity) pairs sorted by similarity.
        """
        # Query embedding
        target = self.W_in[word_id]

        # Compute cosine similarity between the target and all vocabulary words
        target_norm = np.linalg.norm(target) + 1e-10
        all_norms = np.linalg.norm(self.W_in, axis=1) + 1e-10

        sims = (self.W_in @ target) / (all_norms * target_norm)

        # Exclude the word itself
        sims[word_id] = -np.inf

        # Return IDs of the most similar words
        best_ids = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in best_ids]

    def analogy(self, a, b, c, top_k=5):
        """
        Solve a word analogy using vector arithmetic.

        The method computes:

            embedding(a) - embedding(b) + embedding(c)

        and returns the words whose embeddings are closest to the result.

        Parameters
        ----------
        a : int
            Vocabulary ID of the first word.

        b : int
            Vocabulary ID of the second word.

        c : int
            Vocabulary ID of the third word.

        top_k : int, default=5
            Number of candidate answers to return.

        Returns
        -------
        list[tuple[int, float]]
            List of (word_id, cosine_similarity) pairs representing the
            closest analogy matches.
        """
        # Construct the analogy vector
        vec = self.W_in[a] - self.W_in[b] + self.W_in[c]

        # Compute cosine similarity against all embeddings
        vec_norm = np.linalg.norm(vec) + 1e-10
        all_norms = np.linalg.norm(self.W_in, axis=1) + 1e-10

        sims = (self.W_in @ vec) / (all_norms * vec_norm)

        # Exclude the source words from the candidate answers
        sims[a] = -np.inf
        sims[b] = -np.inf
        sims[c] = -np.inf

        # Return the best matches
        best_ids = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in best_ids]