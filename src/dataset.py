import numpy as np
from collections import Counter
from src.utils import tokenize


class TextDataset:
    """
    Utility class responsible for preparing text data for Word2Vec training.

    This class performs several preprocessing steps required before training:
     1) tokenizing the input text
     2) building a vocabulary
     3) assigning integer IDs to words
     4) computing the negative sampling distribution
     5) generating skip-gram training pairs

    Parameters
    ----------
    window_size : int
        Number of words to consider on each side of a center word when
        generating context pairs.

    min_count : int
        Minimum frequency a word must have in order to be included in the
        vocabulary. Words occurring fewer times are discarded.
    """

    def __init__(self, window_size=2, min_count=1):
        self.window_size = window_size
        self.min_count = min_count

    def build_vocab(self, text):
        """
        Build the vocabulary and encode the text as integer word IDs.

        This method performs the following steps:
            1) Tokenizes the input text into words.
            2) Counts word frequencies.
            3) Filters words according to the minimum frequency threshold.
            4) Assigns a unique integer ID to each vocabulary word.
            5) Computes the negative sampling probability distribution.
            6) Converts the original token sequence into integer IDs.

        Negative sampling probabilities follow the standard Word2Vec rule
        where the unigram distribution is raised to the power of 0.75.

        Parameters
        ----------
        text : str
            Raw input text corpus.

        Returns
        -------
        list[int]
            Encoded version of the corpus where each word is represented
            by its vocabulary ID.
        """

        # Convert text into a list of tokens
        tokens = tokenize(text)

        # Count the frequency of each word in the corpus
        counter = Counter(tokens)

        # Keep only words that meet the minimum frequency requirement
        vocab = [w for w, c in counter.items() if c >= self.min_count]
        vocab.sort()

        # Create mappings between words and their integer IDs
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

        # Convert word counts into a NumPy array
        counts = np.array([counter[w] for w in vocab], dtype=np.float64)

        # Compute negative sampling distribution
        # Word2Vec uses unigram^(0.75) to balance frequent and rare words
        probs = counts ** 0.75
        self.neg_probs = probs / probs.sum()

        # Encode the entire corpus as integer IDs
        encoded = [self.word_to_id[w] for w in tokens if w in self.word_to_id]

        return encoded

    def generate_pairs(self, encoded):
        """
        Generate skip-gram training pairs from an encoded corpus.

        For each word in the sequence (the center word), this function
        collects surrounding context words within the specified window
        size and creates training pairs of the form:

            (center_word, context_word)

        These pairs are used to train the skip-gram Word2Vec model.

        Parameters
        ----------
        encoded : list[int]
            Corpus encoded as integer word IDs.

        Returns
        -------
        list[tuple[int, int]]
            List of (center, context) word ID pairs.
        """

        pairs = []

        # Iterate over every word position in the corpus
        for i in range(len(encoded)):

            center = encoded[i]  # Current word acts as the center word

            # Determine context window boundaries
            left = max(0, i - self.window_size)
            right = min(len(encoded), i + self.window_size + 1)

            # Collect context words within the window
            for j in range(left, right):

                # Skip the center word itself
                if i == j:
                    continue

                # Add (center, context) pair
                pairs.append((center, encoded[j]))

        return pairs