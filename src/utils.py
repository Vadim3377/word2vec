import re
import numpy as np


def tokenize(text):
    """
    Tokenize a text string into a list of lowercase word tokens.

    This function performs basic text preprocessing by extracting words using a regular expression.

    The regular expression selects sequences of alphabetic characters
    (a–z) and apostrophes. This allows simple handling of contractions
    such as "don't" or "it's".

    Parameters
    ----------
    text : str
        Input text that will be tokenized.

    Returns
    -------
    list[str]
        A list of word tokens extracted from the input text.

    """
    text = text.lower()  # Normalize text by converting all characters to lowercase
    return re.findall(r"\b[a-z']+\b", text)  # Extract words using regex

def sigmoid(x):
    """
    Compute the sigmoid function.

    The sigmoid function maps any real-valued number to the range (0, 1).
    It is commonly used in machine learning models to convert scores
    into probabilities.

    In Word2Vec with negative sampling, the sigmoid function is used
    to estimate the probability that a word-context pair is valid.

    To improve numerical stability, the input values are clipped to the
    range [-15, 15] before applying the exponential function. This
    prevents overflow when computing exp(x).

    Mathematical definition
    -----------------------
    delta(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value or array.

    Returns
    -------
    float or numpy.ndarray
        Sigmoid activation applied to the input.
    """
    x = np.clip(x, -15, 15)  # Prevent numerical overflow in exp()
    return 1 / (1 + np.exp(-x))  # Apply sigmoid function