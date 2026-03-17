from src.dataset import TextDataset
from src.model import Word2Vec
import numpy as np


def load_text():
    """
    Load the training corpus from a text file.

    The corpus is stored in the `data` directory and is read as a UTF-8
    encoded string.

    Returns
    -------
    str
        Raw text content of the corpus.
    """
    with open("data/sample_corpus.txt", "r", encoding="utf-8") as f:
        return f.read()


def show_similar(model, dataset, word, top_k=5):
    """
    Display the most similar words to a given query word.

    This function looks up the word in the dataset vocabulary, retrieves
    its embedding from the trained model, computes cosine similarities
    against all other embeddings, and prints the top results.

    Parameters
    ----------
    model : Word2Vec
        Trained Word2Vec model.

    dataset : TextDataset
        Dataset object containing vocabulary mappings.

    word : str
        Query word whose nearest neighbours should be displayed.

    top_k : int, default=5
        Number of similar words to print.
    """
    # Check that the query word exists in the vocabulary
    if word not in dataset.word_to_id:
        print(f"Word '{word}' not in vocabulary.")
        return

    # Convert the word into its vocabulary ID
    wid = dataset.word_to_id[word]

    # Retrieve the most similar word IDs from the model
    sims = model.most_similar(wid, top_k=top_k)

    # Convert IDs back into readable words
    decoded = [(dataset.id_to_word[i], round(score, 4)) for i, score in sims]
    print(f"Nearest to '{word}': {decoded}")


def show_analogy(model, dataset, w1, w2, w3, top_k=5):
    """
    Display the result of a word analogy query.


    Parameters
    ----------
    model : Word2Vec
        Trained Word2Vec model.

    dataset : TextDataset
        Dataset object containing vocabulary mappings.

    w1 : str
        First word in the analogy expression.

    w2 : str
        Second word in the analogy expression.

    w3 : str
        Third word in the analogy expression.

    top_k : int, default=5
        Number of candidate answers to print.
    """
    # Ensure all analogy words exist in the vocabulary
    for w in [w1, w2, w3]:
        if w not in dataset.word_to_id:
            print(f"Word '{w}' not in vocabulary.")
            return

    # Compute analogy result using vector arithmetic in embedding space
    result = model.analogy(
        dataset.word_to_id[w1],
        dataset.word_to_id[w2],
        dataset.word_to_id[w3],
        top_k=top_k
    )

    # Decode vocabulary IDs into readable words
    decoded = [(dataset.id_to_word[i], round(score, 4)) for i, score in result]
    print(f"{w1} - {w2} + {w3} ≈ {decoded}")


def main():
    """
    Train the Word2Vec model and run basic evaluation.

    This function executes the full training pipeline:
    1) Load the corpus.
    2) Build the vocabulary and encode the text.
    3) Generate skip-gram training pairs.
    4) Initialize the Word2Vec model.
    5) Train the model for a fixed number of epochs.
    6) Print sanity checks.
    7) Evaluate the embeddings using nearest-neighbour and analogy tests.
    """
    # Load raw training text
    text = load_text()

    # Prepare dataset: tokenize, build vocabulary, encode corpus, and generate pairs
    dataset = TextDataset(window_size=2, min_count=1)
    encoded = dataset.build_vocab(text)
    pairs = dataset.generate_pairs(encoded)

    # Vocabulary size determines embedding matrix dimensions
    vocab_size = len(dataset.word_to_id)

    # Initialize skip-gram Word2Vec model with negative sampling
    model = Word2Vec(
        vocab_size=vocab_size,
        embedding_dim=30,
        neg_samples=4,
        lr=0.03
    )

    # Number of full passes through the training data
    epochs = 200


    # Training loop
    for epoch in range(epochs):
        # Shuffle training pairs at the start of each epoch
        np.random.shuffle(pairs)

        total_loss = 0.0

        # Perform one SGD update per training pair
        for center, context in pairs:
            total_loss += model.train_step(center, context, dataset.neg_probs)

        # Print average loss for monitoring training progress
        print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(pairs):.4f}")

    # Sanity checks
    print("\nSanity checks")
    # Sanity checks verify that embedding matrices have the expected dimensions
    print("W_in shape:", model.W_in.shape)
    print("W_out shape:", model.W_out.shape)
    #This checks whether any value in the embedding matrix became NaN, because of illegal operations
    print("NaNs in W_in:", np.isnan(model.W_in).any())
    print("NaNs in W_out:", np.isnan(model.W_out).any())

    print("\nNearest-neighbour tests")

    # Words related to JetBrains tools, programming languages, and developers
    evaluation_words = [
        "jetbrains",
        "pycharm",
        "intellij",
        "webstorm",
        "clion",
        "developer",
        "programmer",
        "software",
        "python",
        "java"
    ]

    for word in evaluation_words:
        show_similar(model, dataset, word, top_k=5)

    print("\nAnalogy tests")

    # IDE / language relationship
    show_analogy(model, dataset, "pycharm", "python", "java", top_k=3)

    # WebStorm ↔ JavaScript relationship
    show_analogy(model, dataset, "webstorm", "javascript", "python", top_k=3)

    # Developer / programmer relationship
    show_analogy(model, dataset, "developer", "code", "software", top_k=3)


if __name__ == "__main__":
    main()
