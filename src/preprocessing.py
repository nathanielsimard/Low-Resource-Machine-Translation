"""Module having a collection a utilities for doing preprocessing on text."""
OUT_OF_SAMPLE_TOKEN = "<out>"
# The word inside < > must be unique
# for start and end of sample token.
START_OF_SAMPLE_TOKEN = "<start>"
END_OF_SAMPLE_TOKEN = "<end>"


def add_start_end_token(corpus):
    """Add an indicater token for the beginning and the end of a text sample."""
    return [
        f"{START_OF_SAMPLE_TOKEN} {token} {END_OF_SAMPLE_TOKEN}" for token in corpus
    ]


def add_start_token(corpus):
    """Add an indicater token for the beginning and the end of a text sample."""
    return [
        f"{START_OF_SAMPLE_TOKEN} {token}" for token in corpus
    ]


def add_end_token(corpus):
    """Add an indicater token for the beginning and the end of a text sample."""
    return [
        f"{token} {END_OF_SAMPLE_TOKEN}" for token in corpus
    ]
