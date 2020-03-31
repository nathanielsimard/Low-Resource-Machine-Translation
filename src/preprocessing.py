"""Module having a collection a utilities for doing preprocessing on text."""
OUT_OF_SAMPLE_TOKEN = "<out>"
START_OF_SAMPLE_TOKEN = "startofsampletoken"
END_OF_SAMPLE_TOKEN = "endofsampletoken"


def add_start_end_token(corpus):
    """Add an indicater token for the beginning and the end of a text sample."""
    return [
        f"{START_OF_SAMPLE_TOKEN} {token} {END_OF_SAMPLE_TOKEN}" for token in corpus
    ]
