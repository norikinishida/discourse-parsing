SPEAKERS = [
        "Alex", "Bruce", "Chris", "David", "Elliott", "Fred", "George",
        "Henry", "Isaac", "John", "Kim", "Lee", "Mick", "Nicolas",
        "Owen", "Patrick", "Richard", "Scott", "Tom"]


def rename_speaker_names(tokens, speaker_name_to_id_uncased):
    """
    Parameters
    ----------
    tokens: list[str]
    speaker_name_to_id_uncased: dict[str, int]

    Returns
    -------
    list[str]
    """
    new_tokens = []
    for token in tokens:
        token_uncased = token.lower()
        if token_uncased in speaker_name_to_id_uncased:
            token = SPEAKERS[speaker_name_to_id_uncased[token_uncased]]
        new_tokens.append(token)
    return new_tokens


