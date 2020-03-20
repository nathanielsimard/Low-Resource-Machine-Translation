def read_token_file(file_name: str):
    out = []
    with open(file_name, 'r') as stream:
        for line in stream:
            tokens = line.strip().split()
            out.append(tokens)
    return out


def read_text_file(file_name: str):
    out = []
    with open(file_name, 'r') as stream:
        for line in stream:
            tokens = line.strip()
            out.append(tokens)
    return out


def write_text_from_tokens(sentences, output_file):
    with open(output_file, 'w+') as out_stream:
        for tokens in sentences:
            out_stream.write(' '.join(tokens) + '\n')
