import argparse
import re

import utils


def process_line(line):
    """
    Parameters
    ----------
    line: str

    Returns
    -------
    str
    """
    tokens = line.split()
    tokens = [process_token(token) for token in tokens]
    tokens = " ".join(tokens)
    return tokens


def process_token(token):
    """
    Parameters
    ----------
    token: str

    Returns
    -------
    str
    """
    token = token.lower()
    token = re.sub(r"\d", "0", token)
    return token


def main(args):
    if args.filelist is not None:
        files = utils.read_lines(args.filelist)
    else:
        files = args.files

    paths_in = files
    paths_out = [path_in + ".preprocessed" for path_in in paths_in]
    utils.read_process_and_write(paths_in, paths_out, process=process_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--filelist", type=str, default=None)
    args = parser.parse_args()
    main(args)
