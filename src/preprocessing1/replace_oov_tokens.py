import argparse

import utils

def main(args):
    if args.filelist is not None:
        paths_file = utils.read_lines(args.filelist)
    else:
        paths_file = args.files
    path_vocab = args.vocab

    utils.replace_oov_tokens(paths_in=paths_file, paths_out=paths_file, path_vocab=path_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--filelist", type=str, default=None)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()
    main(args)
