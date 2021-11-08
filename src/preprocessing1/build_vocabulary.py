import argparse

import utils

def main(args):
    if args.filelist is not None:
        paths_file = utils.read_lines(args.filelist)
    else:
        paths_file = args.files
    path_vocab = args.vocab

    utils.build_vocabulary(paths_file=paths_file,
                           path_vocab=path_vocab,
                           prune_at=50000,
                           min_count=-1,
                           special_words=["<root>"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--filelist", type=str, default=None)
    parser.add_argument("--vocab", type=str, required=True)
    args = parser.parse_args()
    main(args)
