import argparse

import utils

def main(args):
    paths_vocab = args.vocabs
    path_out = args.output
    assert len(paths_vocab) > 1
    utils.concat_vocabularies(paths_vocab, path_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)
