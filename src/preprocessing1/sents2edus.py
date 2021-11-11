import argparse
import os

import pyprind

import utils

def main(args):
    path = args.path

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):

        edus = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split()) # List[List[str]]
        sents = utils.read_lines(os.path.join(path, filename.replace(".edus.tokens", ".sents.tokens")), process=lambda line: line.split()) # List[List[str]]
        sents_postags = utils.read_lines(os.path.join(path, filename.replace(".edus.tokens", ".sents.postags")), process=lambda line: line.split()) # List[List[str]]
        sents_arcs = utils.read_lines(os.path.join(path, filename.replace(".edus.tokens", ".sents.arcs")), process=lambda line: line.split()) # List[List[str]]
        postags = utils.flatten_lists(sents_postags) # List[str]
        arcs = utils.flatten_lists(sents_arcs) # List[str]

        # Ending positions of gold EDUs
        edu_end_positions = []
        tok_i = 0
        for edu in edus:
            length = len(edu)
            edu_end_positions.append(tok_i + length - 1)
            tok_i += length

        # Ending positions of sentences
        sent_end_positions = []
        tok_i = 0
        for sent in sents:
            length = len(sent)
            sent_end_positions.append(tok_i + length - 1)
            tok_i += length

        # All the ending positions of sentences must match with those of gold EDUs
        assert set(sent_end_positions).issubset(set(edu_end_positions))
        # assert set(sent_end_positions) == set(edu_end_positions) & set(sent_end_positions)

        # Sentence boundaries
        sentence_boundaries = []
        tok_i = 0
        sent_i = 0
        begin_edu_i = 0
        for end_edu_i, edu in enumerate(edus):
            tok_i += len(edu)
            if tok_i - 1 == sent_end_positions[sent_i]:
                sentence_boundaries.append((begin_edu_i, end_edu_i))
                sent_i += 1
                begin_edu_i = end_edu_i + 1
        assert sent_i == len(sent_end_positions)
        with open(os.path.join(path, filename.replace(".edus.tokens", ".sentence_boundaries")), "w") as f:
            # for begin_i, end_i in sentence_boundaries:
            #     f.write("%d %d\n" % (begin_i, end_i))
            f.write("%s\n" % " ".join(["%d-%d" % (b, e) for b, e in sentence_boundaries]))

        # Extract POS tags and dependency arcs corresponding to each EDU
        with open(os.path.join(path, filename.replace(".edus.tokens", ".edus.postags")), "w") as fp,\
             open(os.path.join(path, filename.replace(".edus.tokens", ".edus.arcs")), "w") as fa:
            begin_tok_i = 0
            for edu in edus:
                length = len(edu)

                sub_postags = postags[begin_tok_i:begin_tok_i+length]
                sub_postags = " ".join(sub_postags)
                fp.write("%s\n" % sub_postags)

                sub_arcs = arcs[begin_tok_i:begin_tok_i+length]
                sub_arcs = " ".join(sub_arcs)
                fa.write("%s\n" % sub_arcs)

                begin_tok_i += length
        assert begin_tok_i - 1 == edu_end_positions[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
