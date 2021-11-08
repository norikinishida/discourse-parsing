import argparse
import os

import spacy
import pyprind

import utils
import treetk


def main(args):
    path_in = args.input
    path_out = args.output

    utils.mkdir(path_out)

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    filenames = os.listdir(path_in)
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        if filename == "file1.edus":
            filename2 = "wsj_0764.edus.tokens"
        elif filename == "file2.edus":
            filename2 = "wsj_0430.edus.tokens"
        elif filename == "file3.edus":
            filename2 = "wsj_0766.edus.tokens"
        elif filename == "file4.edus":
            filename2 = "wsj_0778.edus.tokens"
        elif filename == "file5.edus":
            filename2 = "wsj_2172.edus.tokens"
        else:
            filename2 = filename.replace(".out.edus", ".edus.tokens")

        # Read and write one document
        with open(os.path.join(path_out, filename2), "w") as f:
            for line in open(os.path.join(path_in, filename)):
                line = line.strip()
                if line == "":
                    tokens = ""
                else:
                    doc = nlp(line)
                    tokens = [token.text for token in doc]
                    tokens = " ".join(tokens)
                f.write("%s\n" % tokens)

        # Read and write the trees
        with open(os.path.join(
                    path_out,
                    filename2.replace(".edus.tokens", ".labeled.nary.ctree")), "w") as f_labeled_nary, \
             open(os.path.join(
                    path_out,
                    filename2.replace(".edus.tokens", ".unlabeled.nary.ctree")), "w") as f_unlabeled_nary, \
             open(os.path.join(
                    path_out,
                    filename2.replace(".edus.tokens", ".labeled.bin.ctree")), "w") as f_labeled_bin, \
             open(os.path.join(
                    path_out,
                    filename2.replace(".edus.tokens", ".unlabeled.bin.ctree")), "w") as f_unlabeled_bin:

            sexp = treetk.rstdt.read_sexp(os.path.join(path_in, filename.replace(".edus", ".dis")))
            tree = treetk.rstdt.sexp2tree(sexp)
            tree = treetk.rstdt.shift_labels(tree)

            f_labeled_nary.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=True))
            f_unlabeled_nary.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=False))

            # Binarize
            tree = treetk.rstdt.binarize(tree)
            tree = treetk.rstdt.shift_labels(tree)

            f_labeled_bin.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=True))
            f_unlabeled_bin.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

