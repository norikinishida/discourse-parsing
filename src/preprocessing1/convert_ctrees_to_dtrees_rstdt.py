import argparse
import os

import pyprind

import utils
import treetk
import treetk.rstdt


def main(args):
    """
    We use n-ary ctrees (ie., *.labeled.nary.ctree) to generate dtrees.
    Morey et al. (2018) demonstrate that scores evaluated on these dtrees are superficially lower than those on right-heavy binarized trees (ie., *.labeled.bin.ctree).
    """
    path = args.path

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".labeled.nary.ctree")]
    filenames.sort()

    def func_label_rule(node, i, j):
        relations = node.relation_label.split("/")
        if len(relations) == 1:
            return relations[0] # Left-most node is head.
        else:
            if i > j:
                return relations[j]
            else:
                return relations[j-1]

    for filename in pyprind.prog_bar(filenames):
        sexp = utils.read_lines(
                    os.path.join(path, filename),
                    process=lambda line: line.split())
        assert len(sexp) == 1
        sexp = sexp[0]

        # Constituency
        ctree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))

        # Dependency
        # Assign heads
        ctree = treetk.rstdt.assign_heads(ctree)
        # Conversion
        dtree = treetk.ctree2dtree(ctree, func_label_rule=func_label_rule)
        arcs = dtree.tolist(labeled=True)

        # Write
        with open(os.path.join(
                    path,
                    filename.replace(".labeled.nary.ctree", ".arcs")), "w") as f:
            f.write("%s\n" % " ".join(["%d-%d-%s" % (h,d,l) for h,d,l in arcs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

