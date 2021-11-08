import argparse
import os

import pyprind

import utils
import treetk

def main(args):
    path = args.path

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".edus.arcs")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        # dependency sub-arcs for each EDU
        edus_arcs = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split())

        heads = []
        for arcs in edus_arcs:
            arcs = treetk.hyphens2arcs(arcs)

            # Check: Arcs should be arranged in ascending order wrt dependent
            prev_d = -1
            for h,d,l in arcs:
                assert d > prev_d
                prev_d = d

            head_idx = None

            # If its head is the root, it is the head of the EDU
            for idx, (h,d,l) in enumerate(arcs):
                if h == 0:
                    assert l == "ROOT" # TODO
                    head_idx = idx
                    break

            # If its head is outside the span, it is the head of the EDU
            if head_idx is None:
                span_min = arcs[0][1]
                span_max = arcs[-1][1]
                for idx, (h,d,l) in enumerate(arcs):
                    if h < span_min or h > span_max:
                        head_idx = idx
                        break

            heads.append(head_idx)

        # Write
        with open(os.path.join(path, filename.replace(".edus.arcs", ".edus.heads")), "w") as f:
            for head_idx in heads:
                f.write("%d\n" % head_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args)
