import argparse
import os

import pyprind
import spacy

import utils


def main(args):
    input_file = args.input
    output_dir = args.output
    output_filename_prefix = args.output_filename_prefix

    utils.mkdir(output_dir)

    nlp_no_ssplit = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp_no_ssplit.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

    dialogues = utils.read_json(input_file)
    n_dialogues = len(dialogues)

    id_set = set()
    for dialogue in pyprind.prog_bar(dialogues):
        #####################
        # Dialogue
        #####################

        raw_edus = []
        speakers = []
        disc_arcs = []
        for item in dialogue["edus"]:
            raw_edus.append(" ".join(item["text"].strip().split()))
            speakers.append(item["speaker"])
        for item in dialogue["relations"]:
            head = item["x"] + 1
            dep = item["y"] + 1
            rel = item["type"]
            if rel == "C":
                rel = "Comment" # Fix annotation error
            disc_arcs.append((head, dep, rel))
        disc_arcs.append((0, 1, "<root>"))

        # Add a relation from the Root node to each EDU without an incoming edge (head)
        deps_with_head = set([d for h, d, r in disc_arcs])
        deps_all = set([d for d in range(1, len(raw_edus)+1)])
        deps_without_head = deps_all.difference(deps_with_head)
        for dep in deps_without_head:
            disc_arcs.append((0, dep, "<root>"))

        id = dialogue["id"]
        assert not id in id_set # ID must be unique to avoid filename confliction
        id_set.add(id)

        with open(os.path.join(output_dir, "%s_%s.speakers" % (output_filename_prefix, id)), "w") as f:
            for speaker in speakers:
                f.write("%s\n" % speaker)

        with open(os.path.join(output_dir, "%s_%s.sentence_boundaries" % (output_filename_prefix, id)), "w") as f:
            begin_i = 0
            end_i = len(raw_edus) - 1
            f.write("%d %d\n" % (begin_i, end_i))

        with open(os.path.join(output_dir, "%s_%s.paragraph_boundaries" % (output_filename_prefix, id)), "w") as f:
            f.write("0 0\n")

        with open(os.path.join(output_dir, "%s_%s.arcs" % (output_filename_prefix, id)), "w") as f:
            disc_arcs = sorted(disc_arcs, key=lambda x: x[1])
            disc_arcs = ["%d-%d-%s" % (h,d,l) for h,d,l in disc_arcs]
            disc_arcs = " ".join(disc_arcs)
            f.write("%s\n" % disc_arcs)

        edus_tokens = []
        edus_postags = []
        edus_arcs = []
        for raw_edu in raw_edus:
            doc = nlp_no_ssplit(raw_edu)
            sents_ = list(doc.sents)
            assert len(sents_) == 1
            sent = sents_[0]
            edu_tokens = [token.text for token in sent]
            edu_postags = [token.tag_ for token in sent]
            edu_arcs = []
            found_root = False
            for token in sent:
                head = token.head.i + 1
                dep = token.i + 1
                label = token.dep_
                if head == dep:
                    assert label == "ROOT"
                    assert not found_root # Only one token can be the root of dependency graph
                    head = 0
                    found_root = True
                syn_arc = (head, dep, label)
                edu_arcs.append(syn_arc)
            assert found_root
            edu_arcs = ["%d-%d-%s" % (h, d, l) for h, d, l in edu_arcs]
            edus_tokens.append(edu_tokens)
            edus_postags.append(edu_postags)
            edus_arcs.append(edu_arcs)

        with open(os.path.join(output_dir, "%s_%s.edus.tokens" % (output_filename_prefix, id)), "w") as f:
            for edu_tokens in edus_tokens:
                line = " ".join(edu_tokens)
                f.write("%s\n" % line)

        with open(os.path.join(output_dir, "%s_%s.edus.postags" % (output_filename_prefix, id)), "w") as f:
            for edu_postags in edus_postags:
                line = " ".join(edu_postags)
                f.write("%s\n" % line)

        with open(os.path.join(output_dir, "%s_%s.edus.arcs" % (output_filename_prefix, id)), "w") as f:
            for edu_arcs in edus_arcs:
                line = " ".join(edu_arcs)
                f.write("%s\n" % line)

        #####################
        # /Dialogue
        #####################

    utils.writelog("Processed %d dialogues" % n_dialogues)


def prevent_sentence_boundary_detection(doc):
    """
    Parameters
    ----------
    doc: spacy.Doc

    Returns
    -------
    spacy.Doc
    """
    for token in doc:
        token.is_sent_start = False
    return doc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output_filename_prefix", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

