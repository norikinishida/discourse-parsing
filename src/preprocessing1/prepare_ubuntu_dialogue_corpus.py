import argparse
import os

import pyprind
import spacy

import utils


def main(args):
    input_dir = args.input
    output_dir = args.output

    utils.mkdir(output_dir)

    nlp_no_ssplit = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp_no_ssplit.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

    filenames = os.listdir(input_dir)
    filenames = [n for n in filenames if n.endswith(".tsv")]
    filenames.sort()
    n_dialogues = len(filenames)

    for filename in pyprind.prog_bar(filenames):
        lines = utils.read_lines(os.path.join(input_dir, filename),
                                 process=lambda line: line.split("\t"))
        raw_edus = []
        speakers = []
        listeners = []
        for items in lines:
            assert len(items) == 4
            # if len(items) != 4:
            #     print(items)
            #     sys.exit()
            timestamp, speaker, listener, utterance = items
            raw_edus.append(" ".join(utterance.strip().split()))
            speakers.append(speaker)
            listeners.append(listener)

        with open(os.path.join(output_dir, filename.replace(".tsv", ".speakers")), "w") as f:
            for speaker in speakers:
                f.write("%s\n" % speaker)

        with open(os.path.join(output_dir, filename.replace(".tsv", ".listeners")), "w") as f:
            for listener in listeners:
                f.write("%s\n" % listener)

        with open(os.path.join(output_dir, filename.replace(".tsv", ".sentence_boundaries")), "w") as f:
            begin_i = 0
            end_i = len(raw_edus) - 1
            f.write("%d-%d\n" % (begin_i, end_i))

        with open(os.path.join(output_dir, filename.replace(".tsv", ".paragraph_boundaries")), "w") as f:
            f.write("0-0\n")

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

        with open(os.path.join(output_dir, filename.replace(".tsv", ".edus.tokens")), "w") as f:
            for edu_tokens in edus_tokens:
                edu_tokens = " ".join(edu_tokens)
                f.write("%s\n" % edu_tokens)

        with open(os.path.join(output_dir, filename.replace(".tsv", ".edus.postags")), "w") as f:
            for edu_postags in edus_postags:
                edu_postags = " ".join(edu_postags)
                f.write("%s\n" % edu_postags)

        with open(os.path.join(output_dir, filename.replace(".tsv", ".edus.arcs")), "w") as f:
            for edu_arcs in edus_arcs:
                edu_arcs = " ".join(edu_arcs)
                f.write("%s\n" % edu_arcs)

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
    args = parser.parse_args()
    main(args=args)

