import argparse
import json
import io
import os

import spacy
import pyprind

import utils


def main(args):
    path_in = args.input
    path_out = args.output

    utils.mkdir(path_out)

    nlp_no_ssplit = spacy.load("en_core_web_sm", diable=["ner", "textcat"])
    nlp_no_ssplit.tokenizer = nlp_no_ssplit.tokenizer.tokens_from_list
    nlp_no_ssplit.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

    filenames = os.listdir(path_in)
    filenames = [n for n in filenames if n.endswith(".edu.txt.dep")]
    filenames.sort()

    skip_count = 0
    for filename in pyprind.prog_bar(filenames):
        edus, sents, sentence_boundaries, disc_arcs = read_data(os.path.join(path_in, filename))

        if edus is None:
            print("Skippted %s" % filename)
            skip_count += 1
            continue

        assert len(sents) == len(sentence_boundaries)

        with open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".edus.tokens")), "w") as f:
            for edu in edus:
                edu = " ".join(edu)
                f.write("%s\n" % edu)

        with open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".sentence_boundaries")), "w") as f:
            # for begin_i, end_i in sentence_boundaries:
            #     f.write("%d %d\n" % (begin_i, end_i))
            f.write("%s\n" % " ".join(["%d-%d" % (b, e) for b, e in sentence_boundaries]))

        with open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".paragraph_boundaries")), "w") as f:
            n_sents = len(sents)
            f.write("0-%d\n" % (n_sents-1))

        with open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".arcs")), "w") as f:
            disc_arcs = sorted(disc_arcs, key=lambda x: x[1])
            disc_arcs = ["%d-%d-%s" % (h,d,l) for h,d,l in disc_arcs]
            disc_arcs = " ".join(disc_arcs)
            f.write("%s\n" % disc_arcs)

        sents_postags = []
        sents_arcs = []
        for sent in sents:
            doc = nlp_no_ssplit(sent)
            sents_ = list(doc.sents)
            assert len(sents_) == 1
            sent = sents_[0]
            postags = [token.tag_ for token in sent]
            arcs = []
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
                arcs.append(syn_arc)
            assert found_root
            arcs = ["%d-%d-%s" % (h,d,l) for h,d,l in arcs]
            sents_postags.append(postags)
            sents_arcs.append(arcs)
        postags = utils.flatten_lists(sents_postags) # List[str]
        arcs = utils.flatten_lists(sents_arcs) # List[str]

        with open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".edus.postags")), "w") as fp,\
             open(os.path.join(path_out, filename.replace(".edu.txt.dep", ".edus.arcs")), "w") as fa:
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

    print("Processed %d files; %d files are skipped." % (len(filenames) - skip_count, skip_count))



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
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


def read_data(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    list[list[str]]
    list[list[str]]
    list[(int, int)]
    list[(int, int, str)]
    """
    with io.open(path, "rt", encoding="utf_8_sig") as f:
        line = f.read()
    data = json.loads(line)

    data = data["root"]

    assert data[0]["id"] == 0
    assert data[0]["parent"] == -1
    assert data[0]["text"] == "ROOT"
    assert data[0]["relation"] == "null"

    data = data[1:]

    for x in data:
        if x["relation"] == "null":
            return None, None, None, None

    for x in data:
        if "<S>" in x["text"]:
            x["is_sent_end"] = True
        else:
            x["is_sent_end"] = False
        x["text"] = x["text"].replace("<S>", "")
        x["text"] = x["text"].split()
    assert data[-1]["is_sent_end"]

    edus = [x["text"] for x in data]

    sents = []
    sentence_boundaries = []
    begin_edu_i = 0
    sent = []
    for end_edu_i, x in enumerate(data):
        sent.extend(x["text"])
        if x["is_sent_end"]:
            sents.append(sent)
            sent = []
            sentence_boundaries.append((begin_edu_i, end_edu_i))
            begin_edu_i = end_edu_i + 1

    disc_arcs = []
    for x in data:
        head = x["parent"]
        dep = x["id"]
        relation = x["relation"]
        if relation == "ROOT":
            relation = "<root>"
        disc_arcs.append((head, dep, relation))

    return edus, sents, sentence_boundaries, disc_arcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

