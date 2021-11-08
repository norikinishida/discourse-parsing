import argparse
import os

import spacy
import pyprind

import utils


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


def main(args):
    path = args.path
    no_ssplit = args.no_ssplit
    with_gold_edus = args.with_gold_edus

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    nlp_no_ssplit = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp_no_ssplit.tokenizer = nlp_no_ssplit.tokenizer.tokens_from_list
    nlp_no_ssplit.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".doc.tokens")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):

        if with_gold_edus:
            # Gold EDUs
            edus = utils.read_lines(os.path.join(path, filename.replace(".doc.tokens", ".edus.tokens")), process=lambda line: line.split())

            # Ending positions of gold EDUs
            edu_end_positions = []
            tok_i = 0
            for edu in edus:
                length = len(edu)
                edu_end_positions.append(tok_i + length - 1)
                tok_i += length

        # Paragraphs: List[List[List[str]]]
        lines = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split())
        paras = []
        para = []
        for i in range(len(lines)):
            line = lines[i]
            if len(line) == 0 and len(para) == 0:
                continue
            elif len(line) == 0 and len(para) != 0:
                paras.append(para)
                para = []
            else:
                para.append(line)
        if len(para) != 0:
            paras.append(para)

        # Split lines in paragraphs into sentences
        # Constraint: All the ending positions of sentences must match with those of gold EDUs

        paras_tokens = []
        paras_postags = []
        tok_i = 0
        for para in paras:
            sents_tokens = []
            sents_postags = []

            buf_tokens = []
            buf_postags = []
            # buf_arcs = []
            for line in para:
                # Split a line into sentences
                if no_ssplit:
                    doc = nlp_no_ssplit(line)
                else:
                    doc = nlp(line)
                for sent in doc.sents:
                    # Tokens and POS tags for this sentence
                    tokens = [token.text for token in sent]
                    postags = [token.tag_ for token in sent]

                    buf_tokens.extend(tokens)
                    buf_postags.extend(postags)

                    length = len(tokens)
                    tok_i += length
                    if (not with_gold_edus) or (tok_i - 1) in edu_end_positions:
                        sents_tokens.append(buf_tokens)
                        sents_postags.append(buf_postags)
                        buf_tokens = []
                        buf_postags = []

            assert len(buf_tokens) == 0
            assert len(buf_postags) == 0

            paras_tokens.append(sents_tokens)
            paras_postags.append(sents_postags)

        # Dependency parses
        paras_arcs = []
        for para in paras_tokens:
            sents_arcs = []
            for sent in para:

                doc = nlp_no_ssplit(sent)
                sents = list(doc.sents)
                assert len(sents) == 1
                sent = sents[0]

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
                    arc = (head, dep, label)
                    arcs.append(arc)
                assert found_root
                sents_arcs.append(arcs)
            paras_arcs.append(sents_arcs)

        # Paragraph boundaries
        paragraph_boundaries = []
        s_i = 0
        for sents in paras_tokens:
            begin_i = s_i
            end_i = s_i + len(sents) - 1
            paragraph_boundaries.append((begin_i, end_i))
            s_i = end_i + 1

        # Write paragraph boundaries
        with open(os.path.join(path, filename.replace(".doc.tokens", ".paragraph_boundaries")), "w") as f:
            for begin_i, end_i in paragraph_boundaries:
                f.write("%d %d\n" % (begin_i, end_i))

        # Write sentences
        with open(os.path.join(path, filename.replace(".doc.tokens", ".sents.tokens")), "w") as f:
            for sents in paras_tokens:
                for tokens in sents:
                    tokens = " ".join(tokens)
                    f.write("%s\n" % tokens)
        with open(os.path.join(path, filename.replace(".doc.tokens", ".sents.postags")), "w") as f:
            for sents in paras_postags:
                for postags in sents:
                    postags = " ".join(postags)
                    f.write("%s\n" % postags)
        with open(os.path.join(path, filename.replace(".doc.tokens", ".sents.arcs")), "w") as f:
            for sents in paras_arcs:
                for arcs in sents:
                    arcs = ["%d-%d-%s" % (h,d,l) for h,d,l in arcs]
                    arcs = " ".join(arcs)
                    f.write("%s\n" % arcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--no_ssplit", action="store_true")
    parser.add_argument("--with_gold_edus", action="store_true")
    args = parser.parse_args()
    parser.parse_args()
    main(args)
