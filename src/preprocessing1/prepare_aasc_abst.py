import argparse
import os

import spacy
import pyprind

import utils


def main(args):
    input_dir = args.input
    output_dir = args.output

    utils.mkdir(output_dir)

    filenames = os.listdir(input_dir)
    filenames = [n for n in filenames if n.endswith(".ss")]
    filenames.sort()

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    cnt = 0
    for filename in pyprind.prog_bar(filenames):
        text = extract_abstract(os.path.join(input_dir, filename))
        if text == "":
            # print("No Abstract!: %s" % filename)
            continue
        with open(os.path.join(output_dir, filename.replace(".ss", ".doc.tokens")), "w") as f:
            doc = nlp(text)
            tokens = [token.text for token in doc]
            assert len(tokens) > 0
            tokens = " ".join(tokens)
            f.write("%s\n" % tokens)
        cnt += 1

    print("Processed %d/%d files" % (cnt, len(filenames)))


def extract_abstract(path):
    """
    Parameters
    ----------
    path: str

    Returns
    -------
    str
    """
    lines = utils.read_lines(path, process=lambda line: line.split("\t"))
    for line in lines:
        assert len(line) == 3
    lines = [l[2] for l in lines if l[1] == "abstract"]
    text = " ".join(lines)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

