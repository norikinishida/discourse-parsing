import argparse
import os

import spacy
import pyprind

import utils


def main(args):
    path_in_root = args.input
    path_out_root = args.output

    utils.mkdir(path_out_root)

    # Check
    filenames_pdf = get_filenames(os.path.join(path_in_root, "pdf_json"))
    filenames_pmc = get_filenames(os.path.join(path_in_root, "pmc_json"))
    assert len(set(filenames_pdf) & set(filenames_pmc)) == 0

    process(path_in_root=os.path.join(path_in_root, "pdf_json"),
            path_out_root=path_out_root)
    process(path_in_root=os.path.join(path_in_root, "pmc_json"),
            path_out_root=path_out_root)


def get_filenames(path_dir):
    """
    Parameters
    ----------
    path_dir: str

    Returns
    -------
    list[str]
    """
    filenames = os.listdir(path_dir)
    filenames = [n for n in filenames if n.endswith(".json")]
    filenames.sort()
    return filenames


def process(path_in_root, path_out_root):
    """
    Parameters
    ----------
    path_in_root: str
    path_out_root: str
    """
    filenames = get_filenames(path_in_root)
    n_files = len(filenames)

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    cnt = 0
    for filename in pyprind.prog_bar(filenames):
        data = utils.read_json(os.path.join(path_in_root, filename))

        if not "abstract" in data:
            print("Skipped %s" % filename)
            continue

        paras = data["abstract"] # list of paragraphs

        if len(paras) != 0:
            with open(os.path.join(path_out_root, filename.replace(".json", ".doc.tokens")), "w") as f:
                for para in paras:
                    assert para["section"] == "Abstract"
                    doc = nlp(para["text"])
                    tokens = [token.text for token in doc]
                    assert len(tokens) > 0
                    tokens = " ".join(tokens)
                    f.write("%s\n" % tokens)
                    f.write("\n") # Paragraphs are separated by empty lines
            cnt += 1

    print("Processed %d/%d files." % (cnt, n_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)


