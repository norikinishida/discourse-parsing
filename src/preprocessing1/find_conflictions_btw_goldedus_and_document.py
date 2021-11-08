import argparse
import os

import utils


def main(args):
    path = args.path
    check_token = args.check_token
    check_char = args.check_char
    check_boundary = args.check_boundary

    process(path=path,
            check_token=check_token,
            check_char=check_char,
            check_boundary=check_boundary)


def process(path, check_token, check_char, check_boundary):
    """
    Parameters
    ----------
    path: str
    check_token: bool
    check_char: bool
    check_boundary: bool
    """

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    confliction = False
    for filename in filenames:

        # Gold EDUs
        lines_e = utils.read_lines(os.path.join(path, filename))
        text_e =  " ".join(lines_e) # str
        edus = [l.split() for l in lines_e] # List[List[str]]

        # Paragraphs
        lines_d = utils.read_lines(os.path.join(path, filename.replace(".edus.tokens", ".doc.tokens")))
        text_d = " ".join(lines_d) # str
        paras = [] # List[List[str]]
        para = [lines_d[0].split()]
        for i in range(1, len(lines_d)):
            line = lines_d[i].split()
            if len(line) == 0 and len(para) == 0:
                continue
            elif len(line) == 0 and len(para) != 0:
                paras.append(para)
                para = []
            else:
                para.append(line)
        if len(para) != 0:
            paras.append(para)

        # Test
        if check_token and not test_tokenlevel_confliction(text_e, text_d):
            print("Found token-level confliction: %s" % os.path.join(path, filename))
            confliction = True

        if check_char and not test_charlevel_confliction(text_e, text_d):
            print("Found char-level confliction: %s" % os.path.join(path, filename))
            confliction = True

        if check_boundary and not test_boundary_confliction(edus, paras):
            print("Found paragraph-boundary confliction: %s" % os.path.join(path, filename))
            confliction = True

    if not confliction:
        print("Found NO confliction: OK")


def test_tokenlevel_confliction(text_e, text_d):
    """
    Parameters
    ----------
    text_e: str
    text_d: str

    Returns
    -------
    bool
    """
    tokens_e = text_e.split()
    tokens_d = text_d.split()
    if len(tokens_e) != len(tokens_d):
        return False
    for i in range(len(tokens_e)):
        if tokens_e[i] != tokens_d[i]:
            return False
    return True


def test_charlevel_confliction(text_e, text_d):
    """
    Parameters
    ----------
    text_e: str
    text_d: str

    Returns
    -------
    bool
    """
    chars_e = "".join(text_e.split())
    chars_d = "".join(text_d.split())
    if len(chars_e) != len(chars_d):
        return False
    for i in range(len(chars_e)):
        if chars_e[i] != chars_d[i]:
            return False
    return True


def test_boundary_confliction(edus, paras):
    """
    Parameters
    ----------
    edus: list[list[str]]
    paras: list[list[str]]

    Returns
    -------
    bool
    """
    # Ending positions of gold EDUs
    edu_end_positions = []
    tok_i = 0
    for edu in edus:
        length = len(edu)
        edu_end_positions.append(tok_i + length - 1)
        tok_i += length

    # Ending positions of paragraphs
    para_end_positions = []
    tok_i = 0
    for para in paras:
        length = len([token for tokens in para for token in tokens])
        para_end_positions.append(tok_i + length - 1)
        tok_i += length

    # All the ending positions of paragraphs must match with those of gold EDUs
    return set(para_end_positions) == set(edu_end_positions) & set(para_end_positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--check_token", action="store_true")
    parser.add_argument("--check_char", action="store_true")
    parser.add_argument("--check_boundary", action="store_true")
    args = parser.parse_args()
    main(args)

