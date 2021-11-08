import argparse

import utils

def main(args):
    path = args.path
    assert path.endswith(".doc.tokens")

    # Gold EDUs
    edus = utils.read_lines(path.replace(".doc.tokens", ".edus.tokens"), process=lambda line: line.split()) # List[List[str]]

    # Paragraphs
    lines = utils.read_lines(path, process=lambda line: line.split())
    paras = [] # List[List[List[str]]]
    para = [lines[0]]
    for i in range(1, len(lines)):
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
    cnt = len(para_end_positions)

    # Filtered paragraph-ending positions
    para_end_positions = list(set(edu_end_positions) & set(para_end_positions))
    para_end_positions.sort()

    # Re-make paragraphs based on the filtered paragraph-ending positions
    new_paras = []
    new_para = []
    tok_i = 0
    pos_i = 0
    for para in paras:
        new_para.extend(para)
        length = len([token for tokens in para for token in tokens])
        tok_i += length
        if tok_i - 1 == para_end_positions[pos_i]:
            new_paras.append(new_para)
            new_para = []
            pos_i += 1
    assert pos_i == len(para_end_positions) == len(new_paras)
    paras = new_paras

    # Write
    with open(path + ".fixed", "w") as f:
        for para in new_paras:
            for tokens in para:
                tokens = " ".join(tokens)
                f.write("%s\n" % tokens)
            f.write("\n")

    diff = cnt - len(para_end_positions)
    print("Removed %d paragraph boundaries." % diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)

