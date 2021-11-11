import argparse
from collections import defaultdict
import os
import re

import pyprind
import spacy
import xmltodict

import utils


def main(args):
    input_dir = args.input
    output_dir = args.output

    utils.mkdir(output_dir)

    nlp_no_ssplit = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp_no_ssplit.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

    dirs = os.listdir(input_dir)
    dirs.sort()
    dialogues = None
    count_games = 0
    for directory in pyprind.prog_bar(dirs):
        directory = os.path.join(os.path.join(input_dir, directory), "discourse/GOLD")
        if os.path.exists(directory):
            count_games += 1
            filenames = os.listdir(directory)
            filenames = [n for n in filenames if re.match("\S*.ac", n)]
            filenames.sort()
            for filename in filenames:
                id = filename[:filename.find("_")]
                dialogues = read_data(id=id,
                                      filename_prefix=os.path.join(directory, filename[:filename.index(".")]),
                                      dialogues=dialogues)
    count_dialogues = len(dialogues)

    id_to_count = defaultdict(int)
    for dialogue in pyprind.prog_bar(dialogues):
        ####################
        # Dialogue
        ####################

        dialogue_cleaned = process_dialogue(dialogue)

        raw_edus = []
        speakers = []
        disc_arcs = []
        for item in dialogue_cleaned["edus"]:
            raw_edus.append(" ".join(item["text"].strip().split()))
            speakers.append(item["speaker"])
        for item in dialogue_cleaned["relations"]:
            head = item["x"] + 1
            dep = item["y"] + 1
            rel = item["type"]
            disc_arcs.append((head, dep, rel))
        disc_arcs.append((0, 1, "<root>"))

        # Add a relation from the Root node to each EDU without an incoming edge (head)
        deps_with_head = set([d for h, d, r in disc_arcs])
        deps_all = set([d for d in range(1, len(raw_edus)+1)])
        deps_without_head = deps_all.difference(deps_with_head)
        for dep in deps_without_head:
            disc_arcs.append((0, dep, "<root>"))

        id = dialogue_cleaned["id"]
        dial_i = id_to_count[id]
        id_to_count[id] += 1

        with open(os.path.join(output_dir, "%s_%02d.speakers" % (id, dial_i)), "w") as f:
            for speaker in speakers:
                f.write("%s\n" % speaker)

        with open(os.path.join(output_dir, "%s_%02d.sentence_boundaries" % (id, dial_i)), "w") as f:
            begin_i = 0
            end_i = len(raw_edus) - 1
            f.write("%d-%d\n" % (begin_i, end_i))

        with open(os.path.join(output_dir, "%s_%02d.paragraph_boundaries" % (id, dial_i)), "w") as f:
            f.write("0-0\n")

        with open(os.path.join(output_dir, "%s_%02d.arcs" % (id, dial_i)), "w") as f:
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

        with open(os.path.join(output_dir, "%s_%02d.edus.tokens" % (id, dial_i)), "w") as f:
            for edu_tokens in edus_tokens:
                line = " ".join(edu_tokens)
                f.write("%s\n" % line)

        with open(os.path.join(output_dir, "%s_%02d.edus.postags" % (id, dial_i)), "w") as f:
            for edu_postags in edus_postags:
                line = " ".join(edu_postags)
                f.write("%s\n" % line)

        with open(os.path.join(output_dir, "%s_%02d.edus.arcs" % (id, dial_i)), "w") as f:
            for edu_arcs in edus_arcs:
                line = " ".join(edu_arcs)
                f.write("%s\n" % line)

        ####################
        # /Dialogue
        ####################

    utils.writelog("Processed %d games" % count_games)
    utils.writelog("Processed %d dialogues" % count_dialogues)


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


def read_data(id, filename_prefix, dialogues=None):
    """
    Parameters
    ----------
    id: str
    filename_prefix: str
    dialogues: list[dict], default None

    Returns
    -------
    list[dict]
    """
    if dialogues is None:
        dialogues = []

    # Annotation
    with open("%s.aa" % filename_prefix) as f:
        annotations = xmltodict.parse(''.join(f.readlines()))["annotations"]
    units = annotations["unit"]
    if not 'relation' in annotations:
        relations = []
    else:
        relations = annotations["relation"]
    schema = annotations["schema"] if 'schema' in annotations else []

    # Text
    with open("%s.ac" % filename_prefix) as f:
        discourse = f.readline()
        for i in range(len(discourse)):
            if ord(discourse[i]) >= 128:
                discourse = discourse[:i] + " " + discourse[i+1:]

    edus, buf_dialogues = {}, {}
    for item in units:
        # ID
        _id = item["@id"]
        # Span
        start = int(item["positioning"]["start"]["singlePosition"]["@index"])
        end = int(item["positioning"]["end"]["singlePosition"]["@index"])
        # Type
        _type = item["characterisation"]["type"]

        if _type in ["Turn", "NonplayerTurn"]:
            continue
        elif _type == "Dialogue":
            buf_dialogues[_id] = {
                "start": start,
                "end": end,
                "edus": {},
                "cdus": {},
                "relations": []
            }
        else:
            edus[_id] = {
                "id": _id,
                "type": _type,
                "text": discourse[start:end],
                "start": start,
                "end": end
            }

    belong_to = {}
    for id_edu in edus:
        edu = edus[id_edu]
        found = False
        for id_dialogue in buf_dialogues:
            dialog = buf_dialogues[id_dialogue]
            if dialog["start"] <= edu["start"] and dialog["end"] >= edu["end"]:
                found = True
                dialog["edus"][id_edu] = edu
                belong_to[id_edu] = id_dialogue
                break
        if not found:
            raise Warning("Dialogue not found")

    if type(schema) != list:
        schema = [schema]
    for item in schema:
        _id = item["@id"]
        _type = item["characterisation"]["type"]

        if item["positioning"] == None:
            continue

        cdu = []
        if "embedded-unit" in item["positioning"]:
            if type(item["positioning"]["embedded-unit"]) == list:
                cdu = [unit["@id"] for unit in item["positioning"]["embedded-unit"]]
            else:
                cdu = [item["positioning"]["embedded-unit"]["@id"]]
            for edu in cdu:
                if not edu in edus:
                    cdu.remove(edu)
        if "embedded-schema" in item["positioning"]:
            if type(item["positioning"]["embedded-schema"]) == list:
                cdu += [unit["@id"] for unit in item["positioning"]["embedded-schema"]]
            else:
                cdu += [item["positioning"]["embedded-schema"]["@id"]]
        belong_to[_id] = belong_to[cdu[0]]
        buf_dialogues[belong_to[_id]]["cdus"][_id] = cdu

    if type(relations) != list:
        relations = [relations]
    for item in relations:
        _id = item["@id"]
        x = item["positioning"]["term"][0]["@id"]
        y = item["positioning"]["term"][1]["@id"]
        _type = item["characterisation"]["type"]

        buf_dialogues[belong_to[x]]["relations"].append({
            "type": _type,
            "x": x,
            "y": y
        })

    for _id in buf_dialogues:
        buf_dialogues[_id]["id"] = id
        dialogues.append(buf_dialogues[_id])

    return dialogues


def process_dialogue(dialogue):
    """
    Parameters
    ----------
    dialogue: dict

    Returns
    -------
    dict
    """
    has_incoming = {}

    for relation in dialogue["relations"]:
        has_incoming[relation["y"]] = True

    for _id in dialogue["edus"]:
        edu = dialogue["edus"][_id]
        if edu["type"] == "paragraph":
            continue

        for _id_para in dialogue["edus"]:
            def parse_speaker(text):
                return (text.split())[2]

            para = dialogue["edus"][_id_para]
            if para["type"] != "paragraph": continue
            if para["start"] <= edu["start"] and para["end"] >= edu["end"]:
                edu["speaker"] = parse_speaker(para["text"])

    idx = {}
    dialogue["edu_list"] = []
    for _id in dialogue["edus"]:
        if dialogue["edus"][_id]["type"] != "paragraph":
            dialogue["edu_list"].append(dialogue["edus"][_id])
    dialogue["edu_list"] = sorted(dialogue["edu_list"], key=lambda edu: edu["start"])
    for i in range(len(dialogue["edu_list"])):
        edu = dialogue["edu_list"][i]
        idx[edu["id"]] = i

    # for i, edu in enumerate(dialogue["edu_list"]):
    #     utils.writelog("%d %s : %s" % (i, edu["speaker"], edu["text"]))
    # utils.writelog("===")

    for relation in dialogue["relations"]:
        def get_head(x):
            if x in dialogue["edus"]: return x
            else:
                for du in dialogue["cdus"][x]:
                    if not du in has_incoming: return get_head(du)
                raise Warning("Can't find the recursive head")

        relation["x"] = idx[get_head(relation["x"])]
        relation["y"] = idx[get_head(relation["y"])]

    dialogue_cleaned = {
        "id": dialogue["id"],
        "edus": [],
        "relations": []
    }

    for edu in dialogue["edu_list"]:
        dialogue_cleaned["edus"].append({
            "speaker": edu["speaker"],
            "text": edu["text"]
        })
    for relation in dialogue["relations"]:
        dialogue_cleaned["relations"].append({
            "type": relation["type"],
            "x": relation["x"],
            "y": relation["y"]
        })

    return dialogue_cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)


