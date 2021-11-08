import argparse
import os

import pyprind

import utils


def main(args):
    input_dir = args.input
    output_dir = args.output

    utils.mkdir(output_dir)

    filenames = os.listdir(input_dir)
    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        dictionary = {}

        # Path
        path_edus_token = os.path.join(input_dir, filename)
        path_edus_postag = os.path.join(input_dir, filename.replace(".edus.tokens", ".edus.postags"))
        path_edus_arc = os.path.join(input_dir, filename.replace(".edus.tokens", ".edus.arcs"))
        path_edus_head = os.path.join(input_dir, filename.replace(".edus.tokens", ".edus.heads"))
        path_speakers = os.path.join(input_dir, filename.replace(".edus.tokens", ".speakers"))
        path_listeners = os.path.join(input_dir, filename.replace(".edus.tokens", ".listeners"))
        path_sentence_boundaries = os.path.join(input_dir, filename.replace(".edus.tokens", ".sentence_boundaries"))
        path_paragraph_boundaries = os.path.join(input_dir, filename.replace(".edus.tokens", ".paragraph_boundaries"))
        path_json = os.path.join(output_dir, filename.replace(".edus.tokens", ".json"))

        # EDUs
        edus = []
        edus_token = utils.read_lines(path_edus_token)
        edus_postag = utils.read_lines(path_edus_postag)
        edus_arc = utils.read_lines(path_edus_arc)
        edus_head = utils.read_lines(path_edus_head)
        speakers = utils.read_lines(path_speakers)
        listeners = utils.read_lines(path_listeners)
        assert len(edus_token) == len(edus_postag) == len(edus_arc) == len(edus_head) == len(speakers) == len(listeners)
        for edu_token, edu_postag, edu_arc, edu_head, speaker, listener in \
                zip(edus_token, edus_postag, edus_arc, edus_head, speakers, listeners):
            edu = {
                "tokens": edu_token,
                "postags": edu_postag,
                "arcs": edu_arc,
                "head": edu_head,
                "speaker": speaker,
                "listener": listener,
                }
            edus.append(edu)
        dictionary["edus"] = edus

        # Sentence boundaries
        sentence_boundaries = utils.read_lines(path_sentence_boundaries)
        dictionary["sentence_boundaries"] = sentence_boundaries

        # Paragraph boundaries
        paragraph_boundaries = utils.read_lines(path_paragraph_boundaries)
        dictionary["paragraph_boundaries"] = paragraph_boundaries

        # Save
        utils.write_json(path_json, dictionary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args=args)

