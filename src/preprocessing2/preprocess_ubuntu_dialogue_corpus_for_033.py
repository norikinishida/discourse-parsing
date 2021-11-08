from collections import OrderedDict
import os

import numpy as np
from transformers import AutoTokenizer
import pyprind

import utils
import treetk

from berttokenizerwrapper import BertTokenizerWrapper
from preprocess_speakers import SPEAKERS, rename_speaker_names


def main():
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    utils.mkdir(config["caches-033"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper)

    path_output = os.path.join(config["caches-033"], "ubuntu-dialogue-corpus.bert-base-cased.npy")
    np.save(path_output, dataset)


def preprocess(tokenizer_wrapper):
    """
    Parameters
    ----------
    tokenizer_wrapper: BertTokenizerWrapper

    Returns
    -------
    numpy.ndarray(shape=(dataset_size,), dtype="O")
    """

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    # Reading
    dataset = []

    path_root = os.path.join(config["data"], "ubuntu-dialogue-corpus-compiled")

    filenames = os.listdir(path_root)
    filenames = [n for n in filenames if n.endswith(".json")]
    filenames.sort() # Important

    for filename in pyprind.prog_bar(filenames):
        dictionary = utils.read_json(os.path.join(path_root, filename))

        kargs = OrderedDict()

        # File ID
        kargs["id"] = filename.replace(".json", "")

        # Mapping from a speaker name to a speaker ID
        speaker_name_to_id = {}
        for edu_info in dictionary["edus"]:
            speaker = edu_info["speaker"]
            if not speaker in speaker_name_to_id:
                speaker_name_to_id[speaker] = len(speaker_name_to_id)
        speaker_name_to_id_uncased = {n.lower(): id for n, id in speaker_name_to_id.items()}

        # EDUs with speakers
        edus = []
        for edu_info in dictionary["edus"]:
            tokens = edu_info["tokens"].split()
            tokens = rename_speaker_names(tokens=tokens, speaker_name_to_id_uncased=speaker_name_to_id_uncased)
            speaker_name = SPEAKERS[speaker_name_to_id[edu_info["speaker"]]]
            prefix = [speaker_name, ":"]
            edus.append(prefix + tokens) # NOTE: Each EDU contains the speaker name and colon as prefix
        edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # EDUs (POS tags)
        if "postags" in dictionary["edus"][0]:
            edus_postag = [["NNP", ":"] + edu_info["postags"].split() for edu_info in dictionary["edus"]]
            edus_postag = [["<root>"]] + edus_postag
            kargs["edus_postag"] = edus_postag

        # EDUs (dependency relations)
        if "arcs" in dictionary["edus"][0]:
            edus_deprel = [["ROOT", "punct"] + [l for h,d,l in treetk.hyphens2arcs(edu_info["arcs"].split())] for edu_info in dictionary["edus"]]
            edus_deprel = [["<root>"]] + edus_deprel
            kargs["edus_deprel"] = edus_deprel

        # EDUs (head)
        if "head" in dictionary["edus"][0]:
            edus_head = [2 + int(edu_info["head"]) for edu_info in dictionary["edus"]] # NOTE: shifted by +2
            edus_head = [0] + edus_head
            kargs["edus_head"] = edus_head
            use_head = True
        else:
            use_head = False

        # Sentence boundaries
        sentence_boundaries = [tuple([int(x) for x in line.split()]) for line in dictionary["sentence_boundaries"]]
        kargs["sentence_boundaries"] = sentence_boundaries

        # Paragraph boundaries
        if "paragraph_boundaries" in dictionary:
            paragraph_boundaries = [tuple([int(x) for x in line.split()]) for line in dictionary["paragraph_boundaries"]]
        else:
            paragraph_boundaries = [(0, len(sentence_boundaries) - 1)]
        kargs["paragraph_boundaries"] = paragraph_boundaries

        # BERT inputs
        segments, segments_id, segments_mask, edu_begin_indices, edu_end_indices, edu_head_indices \
                = tokenizer_wrapper.tokenize_and_split(edus=edus,
                                                       edus_head=edus_head if use_head else [0] * len(edus),
                                                       sentence_boundaries=sentence_boundaries)
        kargs["segments"] = segments
        kargs["segments_id"] = segments_id
        kargs["segments_mask"] = segments_mask
        kargs["edu_begin_indices"] = edu_begin_indices
        kargs["edu_end_indices"] = edu_end_indices
        if use_head:
            kargs["edu_head_indices"] = edu_head_indices

        data = utils.DataInstance(**kargs)
        dataset.append(data)

    dataset = np.asarray(dataset, dtype="O")

    n_docs = len(dataset)
    n_paras = 0
    for data in dataset:
        n_paras += len(data.paragraph_boundaries)
    n_sents = 0
    for data in dataset:
        n_sents += len(data.sentence_boundaries)
    n_edus = 0
    for data in dataset:
        n_edus += len(data.edus[1:]) # Exclude the ROOT
    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)

    return dataset


if __name__ == "__main__":
    main()

