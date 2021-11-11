from collections import OrderedDict
import os

import numpy as np
from transformers import AutoTokenizer
import pyprind

import utils
import treetk

from berttokenizerwrapper import BertTokenizerWrapper


def main():
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    utils.mkdir(config["caches-033"])

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", additional_special_tokens=["<root>"])
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper, with_root=True)

    path_output = os.path.join(config["caches-033"], "cord19-abst.scibert_scivocab_uncased.npy")
    # path_output = os.path.join(config["caches-033"], "cord19-abst.BiomedNLP-PubMedBERT-base-uncased-abstract.npy")
    np.save(path_output, dataset)


def preprocess(tokenizer_wrapper, with_root=False):
    """
    Parameters
    ----------
    tokenizer_wrapper: BertTokenizerWrapper
    with_root: bool

    Returns
    -------
    numpy.ndarray(shape=(dataset_size,), dtype="O")
    """
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_root = os.path.join(config["data"], "cord19-abst-compiled")

    # Reading
    dataset = []

    filenames = os.listdir(path_root)
    filenames = [n for n in filenames if n.endswith(".json")]
    filenames.sort() # Important

    for filename in pyprind.prog_bar(filenames):
        dictionary = utils.read_json(os.path.join(path_root, filename))

        kargs = OrderedDict()

        # File ID
        kargs["id"] = filename.replace(".json", "")

        # EDUs
        edus = [edu_info["tokens"].split() for edu_info in dictionary["edus"]]
        if with_root:
            edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # EDUs (POS tags)
        if "postags" in dictionary["edus"][0]:
            edus_postag = [edu_info["postags"].split() for edu_info in dictionary["edus"]]
            if with_root:
                edus_postag = [["<root>"]] + edus_postag
            kargs["edus_postag"] = edus_postag

        # EDUs (dependency relations)
        if "arcs" in dictionary["edus"][0]:
            edus_deprel = [[l for h,d,l in treetk.hyphens2arcs(edu_info["arcs"].split())] for edu_info in dictionary["edus_arc"]]
            if with_root:
                edus_deprel = [["<root>"]] + edus_deprel
            kargs["edus_deprel"] = edus_deprel

        # EDUs (head)
        if "head" in dictionary["edus"][0]:
            edus_head = [int(edu_info["head"]) for edu_info in dictionary["edus"]]
            if with_root:
                edus_head = [0] + edus_head
            kargs["edus_head"] = edus_head
            use_head = True
        else:
            use_head = False

        # Sentence boundaries
        sentence_boundaries = [tuple([int(x) for x in span.split("-")]) for span in dictionary["sentence_boundaries"].split()]
        kargs["sentence_boundaries"] = sentence_boundaries

        # Paragraph boundaries
        if "paragraph_boundaries" in dictionary:
            paragraph_boundaries = [tuple([int(x) for x in span.split("-")]) for span in dictionary["paragraph_boundaries"].split()]
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
        if with_root:
            n_edus += len(data.edus[1:]) # Exclude the ROOT
        else:
            n_edus += len(data.edus)
    utils.writelog("# of documents: %d" % n_docs)
    utils.writelog("# of paragraphs: %d" % n_paras)
    utils.writelog("# of sentences: %d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs): %d" % n_edus)

    return dataset


if __name__ == "__main__":
    main()

