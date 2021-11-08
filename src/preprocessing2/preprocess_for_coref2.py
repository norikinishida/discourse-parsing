from collections import OrderedDict
import os

import numpy as np
from transformers import AutoTokenizer
import pyprind

import utils

from berttokenizerwrapper import BertTokenizerWrapper


def main():
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    utils.mkdir(config["caches-040"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    for split in ["train", "dev", "test"]:
        dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper, split=split)

        path_output = os.path.join(config["caches-040"], "ontonotes.%s.english.bert-base-cased.npy" % split)
        np.save(path_output, dataset)


def preprocess(tokenizer_wrapper, split):
    """
    Parameters
    ----------
    tokenizer_wrapper: BertTokenizerWrapper
    split: str

    Returns
    -------
    numpy.ndarray(shape=(dataset_size,), dtype="O")
    """
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_root = os.path.join(config["caches-040"], "ontonotes.%s.english" % split)

    # Reading
    dataset = []

    filenames = os.listdir(path_root)
    filenames = [n for n in filenames if n.endswith(".txt")]
    filenames.sort() # Important

    for filename in pyprind.prog_bar(filenames):
        kargs = OrderedDict()

        # File ID
        kargs["id"] = filename.replace(".txt", "")

        # EDUs
        edus = utils.read_lines(os.path.join(path_root, filename), process=lambda line: line.split())
        edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # Sentence boundaries
        sentence_boundaries = [(i, i) for i in range(len(edus) - 1)]
        kargs["sentence_boundaries"] = sentence_boundaries

        # Paragraph boundaries
        paragraph_boundaries = [(0, len(sentence_boundaries) - 1)]
        kargs["paragraph_boundaries"] = paragraph_boundaries

        # BERT inputs
        segments, segments_id, segments_mask, edu_begin_indices, edu_end_indices, edu_head_indices \
                = tokenizer_wrapper.tokenize_and_split(edus=edus,
                                                       edus_head=[0] * len(edus),
                                                       sentence_boundaries=sentence_boundaries)
        kargs["segments"] = segments
        kargs["segments_id"] = segments_id
        kargs["segments_mask"] = segments_mask
        kargs["edu_begin_indices"] = edu_begin_indices
        kargs["edu_end_indices"] = edu_end_indices
        # kargs["edu_head_indices"] = edu_head_indices

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

