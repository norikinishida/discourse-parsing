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

    utils.mkdir(config["caches-dep"])

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    relations = []
    for split, sub_dir in [("train", ""),
                           ("dev", "gold"),
                           ("dev", "second_annotate"),
                           ("test", "gold"),
                           ("test", "second_annotate")]:
        dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper, split=split, sub_dir=sub_dir, relation_level="coarse-grained")

        path_output = os.path.join(config["caches-dep"], "scidtb.%s-%s.scibert_scivocab_uncased.npy" % (split, sub_dir))
        np.save(path_output, dataset)

        # Cache gold labels
        path_output = os.path.join(config["caches-dep"], "scidtb.%s-%s.gold.arcs" % (split, sub_dir))
        with open(path_output, "w") as f:
            for data in dataset:
                arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in data.arcs]
                f.write("%s\n" % " ".join(arcs))

        for data in dataset:
            relations.extend([l for h, d, l in data.arcs])

    relations = sorted(list(set(relations)))
    utils.write_vocab(os.path.join(config["caches-dep"], "scidtb.relations.vocab.txt"),
                      relations,
                      write_frequency=False)


def preprocess(tokenizer_wrapper, split, sub_dir, relation_level):
    """
    Parameters
    ----------
    tokenizer_wrapper: BertTokenizerWrapper
    split: str
    sub_dir: str
    relation_level: str

    Returns
    -------
    numpy.ndarray(shape=(dataset_size,), dtype="O")
    """
    assert split in ["train", "dev", "test"]
    assert sub_dir in ["", "gold", "second_annotate"]
    assert relation_level in ["coarse-grained", "fine-grained"] # NOTE

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_root = os.path.join(config["data"], "scidtb-compiled", split, sub_dir)

    if relation_level == "coarse-grained":
        relation_mapper = treetk.rstdt.RelationMapper(corpus_name="scidtb")

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
        edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # EDUs (POS tags)
        if "postags" in dictionary["edus"][0]:
            edus_postag = [edu_info["postags"].split() for edu_info in dictionary["edus"]]
            edus_postag = [["<root>"]] + edus_postag
            kargs["edus_postag"] = edus_postag

        # EDUs (dependency relations)
        if "arcs" in dictionary["edus"][0]:
            edus_deprel = [[l for h,d,l in treetk.hyphens2arcs(edu_info["arcs"].split())] for edu_info in dictionary["edus"]]
            edus_deprel = [["<root>"]] + edus_deprel
            kargs["edus_deprel"] = edus_deprel

        # EDUs (head)
        if "head" in dictionary["edus"][0]:
            edus_head = [int(edu_info["head"]) for edu_info in dictionary["edus"]]
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

        # Dependency tree
        hyphens = dictionary["arcs"].split()
        arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
        if relation_level == "coarse-grained":
            arcs = [(h,d,relation_mapper.f2c(l)) for h,d,l in arcs]
        kargs["arcs"] = arcs

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
    utils.writelog("split=%s; sub_dir=%s" % (split, sub_dir))
    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)

    return dataset


if __name__ == "__main__":
    main()
