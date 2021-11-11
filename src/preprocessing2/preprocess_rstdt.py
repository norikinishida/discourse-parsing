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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    con_relations = []
    con_nuclearities = []
    dep_relations = []
    for split in ["train", "test"]:
        dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper, split=split, relation_level="coarse-grained", with_root=True)

        path_output = os.path.join(config["caches-dep"], "rstdt.%s.bert-base-cased.npy" % split)
        np.save(path_output, dataset)

        # Cache gold labels
        path_output = os.path.join(config["caches-dep"], "rstdt.%s.gold.labeled.nary.ctrees" % split)
        with open(path_output, "w") as f:
            for data in dataset:
                f.write("%s\n" % " ".join(data.nary_sexp))

        path_output = os.path.join(config["caches-dep"], "rstdt.%s.gold.labeled.bin.ctrees" % split)
        with open(path_output, "w") as f:
            for data in dataset:
                f.write("%s\n" % " ".join(data.bin_sexp))

        path_output = os.path.join(config["caches-dep"], "rstdt.%s.gold.arcs" % split)
        with open(path_output, "w") as f:
            for data in dataset:
                arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in data.arcs]
                f.write("%s\n" % " ".join(arcs))

        for data in dataset:
            sexp = treetk.preprocess(data.bin_sexp)
            tree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            nodes = treetk.traverse(tree, order="pre-order", include_terminal=False, acc=None)
            for node in nodes:
                relations_ = node.relation_label.split("/")
                con_relations.extend(relations_)
            con_nuclearities.extend([node.nuclearity_label for node in nodes])
            dep_relations.extend([r for (h, d, r) in data.arcs])

    con_relations = sorted(list(set(con_relations)))
    con_nuclearities = sorted(list(set(con_nuclearities)))
    dep_relations = sorted(list(set(dep_relations)))

    utils.write_vocab(os.path.join(config["caches-dep"], "rstdt.constituency_relations.vocab.txt"),
                      con_relations,
                      write_frequency=False)
    utils.write_vocab(os.path.join(config["caches-dep"], "rstdt.constituency_nuclearities.vocab.txt"),
                      con_nuclearities,
                      write_frequency=False)
    utils.write_vocab(os.path.join(config["caches-dep"], "rstdt.dependency_relations.vocab.txt"),
                      dep_relations,
                      write_frequency=False)


def preprocess(tokenizer_wrapper, split, relation_level, with_root=False):
    """
    Parameters
    ----------
    tokenizer_wrapper: BertTokenizerWrapper
    split: str
    relation_level: str
    with_root: bool, default False

    Returns
    -------
    numpy.ndarray(shape=(dataset_size), dtype="O")
    """
    assert split in ["train", "test"]
    assert relation_level in ["coarse-grained", "fine-grained"]

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    path_root = os.path.join(config["data"], "rstdt-compiled", "wsj", split)

    if relation_level == "coarse-grained":
        relation_mapper = treetk.rstdt.RelationMapper()

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
            edus_deprel = [[l for h,d,l in treetk.hyphens2arcs(edu_info["arcs"].split())] for edu_info in dictionary["edus"]]
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

        # Constituent tree
        nary_sexp = dictionary["nary_sexp"].split()
        bin_sexp = dictionary["bin_sexp"].split()
        if relation_level == "coarse-grained":
            nary_tree = treetk.rstdt.postprocess(treetk.sexp2tree(nary_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            bin_tree = treetk.rstdt.postprocess(treetk.sexp2tree(bin_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            nary_tree = treetk.rstdt.map_relations(nary_tree, mode="f2c")
            bin_tree = treetk.rstdt.map_relations(bin_tree, mode="f2c")
            nary_sexp = treetk.tree2sexp(nary_tree)
            bin_sexp = treetk.tree2sexp(bin_tree)
        kargs["nary_sexp"] = nary_sexp
        kargs["bin_sexp"] = bin_sexp

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
        if with_root:
            n_edus += len(data.edus[1:]) # Exclude the ROOT
        else:
            n_edus += len(data.edus)
    utils.writelog("split: %s" % split)
    utils.writelog("# of documents: %d" % n_docs)
    utils.writelog("# of paragraphs: %d" % n_paras)
    utils.writelog("# of sentences: %d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs): %d" % n_edus)

    return dataset


if __name__ == "__main__":
    main()



