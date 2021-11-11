from collections import defaultdict
from collections import OrderedDict
import os

import numpy as np
from transformers import AutoTokenizer
import pyprind

import utils
import treetk

from berttokenizerwrapper import BertTokenizerWrapper
from preprocess_speakers import SPEAKERS, rename_speaker_names


STAC_TO_MOLWENI = {
    "Question-answer_pair": "QAP",
    "Comment": "Comment",
    "Acknowledgement": "Acknowledgement",
    "Continuation": "Continuation",
    "<root>": "<root>",
    "Elaboration": "Elaboration",
    "Q-Elab": "Q-Elab",
    "Contrast": "Contrast",
    "Explanation": "Explanation",
    "Clarification_question": "Clarification_question",
    "Result": "Result",
    "Correction": "Correction",
    "Parallel": "Parallel",
    "Conditional": "Conditional",
    "Alternation": "Alternation",
    "Narration":  "Narration",
    "Background": "Background",
}


def main():
    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    utils.mkdir(config["caches-033"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", additional_special_tokens=["<root>"])
    tokenizer_wrapper = BertTokenizerWrapper(tokenizer=tokenizer)

    for split in ["train", "dev", "test"]:
        dataset = preprocess(tokenizer_wrapper=tokenizer_wrapper, split=split)

        path_output = os.path.join(config["caches-033"], "mapped-stac.%s.bert-base-cased.npy" % split)
        np.save(path_output, dataset)

        # Cache gold labels
        path_output = os.path.join(config["caches-033"], "mapped-stac.%s.gold.arcs" % split)
        with open(path_output, "w") as f:
            for data in dataset:
                arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in data.arcs]
                f.write("%s\n" % " ".join(arcs))


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
    assert split in ["train", "dev", "test"]

    if split == "train":
        directories = ["socl-season1_spect", "socl-season2_spect"]
    elif split == "dev":
        directories = ["pilot_spect"]
    else:
        directories = ["test"]

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name="path")

    # Reading
    dataset = []

    for directory in directories:
        path_root = os.path.join(config["data"], "stac-compiled", directory)

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
            sentence_boundaries = [tuple([int(x) for x in span.split("-")]) for span in dictionary["sentence_boundaries"].split()]
            kargs["sentence_boundaries"] = sentence_boundaries

            # Paragraph boundaries
            if "paragraph_boundaries" in dictionary:
                paragraph_boundaries = [tuple([int(x) for x in span.split("-")]) for span in dictionary["paragraph_boundaries"].split()]
            else:
                paragraph_boundaries = [(0, len(sentence_boundaries) - 1)]
            kargs["paragraph_boundaries"] = paragraph_boundaries

            # Dependency tree
            hyphens = dictionary["arcs"].split()
            arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
            arcs = [(h, d, STAC_TO_MOLWENI[l]) for h, d, l in arcs] # NOTE
            # Transforming multi heads/roots in the training data
            if split == "train":
                # First, create a dependent-to-"heads" map
                dep2heads = defaultdict(list)
                for h, d, r in arcs:
                    if d in dep2heads:
                        dep2heads[d].append((abs(h - d), h, r))
                    else:
                        dep2heads[d] = [(abs(h - d), h, r)]
                assert len(dep2heads) == len(edu_ids) - 1
                # NOTE: Important!!!
                # Second, remove multi heads
                for d in range(1, len(edu_ids)):
                    heads = dep2heads[d]
                    if len(heads) > 1:
                        # Found multi-head; choose the closest dependency
                        heads = sorted(heads, key=lambda tpl: tpl)
                        dep2heads[d] = heads[0:1]
                # NOTE: Important!!!
                # Third, replace multi roots with Parallel relations
                found_root_child = False
                leftmost_root_child = None
                for d in range(1, len(edu_ids)):
                    heads = dep2heads[d]
                    assert len(heads) == 1
                    _, head, rel = heads[0]
                    if rel == "<root>":
                        assert head == 0
                        if not found_root_child:
                            # The leftmost root's child
                            found_root_child = True
                            leftmost_root_child = d
                        else:
                            # The other root's children connected to the leftmost root's child by Paralle relations
                            dep2heads[d] = [(abs(leftmost_root_child - d), leftmost_root_child, "Parallel")]
                # Finally, convert the map to arcs
                new_arcs = []
                for d in range(1, len(edu_ids)):
                    heads = dep2heads[d]
                    assert len(heads) == 1
                    _, head, rel = heads[0]
                    new_arcs.append((head, d, rel))
                arcs = new_arcs
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
    utils.writelog("split: %s" % split)
    utils.writelog("# of documents: %d" % n_docs)
    utils.writelog("# of paragraphs: %d" % n_paras)
    utils.writelog("# of sentences: %d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs): %d" % n_edus)

    return dataset


if __name__ == "__main__":
    main()

