import argparse
import os

import jsonlines
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
import pyprind

import utils
import treetk

import shared_functions
import parsers
import metrics


def main(args):
    ##################
    # Arguments
    ##################

    device = torch.device(f"cuda:{args.gpu}")
    config_name = args.config
    paths_pretrained = args.pretrained
    prefix = args.prefix
    actiontype = args.actiontype

    if prefix is None or prefix == "None":
        prefix = utils.get_current_time()

    assert actiontype in ["train", "evaluate"]

    ##################
    # Config
    ##################

    config = utils.get_hocon_config(config_path="./config/main.conf", config_name=config_name)
    sw = utils.StopWatch()
    sw.start("main")

    if config["bootstrapping_type"] == "so":
        assert paths_pretrained is None
    else:
        assert len(paths_pretrained) == len(config["parser_types"])

    ##################
    # Paths
    ##################

    base_dir = "tacl2022.%s" % config_name

    utils.mkdir(os.path.join(config["results"], base_dir))

    # Log file
    path_log = None
    if actiontype == "train":
        path_log = os.path.join(config["results"], base_dir, prefix + ".training.log")
    elif actiontype == "evaluate":
        path_log = os.path.join(config["results"], base_dir, prefix + ".evaluation.log")

    # Training loss and etc.
    path_train_losses = os.path.join(config["results"], base_dir, prefix + ".train.losses.jsonl")

    # Model snapshot
    path_snapshot = os.path.join(config["results"], base_dir, prefix + ".model")

    # Automatic annotation on unlabeled data
    path_ann = os.path.join(config["results"], base_dir, prefix + ".annotation.arcs")

    # Validation outputs and scores
    path_valid_pred = os.path.join(config["results"], base_dir, prefix + ".valid.pred.arcs")
    path_valid_eval = os.path.join(config["results"], base_dir, prefix + ".valid.eval.jsonl")

    # Evaluation outputs and scores
    path_test_pred = os.path.join(config["results"], base_dir, prefix + ".test.pred.arcs")
    path_test_eval = os.path.join(config["results"], base_dir, prefix + ".test.eval.json")

    # Gold data for validation and evaluation
    if config["test_dataset_name"] == "covid19-dtb":
        path_valid_gold = os.path.join(config["caches-tacl2022"], "mapped-covid19-dtb.dev.gold.arcs")
        path_test_gold = os.path.join(config["caches-tacl2022"], "mapped-covid19-dtb.test.gold.arcs")
    elif config["test_dataset_name"] == "molweni":
        path_valid_gold = os.path.join(config["caches-tacl2022"], "molweni.dev.gold.arcs")
        path_test_gold = os.path.join(config["caches-tacl2022"], "molweni.test.gold.arcs")
    else:
        raise Exception("Never occur.")

    utils.set_logger(path_log)

    utils.writelog("device: %s" % device)
    utils.writelog("config_name: %s" % config_name)
    utils.writelog("paths_pretrained: %s" % paths_pretrained)
    utils.writelog("prefix: %s" % prefix)
    utils.writelog("actiontype: %s" % actiontype)

    utils.writelog(utils.pretty_format_dict(config))

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_losses: %s" % path_train_losses)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_ann: %s" % path_ann)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_eval: %s" % path_valid_eval)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_eval: %s" % path_test_eval)

    ##################
    # Datasets
    ##################

    sw.start("data")

    #####
    # Training labeled dataset
    #####

    # Read a training labeled dataset
    if config["train_labeled_dataset_name"] == "scidtb":
        labeled_dataset = np.load(os.path.join(config["caches-tacl2022"], "mapped-scidtb.train-.scibert_scivocab_uncased.npy"), allow_pickle=True)
    elif config["train_labeled_dataset_name"] == "stac":
        labeled_dataset = np.load(os.path.join(config["caches-tacl2022"], "mapped-stac.train.bert-base-cased.npy"), allow_pickle=True)
    else:
        raise Exception("Never occur.")

    # Exclude non-projective data in the training labeled set (for shift-reduce/backward-sr)
    count = len(labeled_dataset)
    labeled_dataset = utils.filter_dataset(labeled_dataset, condition=lambda data: shared_functions.is_projective(data.arcs))
    utils.writelog("Excluded %d (= %d - %d) non-projective data in the training labeled set" % (count - len(labeled_dataset), count, len(labeled_dataset)))

    # Exclude cyclic data in the training labeled set (for shift-reduce/backward-sr)
    count = len(labeled_dataset)
    labeled_dataset = utils.filter_dataset(labeled_dataset, condition=lambda data: shared_functions.is_dag(arcs=data.arcs, n_nodes=len(data.edus)))
    utils.writelog("Excluded %d (= %d - %d) cyclic data in the training labeled set" % (count - len(labeled_dataset), count, len(labeled_dataset)))

    # Reduce the training labeled set
    count = len(labeled_dataset)
    labeled_dataset = labeled_dataset[:config["seed_size"]]
    utils.writelog("Reduced the training labeled set to %d data" % len(labeled_dataset))

    #####
    # Development and test datasets
    #####

    # Read development and test datasets
    if config["test_dataset_name"] == "covid19-dtb":
        dev_dataset = np.load(os.path.join(config["caches-tacl2022"], "mapped-covid19-dtb.dev.scibert_scivocab_uncased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-tacl2022"], "mapped-covid19-dtb.test.scibert_scivocab_uncased.npy"), allow_pickle=True)
    elif config["test_dataset_name"] == "molweni":
        dev_dataset = np.load(os.path.join(config["caches-tacl2022"], "molweni.dev.bert-base-cased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-tacl2022"], "molweni.test.bert-base-cased.npy"), allow_pickle=True)
    else:
         raise Exception("Never occur.")

    #####
    # Training unlabeled dataset
    #####

    # Read a training unlabeled dataset
    if config["train_unlabeled_dataset_name"] == "cord19-abst":
        unlabeled_dataset = np.load(os.path.join(config["caches-tacl2022"], "cord19-abst.scibert_scivocab_uncased.npy"), allow_pickle=True)
    elif config["train_unlabeled_dataset_name"] == "ubuntu-dialogue-corpus":
        unlabeled_dataset = np.load(os.path.join(config["caches-tacl2022"], "ubuntu-dialogue-corpus.bert-base-cased.npy"), allow_pickle=True)
    else:
        raise Exception("Never occur.")

    # Exclude too long documents in the training unlabeled set
    count = len(unlabeled_dataset)
    unlabeled_dataset = utils.filter_dataset(unlabeled_dataset, condition=lambda data: 3 <= len(data.edus) <= 40)
    utils.writelog("Excluded %d (= %d - %d) too-long documents in the unlabeled set" % (count - len(unlabeled_dataset), count, len(unlabeled_dataset)))

    # Exclude the documents contained in COVID19-DTB from the training unlabeled set
    if config["train_unlabeled_dataset_name"] == "cord19-abst" and config["test_dataset_name"] == "covid19-dtb":
        count = len(unlabeled_dataset)
        covid19dtb_ids = [d.id for d in test_dataset] + [d.id for d in dev_dataset]
        unlabeled_dataset = utils.filter_dataset(unlabeled_dataset, condition=lambda data: not data.id in covid19dtb_ids)
        utils.writelog("Excluded %d (= %d - %d) documents contained in COVID19-DTB from the unlabeled set" % (count - len(unlabeled_dataset), count, len(unlabeled_dataset)))

    # Reduce the training unlabeled set
    if config["reduced_unlabeled_dataset_size"] >= 0:
        count = len(unlabeled_dataset)
        unlabeled_dataset = unlabeled_dataset[:config["reduced_unlabeled_dataset_size"]]
        utils.writelog("Reduced the unlabeled set to %d data" % len(unlabeled_dataset))

    #####
    # Relation classes
    #####

    # Read relation class vocabulary
    if config["train_labeled_dataset_name"] == "scidtb" and config["test_dataset_name"] == "covid19-dtb":
        vocab_relation = utils.read_vocab(os.path.join(config["caches-tacl2022"], "mapped-covid19-dtb.relations.vocab.txt"))
    elif config["train_labeled_dataset_name"] == "stac" and config["test_dataset_name"] == "molweni":
        vocab_relation = utils.read_vocab(os.path.join(config["caches-tacl2022"], "molweni.relations.vocab.txt"))
    else:
        raise Exception("Never occur.")
    utils.writelog("Relation vocabulary: %s" % utils.pretty_format_dict(vocab_relation))

    utils.writelog("Number of training labeled data: %d" % len(labeled_dataset))
    utils.writelog("Number of training unlabeled documents: %d" % len(unlabeled_dataset))
    utils.writelog("Number of validation data: %d" % len(dev_dataset))
    utils.writelog("Number of test data: %d" % len(test_dataset))

    sw.stop("data")
    utils.writelog("Loaded the corpus. %f [sec.]" % sw.get_time("data"))

    ##################
    # Parsers
    ##################

    n_parsers = len(config["parser_types"])

    parser_list = []
    for p_i in range(n_parsers):
        parser_type = config["parser_types"][p_i]
        parser_config_name = config["parser_configs"][p_i]
        parser_config = utils.get_hocon_config(config_path="./config/main.conf", config_name=parser_config_name)
        utils.writelog("Configuration for parsers[%d]:" % p_i)
        utils.writelog(utils.pretty_format_dict(parser_config))

        if parser_type == "arcfactored":
            assert parser_config["model_name"] == "biaffine"
            parser = parsers.ArcFactoredParser(device=device,
                                               config=parser_config,
                                               vocab_relation=vocab_relation)
            parser.parser_type = parser_type
        elif parser_type == "shiftreduce":
            assert parser_config["model_name"] == "arcstandard"
            assert not parser_config["reverse_order"]
            parser = parsers.ShiftReduceParser(device=device,
                                               config=parser_config,
                                               vocab_relation=vocab_relation)
            parser.parser_type = parser_type
        elif parser_type == "backwardsr":
            assert parser_config["model_name"] == "arcstandard"
            assert parser_config["reverse_order"]
            parser = parsers.ShiftReduceParser(device=device,
                                               config=parser_config,
                                               vocab_relation=vocab_relation)
            parser.parser_type = parser_type
        else:
            raise Exception("Never occur.")

        parser_list.append(parser)

    # Load pre-trained parameters
    # Source-only: training (no) -> evaluation (yes)
    # Bootstrapping: training (yes*; from source-only) -> evaluation (yes)
    if config["bootstrapping_type"] == "so":
        if actiontype != "train":
            for p_i in range(n_parsers):
                parser_list[p_i].load_model(path=add_parser_id_to_path(path_snapshot, p_i))
                utils.writelog("Loaded parser_list[%d]'s model from %s" % (p_i, add_parser_id_to_path(path_snapshot, p_i)))
    else:
        if actiontype == "train":
            # We use the source-only models as the initial learners
            for p_i in range(n_parsers):
                init_path_snapshot = os.path.join(config["results"], paths_pretrained[p_i])
                parser_list[p_i].load_model(path=init_path_snapshot)
                utils.writelog("Loaded parser_list[%d]'s model from %s" % (p_i, init_path_snapshot))
        else:
            for p_i in range(n_parsers):
                parser_list[p_i].load_model(path=add_parser_id_to_path(path_snapshot, p_i))
                utils.writelog("Loaded parser_list[%d]'s model from %s" % (p_i, add_parser_id_to_path(path_snapshot, p_i)))

    for p_i in range(n_parsers):
        parser_list[p_i].to_gpu(device=device)

    ##################
    # Action
    ##################

    if actiontype == "train":
        train(
            config=config,
            parser_list=parser_list,
            labeled_dataset=labeled_dataset,
            unlabeled_dataset=unlabeled_dataset,
            dev_dataset=dev_dataset,
            path_train_losses=path_train_losses,
            path_snapshot=path_snapshot,
            path_ann=path_ann,
            path_valid_pred=path_valid_pred,
            path_valid_gold=path_valid_gold,
            path_valid_eval=path_valid_eval)

    elif actiontype == "evaluate":
        with torch.no_grad():
            for p_i in range(n_parsers):
                parse(
                    parser=parser_list[p_i],
                    dataset=test_dataset,
                    path_pred=add_parser_id_to_path(path_test_pred, p_i),
                    confidence_measure=config["confidence_measure"]) # Save confidence scores for analysis
                scores = metrics.attachment_scores(
                            pred_path=add_parser_id_to_path(path_test_pred, p_i),
                            gold_path=path_test_gold) # the path_test_gold should also change
                scores["LAS"] *= 100.0
                scores["UAS"] *= 100.0
                scores["UUAS"] *= 100.0
                scores["RA"] *= 100.0
                utils.write_json(add_parser_id_to_path(path_test_eval, p_i), scores)
                utils.writelog(utils.pretty_format_dict(scores))
                shared_functions.save_prediction_as_scidtb_format(dataset=test_dataset, path_pred=add_parser_id_to_path(path_test_pred, p_i))

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_losses: %s" % path_train_losses)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_ann: %s" % path_ann)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_eval: %s" % path_valid_eval)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_eval: %s" % path_test_eval)
    utils.writelog("Done.")
    sw.stop("main")
    utils.writelog("Time: %f min." % sw.get_time("main", minute=True))


def add_parser_id_to_path(path, parser_id):
    """
    Parameters
    ----------
    path: str
    parser_id: int

    Returns
    -------
    str
    """
    return "%s.%d" % (path, parser_id)


##################################
# Training
##################################


def train(
        config,
        parser_list,
        labeled_dataset,
        unlabeled_dataset,
        dev_dataset,
        path_train_losses,
        path_snapshot,
        path_ann,
        path_valid_pred,
        path_valid_gold,
        path_valid_eval):
    """
    Parameters
    ----------
    config: ConfigTree
    parser_list: list[P], where P denotes ArcFactoredParser or ShiftReduceParser
    labeled_dataset: numpy.ndarray
    unlabeled_dataset: numpy.ndarray
    dev_dataset: numpy.ndarray
    path_train_losses: str
    path_snapshot: str
    path_ann: str
    path_valid_pred: str
    path_valid_gold: str
    path_valid_eval: str
    """
    n_parsers = len(parser_list)

    # Get optimizers and schedulers
    n_labeled = len(labeled_dataset)
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    if config["bootstrapping_type"] == "so":
        total_update_steps = n_labeled * max_epoch // batch_size
        warmup_steps = int(total_update_steps * config["warmup_ratio"])
    elif config["bootstrapping_type"] in ["st", "ct"]:
        # NOTE: In bootstrapping, `total_update_steps` is not the actual total number of update steps,
        #       because the number of selected pseudo-labeled data is the same with or smaller than
        #       the number of sampled unlabeled data (=`unlabeled_data_sampling_size`).
        total_update_steps = (n_labeled + config["unlabeled_data_sampling_size"]) * max_epoch // batch_size
        warmup_steps = 7000 # Set empirically
    elif config["bootstrapping_type"] in ["tt", "at"]:
        total_update_steps = (n_labeled + config["unlabeled_data_sampling_size"] * 2) * max_epoch // batch_size
        warmup_steps = 7000 # Set empirically
    else:
        raise Exception("Never occur.")

    optimizers_list = []
    schedulers_list = []
    for p_i in range(n_parsers):
        if config["bootstrapping_type"] == "so":
            optimizers = shared_functions.get_optimizer(model=parser_list[p_i].model, config=parser_list[p_i].model.config)
            schedulers = shared_functions.get_scheduler(optimizers=optimizers, total_update_steps=total_update_steps, warmup_steps=warmup_steps)
        else:
            optimizers = get_optimizer_for_further_finetuning(model=parser_list[p_i].model, config=parser_list[p_i].model.config)
            schedulers = get_scheduler_for_further_finetuning(optimizers=optimizers, total_update_steps=total_update_steps, warmup_steps=warmup_steps)
        optimizers_list.append(optimizers)
        schedulers_list.append(schedulers)

    utils.writelog("*********************Training*********************")
    utils.writelog("n_labeled: %d" % n_labeled)
    utils.writelog("max_epoch: %d" % max_epoch)
    utils.writelog("batch_size: %d" % batch_size)
    utils.writelog("total_update_steps: %d" % total_update_steps)
    utils.writelog("warmup_steps: %d" % warmup_steps)

    writer_train = jsonlines.Writer(open(path_train_losses, "w"), flush=True)
    writer_valid = jsonlines.Writer(open(path_valid_eval, "w"), flush=True)
    bestscore_holders = {}
    bestscore_holders["joint"] = utils.BestScoreHolder(scale=1.0)
    bestscore_holders["joint"].init()
    bestscore_holders["independent"] = [None for _ in range(n_parsers)]
    for p_i in range(n_parsers):
        bestscore_holders["independent"][p_i] = utils.BestScoreHolder(scale=1.0)
        bestscore_holders["independent"][p_i].init()
    step_list = [0, 0, 0]
    bert_param_list = []
    task_param_list = []
    for p_i in range(n_parsers):
        bert_param, task_param = parser_list[p_i].model.get_params()
        bert_param_list.append(bert_param)
        task_param_list.append(task_param)

    ##################
    # Initial validation phase
    ##################

    best_las = -1.0
    with torch.no_grad():
        for p_i in range(n_parsers):
            parse(
                parser=parser_list[p_i],
                dataset=dev_dataset,
                path_pred=add_parser_id_to_path(path_valid_pred, p_i))
            scores = metrics.attachment_scores(
                            pred_path=add_parser_id_to_path(path_valid_pred, p_i),
                            gold_path=path_valid_gold)
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            scores["UUAS"] *= 100.0
            scores["RA"] *= 100.0
            scores["epoch"] = 0
            writer_valid.write(scores)
            utils.writelog(utils.pretty_format_dict(scores))

            bestscore_holders["independent"][p_i].compare_scores(scores["LAS"], 0)

            # Save the model
            parser_list[p_i].save_model(path=add_parser_id_to_path(path_snapshot, p_i))
            utils.writelog("Saved parser_list[%d]'s model to %s" % (p_i, add_parser_id_to_path(path_snapshot, p_i)))

            if (config["bootstrapping_type"] != "at") or (config["bootstrapping_type"] == "at" and p_i == 0):
                if best_las < scores["LAS"]:
                    best_las = scores["LAS"]

    bestscore_holders["joint"].compare_scores(best_las, 0)

    ##################
    # /Initial validation phase
    ##################

    ##################
    # Training-and-validation loops
    ##################

    for epoch in range(1, max_epoch + 1):

        ##################
        # Annotation phase
        ##################

        if config["bootstrapping_type"] == "so":
            # In the source-only system, the training dataset consists of only manually labeled data
            train_dataset_list = [labeled_dataset]

        elif (epoch == 1) or ((epoch - 1) % config["annotation_reflesh_frequency"] == 0):
            # 0. Reflesh the training dataset
            train_dataset_list = []

            # 1. Sample unlabeled documents' indices
            utils.writelog("1. Sampling unlabeled documents ...")
            subset_indices = np.random.permutation(len(unlabeled_dataset))[:config["unlabeled_data_sampling_size"]]

            # 2. Annotate the sampled unlabeled documents and measure the confidence scores
            utils.writelog("2. Annotating sampled unlabeled documents and measuring the confidence scores ...")
            path_ann_list = [add_parser_id_to_path(path_ann, p_i) for p_i in range(n_parsers)]
            with torch.no_grad():
                for p_i in range(n_parsers):
                    parse(parser=parser_list[p_i],
                          dataset=unlabeled_dataset[subset_indices],
                          path_pred=path_ann_list[p_i],
                          confidence_measure=config["confidence_measure"])
            if config["bootstrapping_type"] == "tt":
                compute_agreement_scores_for_tritraining(path_ann_list=path_ann_list,
                                                         method=config["agreement_method"],
                                                         average=config["agreement_average"])
            elif config["bootstrapping_type"] == "at":
                compute_agreement_scores_for_asymmetric_tritraining(path_ann_list=path_ann_list,
                                                                    method=config["agreement_method"],
                                                                    average=config["agreement_average"])

            # 3. Select pseudo-labeled data using a sample selection criterion
            utils.writelog("3. Selecting pseudo-labeled data ...")
            if config["bootstrapping_type"] == "st":
                pseudo_labeled_dataset_list, info \
                            = select_pseudo_labeled_data_for_selftraining(
                                                            unlabeled_dataset=unlabeled_dataset[subset_indices],
                                                            path_ann_list=path_ann_list,
                                                            topk_ratio=config["topk_ratio"])
            elif config["bootstrapping_type"] == "ct":
                pseudo_labeled_dataset_list, info \
                            = select_pseudo_labeled_data_for_cotraining(
                                                            unlabeled_dataset=unlabeled_dataset[subset_indices],
                                                            path_ann_list=path_ann_list,
                                                            selection_method=config["selection_method"],
                                                            topk_ratio=config["topk_ratio"],
                                                            diff_margin=config["diff_margin"])
            elif config["bootstrapping_type"] == "tt":
                pseudo_labeled_dataset_list, info \
                            = select_pseudo_labeled_data_for_tritraining(
                                                            unlabeled_dataset=unlabeled_dataset[subset_indices],
                                                            path_ann_list=path_ann_list,
                                                            selection_method=config["selection_method"],
                                                            topk_ratio=config["topk_ratio"],
                                                            diff_margin=config["diff_margin"])
            elif config["bootstrapping_type"] == "at":
                pseudo_labeled_dataset_list, info \
                            = select_pseudo_labeled_data_for_asymmetric_tritraining(
                                                            unlabeled_dataset=unlabeled_dataset[subset_indices],
                                                            path_ann_list=path_ann_list,
                                                            selection_method=config["selection_method"],
                                                            topk_ratio=config["topk_ratio"],
                                                            diff_margin=config["diff_margin"])
            else:
                raise Exception("Never occur.")
            assert len(pseudo_labeled_dataset_list) == n_parsers

            # Print info.
            for p_i in range(n_parsers):
                n_pseudo_labeled = len(pseudo_labeled_dataset_list[p_i])
                ratio = float(n_pseudo_labeled) / len(subset_indices) * 100.0
                if config["bootstrapping_type"] in ["tt", "at"]:
                    ratio *= 0.5 # Because two partially-agreed trees for each document were added to the same dataset
                utils.writelog("[Epoch %d; Parser %d] Number of pseudo-labeled data: %d (Utility: %.02f%%; Range: [%.02f, %.02f])" % \
                                    (epoch, p_i, n_pseudo_labeled, ratio, info[p_i]["min_score"], info[p_i]["max_score"]))

            # 4. Combine the labeled and pseudo-labeled datasets
            utils.writelog("4. Combining the labeled and pseudo-labeled datasets ...")
            for p_i in range(n_parsers):
                pseudo_labeled_dataset = np.asarray(pseudo_labeled_dataset_list[p_i], dtype="O")
                if config["bootstrapping_type"] == "at" and p_i == 0:
                    # In asymmetric tri-training, we do not use the labeled source dataset for training the target-domain model (p_i=0)
                    train_dataset = pseudo_labeled_dataset
                else:
                    train_dataset = np.concatenate([labeled_dataset, pseudo_labeled_dataset], axis=0)
                train_dataset_list.append(train_dataset)

        ##################
        # /Annotation phase
        ##################

        ##################
        # Training phase
        ##################

        for p_i in range(n_parsers):

            train_dataset = train_dataset_list[p_i]
            n_train = len(train_dataset)
            perm = np.random.permutation(n_train)

            ##################
            # Arc-Factored
            ##################

            if parser_list[p_i].parser_type == "arcfactored":
                for instance_i in range(0, n_train, batch_size):

                    ##################
                    # Mini batch
                    ##################

                    # Initialize losses
                    loss_attachment = 0.0
                    acc_attachment = 0.0
                    loss_relation = 0.0
                    acc_relation = 0.0
                    actual_batchsize = 0
                    actual_total_arcs = 0

                    for data in train_dataset[perm[instance_i:instance_i + batch_size]]:

                        ##################
                        # One data
                        ##################

                        # Forward and compute the losses
                        one_loss_attachment, \
                            one_acc_attachment, \
                            one_loss_relation, \
                            one_acc_relation, \
                            n_arcs = parser_list[p_i].compute_loss(data=data)

                        # Accumulate the losses
                        loss_attachment = loss_attachment + one_loss_attachment
                        acc_attachment = acc_attachment + one_acc_attachment
                        loss_relation = loss_relation + one_loss_relation
                        acc_relation = acc_relation + one_acc_relation
                        actual_batchsize += 1
                        actual_total_arcs += n_arcs

                        ##################
                        # /One data
                        ##################

                    # Average the losses
                    actual_batchsize = float(actual_batchsize)
                    actual_total_arcs = float(actual_total_arcs)
                    loss_attachment = loss_attachment / actual_total_arcs
                    acc_attachment = acc_attachment / actual_total_arcs
                    loss_relation = loss_relation / actual_total_arcs
                    acc_relation = acc_relation / actual_total_arcs

                    # Merge the losses
                    loss = loss_attachment + loss_relation

                    # Backward
                    parser_list[p_i].model.zero_grad()
                    loss.backward()

                    # Update
                    if parser_list[p_i].model.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param_list[p_i],
                            parser_list[p_i].model.config["max_grad_norm"])
                        torch.nn.utils.clip_grad_norm_(
                            task_param_list[p_i],
                            parser_list[p_i].model.config["max_grad_norm"])
                    for optimizer in optimizers_list[p_i]:
                        optimizer.step()
                    for scheduler in schedulers_list[p_i]:
                        scheduler.step()

                    step_list[p_i] += 1

                    # Write log
                    loss_attachment_data = float(loss_attachment.cpu())
                    loss_relation_data = float(loss_relation.cpu())
                    out = {
                           "parser_id": p_i,
                           "step": step_list[p_i],
                           "epoch": epoch,
                           "progress": "%d/%d" % (instance_i + actual_batchsize, n_train),
                           "progress_ratio": float(instance_i + actual_batchsize) / n_train * 100.0,
                           "attachment_loss": loss_attachment_data,
                           "attachment_accuracy": acc_attachment * 100.0,
                           "relation_loss": loss_relation_data,
                           "relation_accuracy": acc_relation * 100.0,
                           "max_valid_las_ind": bestscore_holders["independent"][p_i].best_score,
                           "patience_ind": bestscore_holders["independent"][p_i].patience,
                           "max_valid_las_joint": bestscore_holders["joint"].best_score,
                           "patience_joint": bestscore_holders["joint"].patience}
                    writer_train.write(out)
                    utils.writelog(utils.pretty_format_dict(out))

                    ##################
                    # /Mini batch
                    ##################

            ##################
            # /Arc-Factored
            ##################

            ##################
            # Shift-Reduce or Backward-SR
            ##################

            elif parser_list[p_i].parser_type == "shiftreduce" or parser_list[p_i].parser_type == "backwardsr":
                for instance_i in range(0, n_train, batch_size):

                    ##################
                    # Mini batch
                    ##################

                    # Initialize loss
                    loss = 0.0
                    acc = 0.0
                    actual_batchsize = 0
                    actual_total_actions = 0

                    for data in train_dataset[perm[instance_i:instance_i + batch_size]]:

                        ##################
                        # One data
                        ##################

                        # Forward and compute loss
                        one_loss, one_acc, n_action_steps = parser_list[p_i].compute_loss(data=data)

                        # Accumulate the loss
                        loss = loss + one_loss
                        acc = acc + one_acc
                        actual_batchsize += 1
                        actual_total_actions += n_action_steps

                        ##################
                        # /One data
                        ##################

                    # Average the loss
                    actual_batchsize = float(actual_batchsize)
                    actual_total_actions = float(actual_total_actions)
                    loss = loss / actual_total_actions
                    acc = acc / actual_total_actions

                    # Backward
                    parser_list[p_i].model.zero_grad()
                    loss.backward()

                    # Update
                    if parser_list[p_i].model.config["max_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            bert_param_list[p_i],
                            parser_list[p_i].model.config["max_grad_norm"])
                        torch.nn.utils.clip_grad_norm_(
                            task_param_list[p_i],
                            parser_list[p_i].model.config["max_grad_norm"])
                    for optimizer in optimizers_list[p_i]:
                        optimizer.step()
                    for scheduler in schedulers_list[p_i]:
                        scheduler.step()

                    step_list[p_i] += 1

                    # Write log
                    loss_data = float(loss.cpu())
                    out = {"parser_id": p_i,
                           "step": step_list[p_i],
                           "epoch": epoch,
                           "progress": "%d/%d" % (instance_i + actual_batchsize, n_train),
                           "progress_ratio": float(instance_i + actual_batchsize) / n_train * 100.0,
                           "loss": loss_data,
                           "accuracy": acc * 100.0,
                           "max_valid_las_ind": bestscore_holders["independent"][p_i].best_score,
                           "patience_ind": bestscore_holders["independent"][p_i].patience,
                           "max_valid_las_joint": bestscore_holders["joint"].best_score,
                           "patience_joint": bestscore_holders["joint"].patience}
                    writer_train.write(out)
                    utils.writelog(utils.pretty_format_dict(out))

                    ##################
                    # /Mini batch
                    ##################

            ##################
            # /Shift-Reduce or Backward-SR
            ##################

        ##################
        # /Training phase
        ##################

        ##################
        # Validation phase
        ##################

        best_las = -1
        with torch.no_grad():
            for p_i in range(n_parsers):
                parse(
                    parser=parser_list[p_i],
                    dataset=dev_dataset,
                    path_pred=add_parser_id_to_path(path_valid_pred, p_i))
                scores = metrics.attachment_scores(
                            pred_path=add_parser_id_to_path(path_valid_pred, p_i),
                            gold_path=path_valid_gold)
                scores["LAS"] *= 100.0
                scores["UAS"] *= 100.0
                scores["UUAS"] *= 100.0
                scores["RA"] *= 100.0
                scores["epoch"] = epoch
                writer_valid.write(scores)
                utils.writelog(utils.pretty_format_dict(scores))

                did_update = bestscore_holders["independent"][p_i].compare_scores(scores["LAS"], epoch)

                # Save the model?
                if did_update:
                    parser_list[p_i].save_model(path=add_parser_id_to_path(path_snapshot, p_i))
                    utils.writelog("Saved parser_list[%d] to %s" % (p_i, add_parser_id_to_path(path_snapshot, p_i)))

                if (config["bootstrapping_type"] != "at") or (config["bootstrapping_type"] == "at" and p_i == 0):
                    if best_las < scores["LAS"]:
                        best_las = scores["LAS"]

        bestscore_holders["joint"].compare_scores(best_las, epoch)
        utils.writelog("[Epoch %d] Max validation LAS: %f" % (epoch, bestscore_holders["joint"].best_score))

        # Finished?
        if bestscore_holders["joint"].ask_finishing(max_patience=10):
            utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holders["joint"].patience)
            writer_train.close()
            writer_valid.close()
            return

        ##################
        # /Validation phase
        ##################

    ##################
    # /Training-and-validation loops
    ##################

    writer_train.close()
    writer_valid.close()


def get_optimizer_for_further_finetuning(model, config):
    """
    Parameters
    ----------
    model: ArcFactoredModel
    config: ConfigTree

    Returns
    -------
    [transformers.AdamW, torch.optim.Adam]
    """
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param, task_param = model.get_params(named=True)
    grouped_bert_param = [
        {
            "params": [p for n,p in bert_param if not any(nd in n for nd in no_decay)],
            "lr": config["finetune_bert_learning_rate"],
            "weight_decay": config["adam_weight_decay"],
        },
        {
            "params": [p for n,p in bert_param if any(nd in n for nd in no_decay)],
            "lr": config["finetune_bert_learning_rate"],
            "weight_decay": 0.0,
        }
    ]
    optimizers = [
        AdamW(grouped_bert_param, lr=config["finetune_bert_learning_rate"], eps=config["adam_eps"]),
        Adam(model.get_params()[1], lr=config["finetune_task_learning_rate"], eps=config["adam_eps"], weight_decay=0)
    ]
    return optimizers


def get_scheduler_for_further_finetuning(optimizers, total_update_steps, warmup_steps):
    """
    Parameters
    ----------
    optimizers: [transformers.AdamW, torcht.optim.Adam]
    total_update_steps: int
    warmup_steps: int

    Returns
    -------
    list[torch.optim.lr_scheduler.LambdaLR]
    """
    # Warm up all parameters
    def lr_lambda_finetune(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
               )

    schedulers = [
        LambdaLR(optimizers[0], lr_lambda_finetune),
        LambdaLR(optimizers[1], lr_lambda_finetune)
    ]
    return schedulers


##################################
# Bootstrapping functions
##################################


def compute_agreement_scores_for_tritraining(path_ann_list, method, average):
    """Agreement-based confidence measure for tri-training

    Parameters
    ----------
    path_ann_list: list[str]
    method: str
    average: bool
    """
    N_PARSERS = 3
    assert len(path_ann_list) == N_PARSERS

    # Read pseudo labels
    batch_arcs_list = [] # list[list[list[(int, int, str)]]]; shape: (3, batch_size, n_arcs)
    for path_ann in path_ann_list:
        batch_arcs = utils.read_lines(path_ann, process=lambda line: treetk.hyphens2arcs(line.split()))
        batch_arcs_list.append(batch_arcs)

    n_data = len(batch_arcs_list[0])
    for batch_arcs in batch_arcs_list:
        assert len(batch_arcs) == n_data

    # Compute agreement scores between p_j and p_k for p_i
    agreement_scores_list = [[] for _ in range(N_PARSERS)]
    for data_i in range(n_data):
        for p_i in range(N_PARSERS):
            p_j = (p_i + 1) % 3
            p_k = (p_i + 2) % 3
            score = compute_agreement_score(arcs1=batch_arcs_list[p_j][data_i],
                                            arcs2=batch_arcs_list[p_k][data_i],
                                            method=method,
                                            average=average)
            agreement_scores_list[p_i].append(score)

    # Save
    for p_i in range(N_PARSERS):
        with open(path_ann_list[p_i].replace(".arcs", ".arcs.agree"), "w") as f:
            for score in agreement_scores_list[p_i]:
                f.write("%f\n" % score)


def compute_agreement_scores_for_asymmetric_tritraining(path_ann_list, method, average):
    """Agreement-based confidence measure for asymmetric tri-training

    Parameters
    ----------
    path_ann_list: list[str]
    method: str
    average: bool
    """
    N_PARSERS = 3
    assert len(path_ann_list) == N_PARSERS

    # Read pseudo labels
    batch_arcs_1 = utils.read_lines(path_ann_list[1], process=lambda line: treetk.hyphens2arcs(line.split()))
    batch_arcs_2 = utils.read_lines(path_ann_list[2], process=lambda line: treetk.hyphens2arcs(line.split()))

    n_data = len(batch_arcs_1)
    assert len(batch_arcs_2) == n_data

    # Compute agreement scores between p_1 and p_2 for p_0
    agreement_scores = []
    for data_i in range(n_data):
        score = compute_agreement_score(arcs1=batch_arcs_1[data_i],
                                        arcs2=batch_arcs_2[data_i],
                                        method=method,
                                        average=average)
        agreement_scores.append(score)

    # Save
    with open(path_ann_list[0].replace(".arcs", ".arcs.agree"), "w") as f:
        for ratio in agreement_scores:
            f.write("%f\n" % ratio)


def compute_agreement_score(arcs1, arcs2, method, average):
    """Agreement score between two dependency structures

    Parameters
    ----------
    arcs1: list[(int, int, str)]
    arcs2: list[(int, int, str)]
    method: str
    average: bool

    Returns
    -------
    float
    """
    assert len(arcs1) == len(arcs2)
    if method == "joint":
        shared_arcs = set(arcs1) & set(arcs2)
        score = float(len(shared_arcs))
    elif method == "independent":
        score = 0.0
        dep2head1 = {d: (h, r) for h, d, r in arcs1}
        dep2head2 = {d: (h, r) for h, d, r in arcs2}
        for dep in dep2head1.keys():
            head1, rel1 = dep2head1[dep]
            head2, rel2 = dep2head2[dep]
            if head1 == head2:
                score += 0.5
                if rel1 == rel2:
                    score += 0.5
    else:
        raise Exception("Never occur.")

    if average:
        score = float(score) / len(arcs1)

    return score


def select_pseudo_labeled_data_for_selftraining(
        unlabeled_dataset,
        path_ann_list,
        topk_ratio):
    """Sample selection function for self-training

    Parameters
    ----------
    unlabeled_dataset: numpy.ndarray
    path_ann_list: list[str]
    topk_ratio: float

    Returns
    -------
    list[numpy.ndarray]
    dict[str, Any]
    """
    N_PARSERS = 1
    assert len(path_ann_list) == N_PARSERS

    pool_size = len(unlabeled_dataset)

    pseudo_labeled_dataset_list = [[] for _ in range(N_PARSERS)]
    info = [{} for _ in range(N_PARSERS)]

    # Read pseudo labels and confidence scores
    batch_arcs_list = []
    confidence_scores_list = []
    for p_i, path_ann in enumerate(path_ann_list):
        # Read pseudo labels
        batch_arcs = utils.read_lines(path_ann, process=lambda line: treetk.hyphens2arcs(line.split()))
        assert len(batch_arcs) == pool_size
        batch_arcs_list.append(batch_arcs)
        # Read confidence scores
        confidence_scores = utils.read_lines(path_ann.replace(".arcs", ".arcs.conf"), process=lambda line: float(line))
        assert len(confidence_scores) == pool_size
        confidence_scores_list.append(confidence_scores)

    # Get indicators for the selected pseudo-labeled data
    indicators_list = []
    for teacher_i in range(N_PARSERS):
        indicators, info[teacher_i]["max_score"], info[teacher_i]["min_score"] \
                = rank_above_k(confidence_scores=confidence_scores_list[teacher_i],
                               topk=int(pool_size * topk_ratio))
        indicators_list.append(indicators)

    # Transform each pseudo-labeled data to DataInstance
    for data_i, data in enumerate(pyprind.prog_bar(unlabeled_dataset)):
        for student_i in range(N_PARSERS):
            teacher_i = student_i
            if indicators_list[teacher_i][data_i]:
                teacher_arcs = batch_arcs_list[teacher_i][data_i]
                pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_arcs)
                pseudo_labeled_dataset_list[student_i].append(pseudo_labeled_data)

    return pseudo_labeled_dataset_list, info


def select_pseudo_labeled_data_for_cotraining(
        unlabeled_dataset,
        path_ann_list,
        selection_method,
        topk_ratio=None,
        diff_margin=None):
    """Sample selection function for co-training

    Parameters
    ----------
    unlabeled_dataset: numpy.ndarray
    path_ann_list: list[str]
    selection_method: str
    topk_ratio: float or None, default None
    diff_margin: int or None, default none

    Returns
    -------
    list[numpy.ndarray]
    dict[str, Any]
    """

    N_PARSERS = 2
    assert len(path_ann_list) == N_PARSERS

    pool_size = len(unlabeled_dataset)

    pseudo_labeled_dataset_list = [[] for _ in range(N_PARSERS)]
    info = [{} for _ in range(N_PARSERS)]

    # Read pseudo labels and confidence scores
    batch_arcs_list = []
    confidence_scores_list = []
    for p_i, path_ann in enumerate(path_ann_list):
        # Read pseudo labels
        batch_arcs = utils.read_lines(path_ann, process=lambda line: treetk.hyphens2arcs(line.split()))
        assert len(batch_arcs) == pool_size
        batch_arcs_list.append(batch_arcs)
        # Read confidence scores
        confidence_scores = utils.read_lines(path_ann.replace(".arcs", ".arcs.conf"), process=lambda line: float(line))
        assert len(confidence_scores) == pool_size
        confidence_scores_list.append(confidence_scores)

    # Get indicators for the selected pseudo-labeled data
    indicators_list = []
    for teacher_i in range(N_PARSERS):
        if selection_method == "above":
            assert topk_ratio is not None
            indicators, info[teacher_i]["max_score"], info[teacher_i]["min_score"] \
                    = rank_above_k(confidence_scores=confidence_scores_list[teacher_i],
                                   topk=int(pool_size * topk_ratio))
        elif selection_method == "diff":
            assert diff_margin is not None
            student_i = (teacher_i + 1) % 2
            indicators, info[teacher_i]["max_score"], info[teacher_i]["min_score"] \
                    = rank_diff_k(teacher_scores=confidence_scores_list[teacher_i],
                                  student_scores=confidence_scores_list[student_i],
                                  diff_margin=diff_margin)
        else:
            raise Exception("Never occur.")
        indicators_list.append(indicators)

    # Transform each pseudo-labeled data to DataInstance
    for data_i, data in enumerate(pyprind.prog_bar(unlabeled_dataset)):
        for student_i in range(N_PARSERS):
            teacher_i = (student_i + 1) % 2
            if indicators_list[teacher_i][data_i]:
                teacher_arcs = batch_arcs_list[teacher_i][data_i]
                pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_arcs)
                pseudo_labeled_dataset_list[student_i].append(pseudo_labeled_data)

    return pseudo_labeled_dataset_list, info


def select_pseudo_labeled_data_for_tritraining(
        unlabeled_dataset,
        path_ann_list,
        selection_method,
        topk_ratio=None,
        diff_margin=None):
    """Sample selection function for tri-training

    Parameters
    ----------
    unlabeled_dataset: numpy.ndarray
    path_ann_list: list[str]
    selection_method: str
    topk_ratio: float or None, default None
    diff_margin: int or None, default none

    Returns
    -------
    list[numpy.ndarray]
    dict[str, Any]
    """

    N_PARSERS = 3
    assert len(path_ann_list) == N_PARSERS

    pool_size = len(unlabeled_dataset)

    pseudo_labeled_dataset_list = [[] for _ in range(N_PARSERS)]
    info = [{} for _ in range(N_PARSERS)]

    # Read pseudo labels and confidence scores
    batch_arcs_list = []
    confidence_scores_list = []
    agreement_scores_list = []
    for p_i, path_ann in enumerate(path_ann_list):
        # Read pseudo labels
        batch_arcs = utils.read_lines(path_ann, process=lambda line: treetk.hyphens2arcs(line.split()))
        assert len(batch_arcs) == pool_size
        batch_arcs_list.append(batch_arcs)
        # Read model-based confidence scores
        confidence_scores = utils.read_lines(path_ann.replace(".arcs", ".arcs.conf"), process=lambda line: float(line))
        assert len(confidence_scores) == pool_size
        confidence_scores_list.append(confidence_scores)
        # Read agreement-based confidence scores
        agreement_scores = utils.read_lines(path_ann.replace(".arcs", ".arcs.agree"), process=lambda line: float(line))
        assert len(agreement_scores) == pool_size
        agreement_scores_list.append(agreement_scores)

    # Get indicators for the selected pseudo-labeled data
    indicators_list = []
    for student_i in range(N_PARSERS):
        if selection_method == "above":
            assert topk_ratio is not None
            indicators, info[student_i]["max_score"], info[student_i]["min_score"] \
                    = rank_above_k(confidence_scores=agreement_scores_list[student_i],
                                   topk=int(pool_size * topk_ratio))

        elif selection_method == "diff":
            assert diff_margin is not None
            indicators, info[student_i]["max_score"], info[student_i]["min_score"] \
                    = rank_diff_k(teacher_scores=agreement_scores_list[student_i],
                                  student_scores=confidence_scores_list[student_i],
                                  diff_margin=diff_margin)
        else:
            raise Exception("Never occur.")
        indicators_list.append(indicators)

    # Transform each pseudo-labeled data to DataInstance
    for data_i, data in enumerate(pyprind.prog_bar(unlabeled_dataset)):
        for student_i in range(N_PARSERS):
            teacher_i = (student_i + 1) % 3
            teacher_j = (student_i + 2) % 3
            if indicators_list[student_i][data_i]:
                teacher_i_arcs = batch_arcs_list[teacher_i][data_i]
                teacher_j_arcs = batch_arcs_list[teacher_j][data_i]
                pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_i_arcs)
                pseudo_labeled_dataset_list[student_i].append(pseudo_labeled_data)
                #
                pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_j_arcs)
                pseudo_labeled_dataset_list[student_i].append(pseudo_labeled_data)

    return pseudo_labeled_dataset_list, info


def select_pseudo_labeled_data_for_asymmetric_tritraining(
        unlabeled_dataset,
        path_ann_list,
        selection_method,
        topk_ratio=None,
        diff_margin=None):
    """Sample selection function for asymmetric tri-training

    Parameters
    ----------
    unlabeled_dataset: numpy.ndarray
    path_ann_list: list[str]
    selection_method: str
    topk_ratio: float or None, default None
    diff_margin: int or None, default none

    Returns
    -------
    list[numpy.ndarray]
    dict[str, Any]
    """

    N_PARSERS = 3
    assert len(path_ann_list) == N_PARSERS

    pool_size = len(unlabeled_dataset)

    pseudo_labeled_dataset = []
    info = {}

    # Read pseudo labels
    batch_arcs_1 = utils.read_lines(path_ann_list[1], process=lambda line: treetk.hyphens2arcs(line.split()))
    batch_arcs_2 = utils.read_lines(path_ann_list[2], process=lambda line: treetk.hyphens2arcs(line.split()))
    assert len(batch_arcs_1) == pool_size
    assert len(batch_arcs_2) == pool_size
    # Read model-based confidence scores
    confidence_scores = utils.read_lines(path_ann_list[0].replace(".arcs", ".arcs.conf"), process=lambda line: float(line))
    assert len(confidence_scores) == pool_size
    # Read agreement-based confidence scores
    agreement_scores = utils.read_lines(path_ann_list[0].replace(".arcs", ".arcs.agree"), process=lambda line: float(line))
    assert len(agreement_scores) == pool_size

    # Get indicators for the selected pseudo-labeled data
    if selection_method == "above":
        assert topk_ratio is not None
        indicators, info["max_score"], info["min_score"] \
                 = rank_above_k(confidence_scores=agreement_scores,
                                topk=int(pool_size * topk_ratio))
    elif selection_method == "diff":
        assert diff_margin is not None
        indicators, info["max_score"], info["min_score"] \
                = rank_diff_k(teacher_scores=agreement_scores,
                              student_scores=confidence_scores,
                              diff_margin=diff_margin)
    else:
        raise Exception("Never occur.")

    # Transform each pseudo-labeled data to DataInstance
    for data_i, data in enumerate(pyprind.prog_bar(unlabeled_dataset)):
        if indicators[data_i]:
            teacher_1_arcs = batch_arcs_1[data_i]
            pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_1_arcs)
            pseudo_labeled_dataset.append(pseudo_labeled_data)
            #
            teacher_2_arcs = batch_arcs_2[data_i]
            pseudo_labeled_data = utils.DataInstance(
                                            edu_ids=data.edu_ids,
                                            edus=data.edus,
                                            sentence_boundaries=data.sentence_boundaries,
                                            paragraph_boundaries=data.paragraph_boundaries,
                                            segments=data.segments,
                                            segments_id=data.segments_id,
                                            segments_mask=data.segments_mask,
                                            edu_begin_indices=data.edu_begin_indices,
                                            edu_end_indices=data.edu_end_indices,
                                            edu_head_indices=data.edu_head_indices,
                                            arcs=teacher_2_arcs)
            pseudo_labeled_dataset.append(pseudo_labeled_data)

    pseudo_labeled_dataset_list = [pseudo_labeled_dataset, pseudo_labeled_dataset, pseudo_labeled_dataset]
    info = [info, info, info]

    return pseudo_labeled_dataset_list, info


def rank_above_k(confidence_scores, topk):
    """Sample selection criterion: Rank-above-k

    Parameters
    ----------
    confidence_scores: list[float]
    topk: int

    Returns
    -------
    numpy.ndarray(shape=(pool_size,), dtype=bool)
    int
    int
    """
    pool_size = len(confidence_scores)
    indicators = np.zeros((pool_size,))

    # Sort data according to confidence scores
    sorted_scores = [(c, i) for i, c in enumerate(confidence_scores)]
    sorted_scores = sorted(sorted_scores, key=lambda tpl: -tpl[0])

    # Select top-k highest-scoring data
    sorted_scores = sorted_scores[:topk]
    max_score = sorted_scores[0][0]
    min_score = sorted_scores[-1][0]

    # Get indices of the selected data
    selected_indices = [i for c, i in sorted_scores]
    assert len(selected_indices) == topk

    # Convert to indicators
    indicators[selected_indices] = 1
    indicators = indicators.astype(np.bool)

    return indicators, max_score, min_score


def rank_diff_k(teacher_scores, student_scores, diff_margin):
    """Sample selection criterion: Rank-diff-k

    Parameters
    ----------
    teacher_scores: list[float]
    student_scores: list[float]
    diff_margin: int

    Returns
    -------
    numpy.ndarray(shape=(pool_size,), dtype=bool)
    int
    int
    """
    pool_size = len(teacher_scores)
    assert len(student_scores) == pool_size
    indicators = np.zeros((pool_size,))

    # Sort data according to scores
    sorted_teacher_scores = [(c, i) for i, c in enumerate(teacher_scores)]
    sorted_teacher_scores = sorted(sorted_teacher_scores, key=lambda tpl: -tpl[0])

    sorted_student_scores = [(c, i) for i, c in enumerate(student_scores)]
    sorted_student_scores = sorted(sorted_student_scores, key=lambda tpl: -tpl[0])

    # Calculate ranking gaps of each data
    teacher_ranks = {data_i: rank_i for rank_i, (conf, data_i) in enumerate(sorted_teacher_scores)}
    student_ranks = {data_i: rank_i for rank_i, (conf, data_i) in enumerate(sorted_student_scores)}
    gaps = [(student_ranks[data_i] - teacher_ranks[data_i], data_i) for data_i in range(pool_size)]

    # Sort data according to the ranking gaps
    sorted_gaps = sorted(gaps, key=lambda tpl: -tpl[0]) # Maybe unnecessary

    # Select data where the ranking of teacher is higher than the rank of student by threshold `diff_margin` or more
    sorted_gaps = [(gap, i) for gap, i in sorted_gaps if gap >= diff_margin]
    if len(sorted_gaps) > 0:
        max_score = sorted_gaps[0][0]
        min_score = sorted_gaps[-1][0]
    else:
        max_score = 0
        min_score = 0

    # Get indices of the selected data
    selected_indices = [i for _, i in sorted_gaps]

    # Convert to indicators
    indicators[selected_indices] = 1
    indicators = indicators.astype(np.bool)

    # Other info.
    return indicators, max_score, min_score


##################################
# Evaluation
##################################


def parse(parser, dataset, path_pred, confidence_measure=None):
    """
    Parameters
    ----------
    parser: ArcFactoredParser or ShiftReduceParser
    dataset: numpy.ndarray
    path_pred: str
    confidence_measure: str or None, default None
    """
    assert confidence_measure in [None, "predictive_probability", "negative_entropy"]
    if confidence_measure is not None:
        f_conf = open(path_pred.replace(".arcs", ".arcs.conf"), "w")

    sw = utils.StopWatch()
    sw.start()

    with open(path_pred, "w") as f:

        for data in pyprind.prog_bar(dataset):
            # Forward and parse
            if parser.parser_type == "arcfactored":
                output = parser.parse(
                                    data=data,
                                    use_sentence_boundaries=True,
                                    use_paragraph_boundaries=True,
                                    confidence_measure=confidence_measure)

            elif parser.parser_type == "shiftreduce" or parser.parser_type == "backwardsr":
                output = parser.parse(
                                    data=data,
                                    confidence_measure=confidence_measure)
            if confidence_measure is None:
                labeled_arcs = output
            else:
                labeled_arcs, confidence = output

            # Write
            dtree = treetk.arcs2dtree(arcs=labeled_arcs)
            labeled_arcs = ["%s-%s-%s" % (x[0], x[1], x[2]) for x in dtree.tolist()]
            f.write("%s\n" % " ".join(labeled_arcs))
            if confidence_measure is not None:
                f_conf.write("%f\n" % confidence)

    if confidence_measure is not None:
        f_conf.flush()
        f_conf.close()

    sw.stop()
    speed = float(len(dataset)) / sw.get_time()
    utils.writelog("Parsed %d documents; Time: %f sec.; Speed: %f docs/sec." % (len(dataset), sw.get_time(), speed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pretrained", nargs="+", default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

