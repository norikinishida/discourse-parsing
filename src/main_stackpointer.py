import argparse
import os

import jsonlines
import numpy as np
import torch
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

    ##################
    # Paths
    ##################

    base_dir = "stackpointer.%s" % config_name

    utils.mkdir(os.path.join(config["results"], base_dir))

    # Log file
    path_log = None
    if actiontype == "train":
        path_log = os.path.join(config["results"], base_dir, prefix + ".training.log")
    elif actiontype == "evaluate":
        path_log = os.path.join(config["results"], base_dir, prefix + ".evaluation.log")

    # Training loss and etc.
    path_train_jsonl = os.path.join(config["results"], base_dir, prefix + ".training.jsonl")

    # Model snapshot
    path_snapshot = os.path.join(config["results"], base_dir, prefix + ".model")

    # Validation outputs and scores
    path_valid_pred = os.path.join(config["results"], base_dir, prefix + ".validation.arcs")
    path_valid_jsonl = os.path.join(config["results"], base_dir, prefix + ".validation.jsonl")

    # Evaluation outputs and scores
    path_test_pred = os.path.join(config["results"], base_dir, prefix + ".evaluation.arcs")
    path_test_json = os.path.join(config["results"], base_dir, prefix + ".evaluation.json")

    # Gold data for validation and evaluation
    if config["dataset_name"] == "rstdt":
        path_valid_gold = os.path.join(config["caches-dep"], "rstdt.dev.gold.arcs")
        path_test_gold = os.path.join(config["caches-dep"], "rstdt.test.gold.arcs")
    elif config["dataset_name"] == "scidtb":
        path_valid_gold = os.path.join(config["caches-dep"], "scidtb.dev-gold.gold.arcs")
        path_test_gold = os.path.join(config["caches-dep"], "scidtb.test-gold.gold.arcs")
    elif config["dataset_name"] == "stac":
        path_valid_gold = os.path.join(config["caches-dep"], "stac.dev.gold.arcs")
        path_test_gold = os.path.join(config["caches-dep"], "stac.test.gold.arcs")
    elif config["dataset_name"] == "molweni":
        path_valid_gold = os.path.join(config["caches-dep"], "molweni.dev.gold.arcs")
        path_test_gold = os.path.join(config["caches-dep"], "molweni.test.gold.arcs")
    else:
        raise Exception("Never occur.")

    utils.set_logger(path_log)

    utils.writelog("device: %s" % device)
    utils.writelog("config_name: %s" % config_name)
    utils.writelog("prefix: %s" % prefix)
    utils.writelog("actiontype: %s" % actiontype)

    utils.writelog(utils.pretty_format_dict(config))

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_json: %s" % path_test_json)

    ##################
    # Datasets
    ##################

    sw.start("data")

    if config["dataset_name"] == "rstdt":
        train_dataset = np.load(os.path.join(config["caches-dep"], "rstdt.train.bert-base-cased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-dep"], "rstdt.test.bert-base-cased.npy"), allow_pickle=True)
        train_dataset, dev_dataset = shared_functions.generate_rstdt_dev_set_for_dep(train_dataset=train_dataset,
                                                                                     n_dev=30, seed=7777,
                                                                                     output_path=path_valid_gold)
        vocab_relation = utils.read_vocab(os.path.join(config["caches-dep"], "rstdt.dependency_relations.vocab.txt"))
    elif config["dataset_name"] == "scidtb":
        train_dataset = np.load(os.path.join(config["caches-dep"], "scidtb.train-.scibert_scivocab_uncased.npy"), allow_pickle=True)
        dev_dataset = np.load(os.path.join(config["caches-dep"], "scidtb.dev-gold.scibert_scivocab_uncased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-dep"], "scidtb.test-gold.scibert_scivocab_uncased.npy"), allow_pickle=True)
        vocab_relation = utils.read_vocab(os.path.join(config["caches-dep"], "scidtb.relations.vocab.txt"))
    elif config["dataset_name"] == "stac":
        train_dataset = np.load(os.path.join(config["caches-dep"], "stac.train.bert-base-cased.npy"), allow_pickle=True)
        dev_dataset = np.load(os.path.join(config["caches-dep"], "stac.dev.bert-base-cased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-dep"], "stac.test.bert-base-cased.npy"), allow_pickle=True)
        vocab_relation = utils.read_vocab(os.path.join(config["caches-dep"], "stac.relations.vocab.txt"))
    elif config["dataset_name"] == "molweni":
        train_dataset = np.load(os.path.join(config["caches-dep"], "molweni.train.bert-base-cased.npy"), allow_pickle=True)
        dev_dataset = np.load(os.path.join(config["caches-dep"], "molweni.dev.bert-base-cased.npy"), allow_pickle=True)
        test_dataset = np.load(os.path.join(config["caches-dep"], "molweni.test.bert-base-cased.npy"), allow_pickle=True)
        vocab_relation = utils.read_vocab(os.path.join(config["caches-dep"], "molweni.relations.vocab.txt"))
    else:
        raise Exception("Never occur.")

    # Exclude non-projective data in training set
    count = len(train_dataset)
    train_dataset = utils.filter_dataset(train_dataset, condition=lambda data: shared_functions.is_projective(data.arcs))
    utils.writelog("Excluded %d (= %d - %d) non-projective data in the training set" % (count - len(train_dataset), count, len(train_dataset)))

    # Exclude cyclic data in the training set
    count = len(train_dataset)
    train_dataset = utils.filter_dataset(train_dataset, condition=lambda data: shared_functions.is_dag(arcs=data.arcs, n_nodes=len(data.edus)))
    utils.writelog("Excluded %d (= %d - %d) cyclic data in the training set" % (count - len(train_dataset), count, len(train_dataset)))

    utils.writelog("Number of training data: %d" % len(train_dataset))
    utils.writelog("Number of validation data: %d" % len(dev_dataset))
    utils.writelog("Number of test data: %d" % len(test_dataset))

    sw.stop("data")
    utils.writelog("Loaded the corpus. %f [sec.]" % sw.get_time("data"))

    ##################
    # Parser
    ##################

    parser = parsers.StackPointerParser(device=device,
                                        config=config,
                                        vocab_relation=vocab_relation)

    # Load pre-trained parameters
    if actiontype != "train":
        parser.load_model(path=path_snapshot)
        utils.writelog("Loaded model from %s" % path_snapshot)

    parser.to_gpu(device=device)

    ##################
    # Action
    ##################

    if actiontype == "train":
        train(
            config=config,
            parser=parser,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            path_train_jsonl=path_train_jsonl,
            path_snapshot=path_snapshot,
            path_valid_pred=path_valid_pred,
            path_valid_gold=path_valid_gold,
            path_valid_jsonl=path_valid_jsonl)

    elif actiontype == "evaluate":
        with torch.no_grad():
            parse(
                parser=parser,
                dataset=test_dataset,
                path_pred=path_test_pred)
            scores = metrics.attachment_scores(
                        pred_path=path_test_pred,
                        gold_path=path_test_gold)
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            scores["UUAS"] *= 100.0
            scores["RA"] *= 100.0
            utils.write_json(path_test_json, scores)
            utils.writelog(utils.pretty_format_dict(scores))
            shared_functions.save_prediction_as_scidtb_format(dataset=test_dataset, path_pred=path_test_pred)

    utils.writelog("path_log: %s" % path_log)
    utils.writelog("path_train_jsonl: %s" % path_train_jsonl)
    utils.writelog("path_snapshot: %s" % path_snapshot)
    utils.writelog("path_valid_pred: %s" % path_valid_pred)
    utils.writelog("path_valid_gold: %s" % path_valid_gold)
    utils.writelog("path_valid_jsonl: %s" % path_valid_jsonl)
    utils.writelog("path_test_pred: %s" % path_test_pred)
    utils.writelog("path_test_gold: %s" % path_test_gold)
    utils.writelog("path_test_json: %s" % path_test_json)
    utils.writelog("Done.")
    sw.stop("main")
    utils.writelog("Time: %f min." % sw.get_time("main", minute=True))


####################################
# Training
####################################


def train(config,
          parser,
          train_dataset,
          dev_dataset,
          path_train_jsonl,
          path_snapshot,
          path_valid_pred,
          path_valid_gold,
          path_valid_jsonl):
    """
    Parameters
    ----------
    config: ConfigTree
    parser: StackPointerParser
    train_dataset: numpy.ndarray
    dev_dataset: numpy.ndarray
    path_train_jsonl: str
    path_snapshot: str
    path_valid_pred: str
    path_valid_gold: str
    path_valid_jsonl: str

    """
    # Get optimizers and schedulers
    n_train = len(train_dataset)
    max_epoch = config["max_epoch"]
    batch_size = config["batch_size"]
    total_update_steps = n_train * max_epoch // batch_size
    warmup_steps = int(total_update_steps * config["warmup_ratio"])

    optimizers = shared_functions.get_optimizer(model=parser.model, config=config)
    schedulers = shared_functions.get_scheduler(optimizers=optimizers, total_update_steps=total_update_steps, warmup_steps=warmup_steps)

    utils.writelog("*********************Training***********************")
    utils.writelog("n_train: %d" % n_train)
    utils.writelog("max_epoch: %d" % max_epoch)
    utils.writelog("batch_size: %d" % batch_size)
    utils.writelog("total_update_steps: %d" % total_update_steps)
    utils.writelog("warmup_steps: %d" % warmup_steps)

    writer_train = jsonlines.Writer(open(path_train_jsonl, "w"), flush=True)
    writer_valid = jsonlines.Writer(open(path_valid_jsonl, "w"), flush=True)
    bestscore_holder = utils.BestScoreHolder(scale=1.0)
    bestscore_holder.init()
    step = 0
    bert_param, task_param = parser.model.get_params()

    ##################
    # Initial validation phase
    ##################

    with torch.no_grad():
        parse(
            parser=parser,
            dataset=dev_dataset,
            path_pred=path_valid_pred)
        scores = metrics.attachment_scores(
                    pred_path=path_valid_pred,
                    gold_path=path_valid_gold)
        scores["LAS"] *= 100.0
        scores["UAS"] *= 100.0
        scores["UUAS"] *= 100.0
        scores["RA"] *= 100.0
        scores["epoch"] = 0
        writer_valid.write(scores)
        utils.writelog(utils.pretty_format_dict(scores))

    bestscore_holder.compare_scores(scores["LAS"], 0)

    # Save the model
    parser.save_model(path=path_snapshot)
    utils.writelog("Saved model to %s" % path_snapshot)

    ##################
    # /Initial validation phase
    ##################

    ##################
    # Training-and-validation loops
    ##################

    for epoch in range(1, max_epoch + 1):

        ##################
        # Training phase
        ##################

        perm = np.random.permutation(n_train)

        for instance_i in range(0, n_train, batch_size):

            ##################
            # Mini batch
            ##################

            # Initialize losses
            loss_action = 0.0
            acc_action = 0.0
            loss_relation = 0.0
            acc_relation = 0.0
            actual_batchsize = 0
            actual_total_actions = 0
            actual_total_arcs = 0

            for data in train_dataset[perm[instance_i:instance_i+batch_size]]:

                ##################
                # One data
                ##################

                # Forward and compute losses
                one_loss_action, \
                    one_acc_action, \
                    one_loss_relation, \
                    one_acc_relation, \
                    n_action_steps, \
                    n_labeling_steps = parser.compute_loss(data=data)

                # Accumulate the losses
                loss_action = loss_action + one_loss_action
                acc_action = acc_action + one_acc_action
                loss_relation = loss_relation + one_loss_relation
                acc_relation = acc_relation + one_acc_relation
                actual_batchsize += 1
                actual_total_actions += n_action_steps
                actual_total_arcs += n_labeling_steps

                ##################
                # /One data
                ##################

            # Average the losses
            actual_batchsize = float(actual_batchsize)
            actual_total_actions = float(actual_total_actions)
            actual_total_arcs = float(actual_total_arcs)
            loss_action = loss_action / actual_total_actions
            acc_action = acc_action / actual_total_actions
            loss_relation = loss_relation / actual_total_arcs
            acc_relation = acc_relation / actual_total_arcs

            # Merge the losses
            loss = loss_action + loss_relation

            # Backward
            parser.model.zero_grad()
            loss.backward()

            # Update
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(bert_param, config["max_grad_norm"])
                torch.nn.utils.clip_grad_norm_(task_param, config["max_grad_norm"])
            for optimizer in optimizers:
                optimizer.step()
            for scheduler in schedulers:
                scheduler.step()

            step += 1

            # Write log
            loss_action_data = float(loss_action.cpu())
            loss_relation_data = float(loss_relation.cpu())
            out = {"step": step,
                   "epoch": epoch,
                   "progress": "%d/%d" % (instance_i + actual_batchsize, n_train),
                   "progress_ratio": float(instance_i + actual_batchsize) / n_train * 100.0,
                   "action_loss": loss_action_data,
                   "action_accuracy": acc_action * 100.0,
                   "relation_loss": loss_relation_data,
                   "relation_accuracy": acc_relation * 100.0,
                   "max_valid_las": bestscore_holder.best_score,
                   "patience": bestscore_holder.patience}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))

            ##################
            # /Mini batch
            ##################

        ##################
        # /Training phase
        ##################

        ##################
        # Validation phase
        ##################

        with torch.no_grad():
            parse(
                parser=parser,
                dataset=dev_dataset,
                path_pred=path_valid_pred)
            scores = metrics.attachment_scores(
                        pred_path=path_valid_pred,
                        gold_path=path_valid_gold)
            scores["LAS"] *= 100.0
            scores["UAS"] *= 100.0
            scores["UUAS"] *= 100.0
            scores["RA"] *= 100.0
            scores["epoch"] = epoch
            writer_valid.write(scores)
            utils.writelog(utils.pretty_format_dict(scores))

        did_update = bestscore_holder.compare_scores(scores["LAS"], epoch)
        utils.writelog("[Step %d] Max validation LAS = %f" % (step, bestscore_holder.best_score))

        # Save the model?
        if did_update:
            parser.save_model(path=path_snapshot)
            utils.writelog("Saved model to %s" % path_snapshot)

        ##################
        # /Validation phase
        ##################

    ##################
    # /Training-and-validation loops
    ##################

    writer_train.close()
    writer_valid.close()


####################################
# Evaluation
####################################


def parse(parser, dataset, path_pred):
    """
    Parameters
    ----------
    parser: StackPointerParser
    dataset: numpy.ndarray
    path_pred: str
    """
    sw = utils.StopWatch()
    sw.start()

    with open(path_pred, "w") as f:

        for data in pyprind.prog_bar(dataset):
            # Forward and parse
            labeled_arcs = parser.parse(data=data)

            # Write
            dtree = treetk.arcs2dtree(arcs=labeled_arcs)
            labeled_arcs = ["%s-%s-%s" % (x[0],x[1],x[2]) for x in dtree.tolist()]
            f.write("%s\n" % " ".join(labeled_arcs))

    sw.stop()
    speed = float(len(dataset)) / sw.get_time()
    utils.writelog("Parsed %d documents; Time: %f sec.; Speed: %f docs/sec." % (len(dataset), sw.get_time(), speed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

