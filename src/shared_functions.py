import os
import random

import numpy as np
import torch
from torch.optim import Adam
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR

import utils
import treetk


####################################
# Dataset processing
####################################


def generate_rstdt_dev_set_for_dep(train_dataset, n_dev, seed, output_path):
    """
    Parameters
    ----------
    dataset : numpy.ndarray
    n_dev: int
    seed: int
    output_path: str

    Returns
    -------
    numpy.ndarray
    numpy.ndarray
    """
    train_dataset, dev_dataset = utils.split_dataset(dataset=train_dataset, n_dev=n_dev, seed=seed)
    with open(output_path, "w") as f:
        for data in dev_dataset:
            line = ["%s-%s-%s" % (h, d, r) for h, d, r in data.arcs]
            line = sorted(line, key=lambda x: x[1])
            line = " ".join(line)
            f.write("%s\n" % line)
    utils.writelog("Saved RST-DT dev set (size=%d, seed=%d) to %s" % (n_dev, seed, output_path))
    return train_dataset, dev_dataset


def is_projective(arcs):
    """
    Parameters
    ----------
    arcs: list[(int, int, str)]

    Returns
    -------
    bool
    """
    n_arcs = len(arcs)
    for i in range(n_arcs):
        ai, bi, _ = arcs[i]
        ai, bi = sorted([ai, bi])
        for j in range(i+1, n_arcs):
            aj, bj, _ = arcs[j]
            aj, bj = sorted([aj, bj])
            if ai < aj < bi < bj or aj < ai < bj < bi:
                return False
    return True


def is_dag(arcs, n_nodes):
    """
    Parameters
    ----------
    arcs: list[(int, int, str)]

    Returns
    -------
    bool
    """
    head2deps = [[] for _ in range(n_nodes)]
    for head, dep, _ in arcs:
        head2deps[head].append(dep)

    # Keep track of whether a node is discovered or not
    discovered = [False] * n_nodes # Indicators for whether each node is discovered or not

    # Keep track of the departure time of a node
    departure = [None] * n_nodes # Departure time of each node

    time = 0

    # Perform DFS traversal from all undiscovered nodes
    for head in range(n_nodes):
        if not discovered[head]:
            time = DFS(head2deps=head2deps, head=head, discovered=discovered, departure=departure, time=time)

    # Check if the given directed graph is DAG or not
    for head in range(n_nodes):
        for dep in head2deps[head]:
            # If the  departure time of the dep is greater than or equal to the departure time of the head, they form a cycle
            if departure[head] <= departure[dep]:
                return False
    return True


def DFS(head2deps, head, discovered, departure, time):
    """
    Parameters
    ----------
    head2deps: list[list[int]]
    head: int
    discovered: list[bool]
    departure: list[int]
    time: int

    Returns
    -------
    int
    """
    # Mark the current node as discovered
    discovered[head] = True

    # Do for every arc head->dep
    for dep in head2deps[head]:
        if not discovered[dep]:
            time = DFS(head2deps, dep, discovered, departure, time)

    # Set departure time of the head
    departure[head] = time
    time = time + 1

    return time


####################################
# Optimization
####################################


def get_optimizer(model, config):
    """
    Parameters
    ----------
    model: Model
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
            "lr": config["bert_learning_rate"],
            "weight_decay": config["adam_weight_decay"],
        },
        {
            "params": [p for n,p in bert_param if any(nd in n for nd in no_decay)],
            "lr": config["bert_learning_rate"],
            "weight_decay": 0.0,
        }
    ]
    optimizers = [
        AdamW(grouped_bert_param, lr=config["bert_learning_rate"], eps=config["adam_eps"]),
        Adam(model.get_params()[1], lr=config["task_learning_rate"], eps=config["adam_eps"], weight_decay=0)
    ]
    return optimizers


def get_scheduler(optimizers, total_update_steps, warmup_steps):
    """
    Parameters
    ----------
    optimizers: [transformers.AdamW, torch.optim.Adam]
    total_update_steps: int
    warmup_steps: int

    Returns
    -------
    list[torch.optim.lr_scheduler.LambdaLR]
    """
    # Only warm up bert lr
    def lr_lambda_bert(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
               )

    def lr_lambda_task(current_step):
        return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

    schedulers = [
        LambdaLR(optimizers[0], lr_lambda_bert),
        LambdaLR(optimizers[1], lr_lambda_task)
    ]
    return schedulers


####################################
# Evaluation
####################################


def save_prediction_as_scidtb_format(dataset, path_pred):
    """
    Parameters
    ----------
    dataset: numpy.ndarray
    path_pred: str
    """
    # output_dir = os.path.join(
    #                     os.path.dirname(path_pred),
    #                     utils.get_basename_without_ext(path_pred))
    output_dir = path_pred + ".dep"
    utils.mkdir(output_dir)

    batch_arcs = utils.read_lines(path_pred, process=lambda line: treetk.hyphens2arcs(line.split()))
    assert len(batch_arcs) == len(dataset)

    for data_i in range(len(dataset)):
        filename = dataset[data_i].id
        edus = dataset[data_i].edus
        arcs = batch_arcs[data_i]
        output = transform_prediction_to_scidtb_format(edus=edus, arcs=arcs)
        utils.write_json(os.path.join(output_dir, "%s.dep" % filename), output)


def transform_prediction_to_scidtb_format(edus, arcs):
    """
    Parameters
    ----------
    edus: list[list[str]]
    arcs: list[(int, int, str)]

    Returns
    -------
    dict[str, list[str, any]]
    """
    output = {"root": []}
    output["root"].append({"id": 0,
                           "parent": -1,
                           "text": "ROOT",
                           "relation": "null"})
    for head, dep, rel in arcs:
        assert dep != 0
        edu_info = {"id": dep,
                    "parent": head,
                    "text": " ".join(edus[dep]),
                    "relation": rel}
        output["root"].append(edu_info)
    return output


####################################
# Debug
####################################


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    utils.writelog("Set random seed: %d" % seed)

