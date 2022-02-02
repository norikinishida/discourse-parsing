import torch
import torch.nn as nn

import utils

import models
import decoders


class ShiftReduceParser:

    def __init__(self, device, config, vocab_relation):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree
        vocab_relation: dict[int, str]
        """
        self.device = device
        self.config = config
        self.model_name = config["model_name"]
        self.use_edu_head_information = config["use_edu_head_information"]
        self.reverse_order = config["reverse_order"]

        # Initialize model
        if self.model_name == "arcstandard":
            self.model = models.ArcStandardModel(device=device,
                                                 config=config,
                                                 vocab_relation=vocab_relation)
        else:
            raise Exception("Invalid model_name: %s" % self.model_name)

        # Show parameter shapes
        utils.writelog("Model parameters:")
        for name, param in self.model.named_parameters():
            utils.writelog("%s: %s" % (name, tuple(param.shape)))

        # Initialize decoder
        if self.model_name == "arcstandard":
            self.decoder = decoders.ArcStandardDecoder()
        else:
            raise Exception("Invalid model_name: %s" % self.model_name)

        # Initialize loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")

    def load_model(self, path):
        """
        Parameters
        ----------
        path: str

        Returns
        -------
        torch.nn.Module
        """
        self.model.load_state_dict(torch.load(path, map_location=torch.device("cpu")), strict=False)

    def save_model(self, path):
        """
        Parameters
        ----------
        path: str
        """
        torch.save(self.model.state_dict(), path)

    def to_gpu(self, device):
        """
        Parameters
        ----------
        device: str
        """
        self.model.to(device)

    def compute_loss(self, data):
        """
        Parameters
        ----------
        data: utils.DataInstance

        Returns
        -------
        torch.Tensor
        float
        int
        """
        # Prepare targets
        gold_arcs = data.arcs
        assert sum([1 for h,d,l in gold_arcs if h == 0]) == 1

        # Switch to training mode
        self.model.train()

        # Forward and decode
        edu_ids = data.edu_ids
        if not self.reverse_order:
            _, pred_actions, gold_actions = self.decoder.decode(
                                                model=self.model,
                                                edu_ids=edu_ids,
                                                data=data,
                                                #
                                                gold_arcs=gold_arcs)
        else:
            _, pred_actions, gold_actions = self.decoder.decode(
                                                model=self.model,
                                                edu_ids=edu_ids[0:1] + edu_ids[1:][::-1],
                                                data=data,
                                                #
                                                gold_arcs=gold_arcs)
        assert len(pred_actions) == len(gold_actions)
        pred_actions = torch.cat(pred_actions, axis=0) # (n_action_steps, n_actions)

        # Tensorize targets
        gold_actions = [self.model.vocab_action[a] for a in gold_actions] # (n_action_steps,)
        gold_actions = torch.tensor(gold_actions, device=self.model.device) # (n_action_steps,)

        # Compute loss and accuracy (summed over action steps)
        loss = self.cross_entropy_loss(pred_actions, gold_actions)
        acc = (pred_actions.max(axis=1)[1] == gold_actions).to(torch.float).sum().item()

        n_action_steps = len(gold_actions)

        return loss, acc, n_action_steps

    def parse(self, data, confidence_measure=None):
        """
        Parameters
        ----------
        data: utils.DataInstance
        confidence_measure: str or None, default None

        Returns
        -------
        list[(int, int, str)]
        float
        """
        # Switch to inference mode
        self.model.eval()

        # Forward and decode
        edu_ids = data.edu_ids
        if not self.reverse_order:
            parser_state, _, _ = self.decoder.decode(
                                    model=self.model,
                                    edu_ids=edu_ids,
                                    data=data,
                                    confidence_measure=confidence_measure)
        else:
            parser_state, _, _ = self.decoder.decode(
                                    model=self.model,
                                    edu_ids=edu_ids[0:1] + edu_ids[1:][::-1],
                                    data=data,
                                    confidence_measure=confidence_measure)
        labeled_arcs = parser_state.arcs # list[(int, int, str)]

        if confidence_measure is None:
            return labeled_arcs

        confidence = parser_state.confidence
        return labeled_arcs, confidence


