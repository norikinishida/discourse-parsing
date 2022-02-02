import torch
import torch.nn as nn

import utils

import models
import decoders


class StackPointerParser:

    def __init__(self, device, config, vocab_relation):
        """
        Parameters
        ---------
        device: str
        config: ConfigTree
        vocab_relation: dict[str, int]
        """
        self.device = device
        self.config = config
        self.model_name = config["model_name"]
        self.use_edu_head_information = config["use_edu_head_information"]

        # Initialize model
        if self.model_name == "stackpointer":
            self.model = models.StackPointerModel(device=device,
                                                  config=config,
                                                  vocab_relation=vocab_relation)
        else:
            raise Exception("Invalid model_name %s" % self.model_name)

        # Show parameter shapes
        utils.writelog("Model parameters:")
        for name, param in self.model.named_parameters():
            utils.writelog("%s = %s" % (name, tuple(param.shape)))

        # Initialize decoder
        self.decoder = decoders.StackPointerDecoder(config=config)

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
        torch.Tensor
        float
        int
        int
        """
        # Generate targets and stack inputs
        gold_arcs = data.arcs
        assert sum([1 for h,d,l in gold_arcs if h == 0]) == 1
        gold_actions, gold_relations, stack_top_indices, valences, action_masks \
            = self.decoder.simulate(edu_ids=data.edu_ids, gold_arcs=gold_arcs)
        labeling_steps = [i for i, r in enumerate(gold_relations) if r != -1]
        gold_relations = [r for r in gold_relations if r != -1]
        gold_relations = [self.model.vocab_relation[r] for r in gold_relations]

        # Tensorize targets and stack inputs
        gold_actions = torch.tensor(gold_actions, device=self.model.device) # (n_action_steps,)
        gold_relations = torch.tensor(gold_relations, device=self.model.device) # (n_edus - 1,)
        stack_top_indices = torch.tensor(stack_top_indices, device=self.model.device)
        valences = torch.tensor(valences, device=self.model.device)
        action_masks = torch.tensor(action_masks, device=self.model.device) # (n_action_steps, n_edus)
        labeling_steps = torch.tensor(labeling_steps, device=self.model.device)  # (n_edus - 1,)

        # Switch to training mode
        self.model.train()

        # Forward
        pred_actions, pred_relations = self.model.forward_for_training(
                                                data=data,
                                                stack_top_indices=stack_top_indices,
                                                valences=valences) # (action_length, n_edus), (action_length, n_edus, n_relations)

        # Compute action loss and accuracy (summed over action steps)
        pred_actions = pred_actions - 10000000.0 * action_masks # (n_action_steps, n_edus)
        loss_action = self.cross_entropy_loss(pred_actions, gold_actions)
        acc_action = (pred_actions.max(axis=1)[1] == gold_actions).to(torch.float).sum().item()

        # Compute relation loss and accuracy (summed over arcs)
        pred_relations = pred_relations[labeling_steps] # (n_edus - 1, n_edus, n_relations)
        filtered_gold_actions = gold_actions[labeling_steps] # (n_edus - 1,)
        pred_relations = pred_relations[torch.arange(0, len(labeling_steps)), filtered_gold_actions] # (n_edus - 1, n_relations)
        loss_relation = self.cross_entropy_loss(pred_relations, gold_relations)
        acc_relation = (pred_relations.max(axis=1)[1] == gold_relations).to(torch.float).sum().item()

        n_action_steps = len(gold_actions)
        n_labeling_steps = len(gold_relations)

        return loss_action, acc_action, loss_relation, acc_relation, n_action_steps, n_labeling_steps

    def parse(self, data):
        """
        Parameters
        ----------
        data: utils.DataInstance

        Returns
        -------
        list[(int, int, str)]
        """
        # Switch to inference mode
        self.model.eval()

        # Forward and decode
        parser_state = self.decoder.decode(
                            model=self.model,
                            data=data)
        labeled_arcs = parser_state.arcs

        # Postprocessing: Only Root-attachment arc can hold Root relation
        # TODO
        for i, (h, d, r) in enumerate(labeled_arcs):
            if h == 0:
                r = "<root>"
                labeled_arcs[i] = (h, d, r)
            elif h != 0 and r == "<root>":
                r = "ELABORATION"
                labeled_arcs[i] = (h, d, r)

        return labeled_arcs


