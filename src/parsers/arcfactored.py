import numpy as np
import torch
import torch.nn as nn

import utils

import models
import decoders


class ArcFactoredParser:

    def __init__(self, device, config, vocab_relation):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree
        vocab_relation: dict[str, int]
        """
        self.device = device
        self.config = config
        self.model_name = config["model_name"]
        self.use_edu_head_information = config["use_edu_head_information"]

        # Initialize model
        if self.model_name == "biaffine":
            self.model = models.BiaffineModel(device=device,
                                              config=config,
                                              vocab_relation=vocab_relation)
        else:
            raise Exception("Invalid model_name: %s" % self.model_name)

        # Show parameter shapes
        utils.writelog("Model parameters:")
        for name, param in self.model.named_parameters():
            utils.writelog("%s: %s" % (name, tuple(param.shape)))

        # Initialize decoder
        self.decoder = decoders.IncrementalEisnerDecoder()

        # Initialize loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")
        self.cross_entropy_loss_without_reduction = nn.CrossEntropyLoss(reduction="none")

    def load_model(self, path):
        """
        Parameters
        ----------
        path: str
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

    def compute_loss(self, data, labeled_parsing=True):
        """
        Parameters
        ----------
        data: utils.DataInstance
        labeled_parsing: bool, default True

        Returns
        -------
        torch.Tensor
        float
        torch.Tensor
        float
        int
        """
        # Switch to training mode
        self.model.train()

        # Forward
        arc_scores, rel_logits = self.model.forward(data=data) # (n_edus, n_edus), (n_edus, n_edus, n_relations)

        # Tensorize targets
        gold_arcs = data.arcs
        assert sum([1 for h,d,l in gold_arcs if h == 0]) == 1
        gold_arcs = sorted(gold_arcs, key=lambda arc: arc[1])
        gold_heads = [h for h,d,r in gold_arcs]
        if labeled_parsing:
            gold_relations = [self.model.vocab_relation[r] for h,d,r in gold_arcs]

        gold_heads = torch.tensor(gold_heads, device=self.model.device) # (n_edus - 1,)
        if labeled_parsing:
            gold_relations = torch.tensor(gold_relations, device=self.model.device) # (n_edus - 1,)

        # Compute attachment loss and accuracy (summed over arcs)
        arc_scores = arc_scores.permute(1, 0) # Head x Dep -> Dep x Head
        arc_scores = arc_scores[1:] # (n_edus - 1, n_edus)
        loss_attachment = self.cross_entropy_loss(arc_scores, gold_heads)
        acc_attachment = (arc_scores.max(axis=1)[1] == gold_heads).to(torch.float).sum().item()

        # Compute relation loss and accuracy (summed over arcs)
        if labeled_parsing:
            rel_logits = rel_logits.permute(1, 0, 2) # Head x Dep x Rel -> Dep x Head x Rel
            rel_logits = rel_logits[torch.arange(1, len(data.edu_ids)), gold_heads] # (n_edus - 1, n_relations)
            loss_relation = self.cross_entropy_loss(rel_logits, gold_relations)
            acc_relation = (rel_logits.max(axis=1)[1] == gold_relations).to(torch.float).sum().item()

        n_arcs = len(gold_arcs)

        if labeled_parsing:
            return loss_attachment, acc_attachment, loss_relation, acc_relation, n_arcs
        else:
            return loss_attachment, acc_attachment, n_arcs

    def compute_loss_with_mask(self, data, labeled_parsing=True):
        """
        Parameters
        ----------
        data: utils.DataInstance
        labeled_parsing: bool, default True

        Returns
        -------
        torch.Tensor
        float
        torch.Tensor
        float
        int
        """
        # Switch to training mode
        self.model.train()

        # Forward
        arc_scores, rel_logits = self.model.forward(data=data) # (n_edus, n_edus), (n_edus, n_edus, n_relations)

        # Tensorize targets
        gold_arcs = data.arcs
        assert sum([1 for h,d,l in gold_arcs if h == 0]) <= 1
        gold_arcs = sorted(gold_arcs, key=lambda arc: arc[1])
        gold_heads = [h for h,d,r in gold_arcs]
        if labeled_parsing:
            gold_relations = [self.model.vocab_relation[r] for h,d,r in gold_arcs]

        gold_heads = torch.tensor(gold_heads, device=self.model.device) # (sample_size_edu,)
        if labeled_parsing:
            gold_relations = torch.tensor(gold_relations, device=self.model.device) # (sample_size_edus,)
        annotated_deps = torch.tensor([d - 1 for h,d,r in gold_arcs], device=self.model.device) # (sample_size_edu,)

        # Compute attachment loss and accuracy (summed over arcs)
        arc_scores = arc_scores.permute(1, 0) # Head x Dep -> Dep x Head
        arc_scores = arc_scores[1:] # (n_edus - 1, n_edus)
        arc_scores = arc_scores[annotated_deps] # (sample_size_edu, n_edus)
        loss_attachment = self.cross_entropy_loss(arc_scores, gold_heads)
        acc_attachment = (arc_scores.max(axis=1)[1] == gold_heads).to(torch.float).sum().item()

        # Compute relation loss and accuracy (summed over arcs)
        if labeled_parsing:
            rel_logits = rel_logits.permute(1, 0, 2) # Head x Dep x Rel -> Dep x Head x Rel
            dummy_gold_heads = np.zeros((len(data.edu_ids) - 1,), dtype=np.int64) # (n_edus - 1,)
            for h, d, r in gold_arcs:
                dummy_gold_heads[d - 1] = h
            dummy_gold_heads = torch.tensor(dummy_gold_heads, device=self.model.device)
            rel_logits = rel_logits[torch.arange(1, len(data.edu_ids)), dummy_gold_heads] # (n_edus - 1, n_relations)
            rel_logits = rel_logits[annotated_deps] # (sample_size_edu, n_relations)
            loss_relation = self.cross_entropy_loss(rel_logits, gold_relations)
            acc_relation = (rel_logits.max(axis=1)[1] == gold_relations).to(torch.float).sum().item()

        n_arcs = len(gold_arcs)

        if labeled_parsing:
            return loss_attachment, acc_attachment, loss_relation, acc_relation, n_arcs
        else:
            return loss_attachment, acc_attachment, n_arcs

    def parse(self, data, use_sentence_boundaries, use_paragraph_boundaries, confidence_measure=None, confidence_reduction=True):
        """
        Parameters
        ----------
        data: utils.DataInstance
        use_sentence_boundaries: bool
        use_paragraph_boundaries: bool
        confidence_measure: str or None, default None
        confidence_reduction: bool, default True

        Returns
        -------
        list[(int, int, str)]
        float or list[int]
        """
        # Switch to inference mode
        self.model.eval()

        # Forward
        arc_scores, rel_logits_tensor = self.model.forward(data=data) # (n_edus, n_edus), (n_edus, n_edus, n_relations)

        # Decode
        edu_ids = data.edu_ids
        sentence_boundaries = data.sentence_boundaries
        paragraph_boundaries = data.paragraph_boundaries
        unlabeled_arcs = self.decoder.decode(
                                arc_scores=arc_scores.cpu().numpy(),
                                edu_ids=edu_ids,
                                #
                                sentence_boundaries=sentence_boundaries,
                                paragraph_boundaries=paragraph_boundaries,
                                use_sentence_boundaries=use_sentence_boundaries,
                                use_paragraph_boundaries=use_paragraph_boundaries) # list of (int, int)
        unlabeled_arcs = sorted(unlabeled_arcs, key=lambda arc: arc[1])
        pred_heads = torch.tensor([h for h,d in unlabeled_arcs], device=self.model.device) # (n_edus - 1,)

        # Label relations
        rel_logits_tensor = rel_logits_tensor.permute(1, 0, 2) # Head x Dep x Rel -> Dep x Head x Rel
        rel_logits_mat = rel_logits_tensor[torch.arange(1, len(edu_ids)), pred_heads] # (n_edus - 1, n_relations)

        # Postprocessing: Only Root-attachment arc can hold Root relation
        rel_logits_mat_cpu = rel_logits_mat.cpu().numpy()
        root_rel_column = self.model.vocab_relation["<root>"] # Column index
        root_arc_row = [i for i, h in enumerate(pred_heads) if h == 0] # Row indices
        assert len(root_arc_row) == 1 # Single Root
        root_arc_row = root_arc_row[0] # Row index
        mask = np.zeros_like(rel_logits_mat_cpu)
        mask[:, root_rel_column] = 1.0
        mask[root_arc_row] = 1.0 - mask[root_arc_row]
        # Example:
        #   n_edus = 5, root_rel_column = 2, root_arc_row = 1
        #   mask = [[0, 0, 1, 0, 0],
        #           [1, 1, 0, 1, 1],
        #           [0, 0, 1, 0, 0],
        #           [0, 0, 1, 0, 0],
        #           [0, 0, 1, 0, 0]]
        rel_logits_mat_cpu = rel_logits_mat_cpu - 1000.0 * mask

        pred_relations = np.argmax(rel_logits_mat_cpu, axis=1) # (n_edus - 1,)
        pred_relations = [self.model.ivocab_relation[r] for r in pred_relations] # list[str]
        labeled_arcs = [(h,d,r) for (h,d),r in zip(unlabeled_arcs, pred_relations)] # list[(int, int, str)]

        if confidence_measure is None:
            return labeled_arcs

        # Measure confidence
        if confidence_measure == "predictive_probability":
            arc_scores = arc_scores.permute(1, 0) # Head x Dep -> Dep x Head
            conf_att = torch.softmax(arc_scores, axis=1)[torch.arange(1, len(edu_ids)), pred_heads] # (n_edus - 1,)
            conf_rel = torch.max(torch.softmax(rel_logits_mat, axis=1), axis=1)[0] # (n_edus - 1,)
            if confidence_reduction:
                conf_att = conf_att.mean().item() # float
                conf_rel = conf_rel.mean().item() # float
                confidence = 0.5 * (conf_att + conf_rel)
            else:
                conf_att = conf_att.cpu().numpy() # (n_edus - 1,)
                conf_rel = conf_rel.cpu().numpy() # (n_edus - 1,)
                confidence_list = 0.5 * (conf_att + conf_rel)
        elif confidence_measure == "negative_entropy":
            arc_scores = arc_scores.permute(1, 0) # Head x Dep -> Dep x Head
            arc_scores = arc_scores[1:] # (n_edus - 1, n_edus)
            neg_ent_att = torch.softmax(arc_scores, axis=1) * torch.log_softmax(arc_scores, axis=1) # (n_edus - 1, n_edus)
            neg_ent_rel = torch.softmax(rel_logits_mat, axis=1) * torch.log_softmax(rel_logits_mat, axis=1) # (n_edus - 1, n_relations)
            neg_ent_att = neg_ent_att.sum(axis=1) # (n_edus - 1,)
            neg_ent_rel = neg_ent_rel.sum(axis=1) # (n_edus - 1,)
            if confidence_reduction:
                neg_ent_att = neg_ent_att.mean().item() # float
                neg_ent_rel = neg_ent_rel.mean().item() # float
                confidence = 0.5 * (neg_ent_att + neg_ent_rel) # float
            else:
                neg_ent_att = neg_ent_att.cpu().numpy() # (n_edus - 1,)
                neg_ent_rel = neg_ent_rel.cpu().numpy() # (n_edus - 1,)
                confidence_list = 0.5 * (neg_ent_att + neg_ent_rel) # (n_edus - 1,)
        elif confidence_measure == "oracle_negative_loss":
            # Only in the case of active learning with oracle information
            # e.g., high loss value represents the low confidence
            # Tensorize oracle targets
            gold_arcs = data.stored_gold_arcs
            gold_arcs = sorted(gold_arcs, key=lambda arc: arc[1])
            gold_heads = torch.tensor([h for h,d,r in gold_arcs], device=self.model.device) # (n_edus - 1,)
            gold_relations = torch.tensor([self.model.vocab_relation[r] for h,d,r in gold_arcs], device=self.model.device) # (n_edus - 1,)
            # Compute loss
            arc_scores = arc_scores.permute(1, 0) # Head x Dep -> Dep x Head
            arc_scores = arc_scores[1:] # (n_edus - 1, n_edus)
            rel_logits_mat = rel_logits_tensor[torch.arange(1, len(edu_ids)), gold_heads] # (n_edus - 1, n_relations)
            if confidence_reduction:
                loss_attachment = self.cross_entropy_loss(arc_scores, gold_heads)
                loss_relation = self.cross_entropy_loss(rel_logits_mat, gold_relations)
                confidence = -0.5 * (float(loss_attachment) + float(loss_relation)) / len(gold_arcs) # float
            else:
                loss_attachment = self.cross_entropy_loss_without_reduction(arc_scores, gold_heads).cpu().numpy() # (n_edus - 1,)
                loss_relation = self.cross_entropy_loss_without_reduction(rel_logits_mat, gold_relations).cpu().numpy() # (n_edus - 1,)
                confidence_list = -0.5 * (loss_attachment + loss_relation) # (n_edus - 1,)
        else:
            raise Exception("Never occur.")

        if confidence_reduction:
            return labeled_arcs, confidence
        else:
            return labeled_arcs, confidence_list



