import copy
import random

import numpy as np
import torch


OPEN = 0
CLOSE = 1
CHOSEN = -2


class StackPointerParserState(object):

    def __init__(self, input):
        """
        Parameters
        ----------
        input: list[int]
        """
        assert input[0] == 0 # Root
        self.stack = [0]
        self.pre_mask = [OPEN for _ in range(len(input))]
        self.pre_mask[0] = CLOSE
        self.span_heads = np.asarray([-1 for _ in range(len(input))])
        self.arcs = []
        self.action_history = []

    def get_mask_copy(self):
        """
        Returns
        -------
        list[int]
        """
        mask = copy.deepcopy(self.pre_mask)
        return mask

    def __str__(self):
        return "StackPointerParserState(stack = %s; arcs = %s; action_history = %s)" \
                % (self.stack, self.arcs, self.action_history)


class StackPointerDecoder(object):

    def __init__(self, config):
        """
        Parameters
        ----------
        config: ConfigTree
        """
        self.target_order = config["target_order"]
        assert self.target_order in ["inside_out_left_then_right", "inside_out_right_then_left", "random"]

    def simulate(self, edu_ids, gold_arcs):
        """
        Parameters
        ----------
        edu_ids: list[int]
        gold_arcs: list[(int, int, str)]

        Returns
        -------
        list[int]
        list[str]
        list[int]
        list[int]
        list[list[int]]
        """
        gold_actions = []
        gold_relations = []
        stack_top_indices = []
        valences = []
        action_masks = []

        head2deps = {h: [] for h in range(len(edu_ids))}
        for h,d,r in gold_arcs:
            head2deps[h].append(d)
        for h in range(len(edu_ids)):
            if self.target_order == "inside_out_left_then_right":
                left = [d for d in head2deps[h] if d < h]
                right = [d for d in head2deps[h] if h < d]
                left = sorted(left, key=lambda x: -x)
                right = sorted(right, key=lambda x: x)
                new_deps = left + right
            elif self.target_order == "inside_out_right_then_left":
                left = [d for d in head2deps[h] if d < h]
                right = [d for d in head2deps[h] if h < d]
                left = sorted(left, key=lambda x: -x)
                right = sorted(right, key=lambda x: x)
                new_deps = right + left
            elif self.target_order == "random":
                deps = head2deps[h]
                new_deps = random.sample(deps, len(deps))
            else:
                raise Exception("Never occur.")
            head2deps[h] = new_deps
        dep2rel = {}
        for h,d,r in gold_arcs:
            dep2rel[d] = r

        parser_state = StackPointerParserState(input=edu_ids)
        head2vals = {h: 0 for h in range(len(edu_ids))}
        # while not self.is_finished(parser_state):
        while len(parser_state.stack) > 0:
            head = parser_state.stack[-1]
            stack_top_indices.append(head)
            valences.append(head2vals[head])
            mask = parser_state.get_mask_copy()
            assert mask[head] == CLOSE
            mask[head] = OPEN
            action_masks.append(mask)
            deps = head2deps[head]
            if len(deps) == 0:
                gold_actions.append(head)
                gold_relations.append(-1)
                parser_state.stack.pop()
            else:
                dep = deps.pop(0)
                gold_actions.append(dep)
                rel = dep2rel[dep]
                gold_relations.append(rel)
                parser_state.stack.append(dep)
                parser_state.pre_mask[dep] = CLOSE
                head2vals[head] += 1

        return gold_actions, gold_relations, stack_top_indices, valences, action_masks

    def decode(self,
               model,
               edu_ids,
               edus,
               segments,
               segments_id,
               segments_mask,
               edu_begin_indices,
               edu_end_indices,
               edu_head_indices,
               #
               sentence_boundaries,
               paragraph_boundaries,
               use_sentence_boundaries,
               use_paragraph_boundaries):
        """
        Parameters
        ----------
        model: StackPointerModel
        edu_ids: list[int]
        edus: list[list[str]]
        segments: list[list[str]]
        segments_id: Tensor(shape=(n_segments, max_seg_len), dtype=torch.long)
        segments_mask: Tensor(shape=(n_segments, max_seg_len), dtype=torch.long)
        edu_begin_indices: Tensor(shape=(n_edus,), dtype=torch.long)
        edu_end_indices: Tensor(shape=(n_edus,), dtype=torch.long)
        edu_head_indices: Tensor(shape=(n_edus,), dtype=torch.long) or None
        sentence_boundaries: list[(int, int)]
        paragraph_boundaries: list[(int, int)]
        use_sentence_boundaries: bool
        use_paragraph_boundaries: bool

        Returns
        -------
        StackPointerParserState
        """
        assert edu_ids[0] == 0 # ROOT

        parser_state = StackPointerParserState(input=edu_ids)
        head2vals = {h: 0 for h in range(len(edu_ids))}

        # Encoder
        edu_vectors = model.encode(edus=edus,
                                   segments=segments,
                                   segments_id=segments_id,
                                   segments_mask=segments_mask,
                                   edu_begin_indices=edu_begin_indices,
                                   edu_end_indices=edu_end_indices,
                                   edu_head_indices=edu_head_indices) # (n_edus, edu_dim)

        # Biaffine inputs (dependent side)
        arc_dep_vectors = model.mlp_arc_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_arc_dim)
        rel_dep_vectors = model.mlp_rel_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_rel_dim)

        # Initialization of decoder state
        decoder_state = model.initialize_decoder_state(edu_vectors=edu_vectors) # ((1, 1, lstm_dim), (1, 1, lstm_dim))

        while not self.is_finished(parser_state):
            head = parser_state.stack[-1]

            # Decoder
            stack_top_index = torch.tensor([head], device=model.device)
            valence = torch.tensor([head2vals[head]], device=model.device)
            decoder_state_vector, decoder_state = model.decode(
                                                        edu_vectors=edu_vectors,
                                                        stack_top_indices=stack_top_index,
                                                        valences=valence,
                                                        decoder_state=decoder_state) # (1, lstm_dim), STATE

            # Biaffine inputs (head side)
            arc_head_vector = model.mlp_arc_h(decoder_state_vector).unsqueeze(0) # (1, 1, mlp_arc_dim)
            rel_head_vector = model.mlp_rel_h(decoder_state_vector).unsqueeze(0) # (1, 1, mlp_rel_dim)

            # Arc scoring and relation classification
            pred_actions = model.biaffine_arc(arc_head_vector, arc_dep_vectors).squeeze(0).squeeze(0) # (1, n_edus)
            pred_relations = model.biaffine_rel(rel_head_vector, rel_dep_vectors).permute(0, 2, 3, 1).squeeze(0).squeeze(0) # (n_edus, n_relations)

            # Choose dependent (1/2)
            pred_actions = torch.softmax(pred_actions, axis=1) # (1, n_edus)

            # Create mask
            mask = parser_state.get_mask_copy() # list of int
            # Head (ie., stack top) can be chosen if it is not the Root's dependent
            assert mask[head] == CLOSE
            if len(parser_state.stack) > 2:
                mask[head] = OPEN
            # If subtree is still incomplete, current head cannot be chosen
            if head in parser_state.span_heads:
                mask[head] = CLOSE
            # Some positions cannot be chosen for avoiding cross dependencies
            for dep in range(len(mask)):
                if head != dep and mask[dep] == OPEN:
                    a1, b1 = sorted([head, dep])
                    for h, d, _ in parser_state.arcs:
                        a2, b2 = sorted([h, d])
                        if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1:
                            mask[dep] = CLOSE

            # Choose dependent (2/2)
            mask = torch.tensor(mask, device=model.device).unsqueeze(0) # (1, n_edus)
            pred_actions = pred_actions - 10000.0 * mask # (1, n_edus)
            dep = torch.argmax(pred_actions, axis=1).item()
            assert mask[0,dep].item() == OPEN
            parser_state.action_history.append(dep)

            # Choose relation
            pred_relations = pred_relations[dep:dep+1] # (1, n_relations)
            rel_id = torch.argmax(pred_relations, axis=1).item()
            rel = model.ivocab_relation[rel_id]

            # Update state
            if dep != head:
                parser_state.stack.append(dep)
                parser_state.arcs.append((head, dep, rel))
                parser_state.pre_mask[dep] = CLOSE

                if head < dep:
                    indices = parser_state.span_heads == CHOSEN
                    parser_state.span_heads[head+1:dep] = head
                    parser_state.span_heads[indices] = CHOSEN
                else:
                    indices = parser_state.span_heads == CHOSEN
                    parser_state.span_heads[dep+1:head] = head
                    parser_state.span_heads[indices] = CHOSEN
                parser_state.span_heads[head] = CHOSEN
                parser_state.span_heads[dep] = CHOSEN

                head2vals[head] += 1
            else:
                parser_state.stack.pop()
                parser_state.pre_mask[dep] = CLOSE

        return parser_state

    def is_finished(self, parser_state):
        """
        Parameters
        ----------
        parser_state: StackPointerParserState

        Returns
        -------
        bool
        """
        if sum(parser_state.pre_mask) == len(parser_state.pre_mask):
            return True
        else:
            return False



