import copy

import numpy as np
import scipy.special


class ShiftReduceParserState(object):

    def __init__(self, input):
        """
        Parameters
        ----------
        input: list[int]
        """
        assert not 0 in input # No ROOT
        self.stack = []
        self.buffer = copy.deepcopy(input)
        self.arcs = []
        self.action_history = []

    def __str__(self):
        return "ShiftReduceParserState(stack: %s; buffer: %s; arcs: %s; action_history: %s)" \
                % (self.stack, self.buffer, self.arcs, self.action_history)


class ArcStandardDecoder(object):

    def __init__(self):
        pass

    def decode(self,
               model,
               edu_ids,
               data,
               #
               gold_arcs=None,
               #
               confidence_measure=None):
        """
        Parameters
        ----------
        model: ArcStandardModel
        edu_ids: list[int]
        data: DataInstance
        gold_arcs: list[(int, int, str)]
        confidence_measure: str

        Returns
        -------
        ShiftReduceParserState
        list[Tensor(shape=(1,n_actions), dtype=np.float32)]
        list[str]
        """
        assert edu_ids[0] == 0 # ROOT

        parser_state = ShiftReduceParserState(input=edu_ids[1:])

        # ``logits_list`` does not include the root-attachment prediction
        logits_list = []
        gold_actions = []

        if confidence_measure is not None:
            confidence = 0.0
            confidence_count = 0

        # Encode EDUs
        #   (1) Including the ROOT feature;
        #   (2) Irrelevant with the input order (LR, RL)
        edu_vectors = model.forward_edus(data=data) # (n_edus + 1, edu_dim)

        if gold_arcs is None:
            #####
            # Inference mode
            #####
            while not self.is_finished(parser_state):
                # Predict action scores (logits)
                logits = model.forward_parser_state(
                                    edu_vectors=edu_vectors,
                                    parser_state=parser_state) # (1, n_actions)
                logits_list.append(logits)

                if confidence_measure is not None:
                    dist = scipy.special.softmax(logits.cpu().numpy(), axis=1)

                # Sort actions based on the scores
                logits = logits[0].tolist() # (n_actions,)
                logits_and_actions = [(l,i) for i,l in enumerate(logits)]
                logits_and_actions = sorted(logits_and_actions, key=lambda tpl: -tpl[0])
                actions = [i for l,i in logits_and_actions]
                actions = [model.ivocab_action[i] for i in actions]

                # Get a set of legal actions
                right_ok, left_ok, shift_ok = self.get_legal_actions(parser_state)

                # Take the highest-scoring legal action
                took_action = False
                for action in actions:
                    action_op = action.split("-")[0]
                    action_rel = "-".join(action.split("-")[1:])

                    if action_op == "RIGHT" and right_ok:
                        parser_state = self.reduce_right(parser_state, action_rel)
                        took_action = True
                        break
                    elif action_op == "LEFT" and left_ok:
                        parser_state = self.reduce_left(parser_state, action_rel)
                        took_action = True
                        break
                    elif action_op == "SHIFT" and shift_ok:
                        parser_state = self.shift(parser_state)
                        took_action = True
                        break
                assert took_action

                # Measure confidence
                if confidence_measure == "predictive_probability":
                    act_i = model.vocab_action[parser_state.action_history[-1]]
                    confidence += dist[0, act_i]
                    confidence_count += 1
                elif confidence_measure == "negative_entropy":
                    confidence += np.sum((dist + 1e-6) * np.log(dist + 1e-6), axis=1)
                    confidence_count += 1
        else:
            #####
            # Training time
            #####
            gold_arcs = copy.deepcopy(gold_arcs)
            relations = [r for _,_,r in gold_arcs]
            gold_arcs = [(h,d) for h,d,_ in gold_arcs]

            while not self.is_finished(parser_state):
                # Predict action scores (logits)
                logits = model.forward_parser_state(
                                    edu_vectors=edu_vectors,
                                    parser_state=parser_state) # (1, n_actions)
                logits_list.append(logits)

                # Get a set of legal actions
                right_ok, left_ok, shift_ok = self.get_legal_actions(parser_state, gold_arcs=gold_arcs)

                # Take the highest-scoring legal action
                action_op = None
                if right_ok:
                    action_op = "RIGHT"
                elif left_ok:
                    action_op = "LEFT"
                elif shift_ok:
                    action_op = "SHIFT"
                else:
                    raise Exception("Unable to find the legal actions!")
                if action_op == "RIGHT":
                    s0 = parser_state.stack[-1] # x_{i}
                    s1 = parser_state.stack[-2] # x_{i-1}
                    index = gold_arcs.index((s1, s0))
                    relation = relations[index]
                    #
                    parser_state = self.reduce_right(parser_state, relation)
                    gold_actions.append("%s-%s" % (action_op, relation))
                    # Remove
                    gold_arcs.pop(index)
                    relations.pop(index)
                elif action_op == "LEFT":
                    s0 = parser_state.stack[-1]
                    b = parser_state.buffer[0]
                    index = gold_arcs.index((b, s0))
                    relation = relations[index]
                    #
                    parser_state = self.reduce_left(parser_state, relation)
                    gold_actions.append("%s-%s" % (action_op, relation))
                    # Remove
                    gold_arcs.pop(index)
                    relations.pop(index)
                elif action_op == "SHIFT":
                    parser_state = self.shift(parser_state)
                    gold_actions.append(action_op)
                else:
                    raise Exception("Invalid action_op=%s" % action_op)

            assert len(logits_list) == len(gold_actions)

        # Root attachment
        parser_state = self.attach_root(parser_state)

        if confidence_measure is not None:
            parser_state.confidence = confidence / confidence_count

        return parser_state, logits_list, gold_actions

    def shift(self, parser_state):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState

        Returns
        -------
        ShiftReduceParserState
        """
        # <[..], [b,..], T> to <[..,b], [..], T>
        b = parser_state.buffer.pop(0)
        parser_state.stack.append(b)
        parser_state.action_history.append("SHIFT")
        return parser_state

    def reduce_right(self, parser_state, relation):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState
        relation: str

        Returns
        -------
        ShiftReduceParserState
        """
        # <[..,s1,s0], [..], T> to <[..,s1], [..], T+(s1,s0,r)>
        s0 = parser_state.stack.pop()
        s1 = parser_state.stack[-1]
        parser_state.arcs.append((s1, s0, relation))
        parser_state.action_history.append("RIGHT-%s" % relation)
        return parser_state

    def reduce_left(self, parser_state, relation):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState
        relation: str

        Returns
        -------
        ShiftReduceParserState
        """
        # <[..,s1,s0], [b,..], T> to <[..,s1], [b,..], T+(b,s0,r)>
        s0 = parser_state.stack.pop()
        b = parser_state.buffer[0]
        parser_state.arcs.append((b, s0, relation))
        parser_state.action_history.append("LEFT-%s" % relation)
        return parser_state

    def attach_root(self, parser_state):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState

        Returns
        -------
        ShiftReduceParserState
        """
        # <[s0], [], T> to <[s0], [], T+(0,s0,"<root>")>
        assert self.is_finished(parser_state)
        dep = parser_state.stack[0]
        parser_state.arcs.append((0, dep, "<root>"))
        parser_state.action_history.append("ROOT-<root>")
        return parser_state

    def is_finished(self, parser_state):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState

        Returns
        -------
        bool
        """
        if len(parser_state.stack) == 1 and len(parser_state.buffer) == 0:
            return True
        else:
            return False

    def get_legal_actions(self, parser_state, gold_arcs=None):
        """
        Parameters
        ----------
        parser_state: ShiftReduceParserState
        gold_arcs: list[(int, int)] or None

        Returns
        -------
        bool
        bool
        bool
        """
        right_ok, left_ok, shift_ok = False, False, False
        if len(parser_state.stack) >= 2:
            if gold_arcs is None:
                right_ok = True
            else:
                s0 = parser_state.stack[-1]
                s1 = parser_state.stack[-2]
                if (s1, s0) in gold_arcs and not self.has_dependent(s0, gold_arcs):
                    right_ok = True
        if len(parser_state.stack) >= 1 and len(parser_state.buffer) >= 1:
            if gold_arcs is None:
                left_ok = True
            else:
                s0 = parser_state.stack[-1]
                b = parser_state.buffer[0]
                if (b, s0) in gold_arcs and not self.has_dependent(s0, gold_arcs):
                    left_ok = True
        if len(parser_state.buffer) >= 1:
            shift_ok = True
        return right_ok, left_ok, shift_ok

    def has_dependent(self, head, gold_arcs):
        """
        Parameters
        ----------
        head: int
        gold_arcs: list[(int, int)]

        Returns
        -------
        bool
        """
        for (h, d) in gold_arcs:
            if h == head:
                return True
        return False


