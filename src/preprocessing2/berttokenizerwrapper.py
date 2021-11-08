import numpy as np

import utils


class BertTokenizerWrapper:

    def __init__(self, tokenizer):
        """
        Parameters
        ----------
        tokenizer: transformers.PreTrainedTokenizer
        """
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.max_seg_len = 512


    def tokenize_and_split(self, edus, edus_head, sentence_boundaries):
        """
        Parameters
        ----------
        edus: list[list[str]]
        edus_head: list[int]
        sentence_boundaries: list[(int, int)]

        Returns
        -------
        list[list[str]]
        list[list[int]]
        list[list[int]]
        list[int]
        list[int]
        list[int]
        """
        # subtoken分割、subtoken単位でのtoken終点、文終点、EDU始点、EDU終点のindicatorsの作成
        edus_subtoken, edus_token_end, edus_sentence_end, edus_edu_begin, edus_edu_end, edus_edu_head \
                = self.tokenize(edus=edus, edus_head=edus_head, sentence_boundaries=sentence_boundaries)

        # 1階層のリストに変換
        doc_subtoken = utils.flatten_lists(edus_subtoken)
        doc_token_end = utils.flatten_lists(edus_token_end)
        doc_sentence_end = utils.flatten_lists(edus_sentence_end)
        doc_edu_begin = utils.flatten_lists(edus_edu_begin)
        doc_edu_end = utils.flatten_lists(edus_edu_end)
        doc_edu_head = utils.flatten_lists(edus_edu_head)

        # BERTセグメントに分割
        segments, edu_begin_indices, edu_end_indices, edu_head_indices \
                = self.split(doc_subtoken=doc_subtoken,
                             doc_sentence_end=doc_sentence_end,
                             doc_token_end=doc_token_end,
                             doc_edu_begin=doc_edu_begin,
                             doc_edu_end=doc_edu_end,
                             doc_edu_head=doc_edu_head)
        assert len(edu_begin_indices) == len(edu_end_indices) == len(edu_head_indices) == len(edus)

        # subtoken IDへの変換とpaddingマスクの作成
        segments_id, segments_mask = self.convert_to_token_ids_with_padding(segments=segments)

        return segments, segments_id, segments_mask, edu_begin_indices, edu_end_indices, edu_head_indices


    def tokenize(self, edus, edus_head, sentence_boundaries):
        """
        Parameters
        ----------
        edus: list[list[str]]
        edus_head: list[int]
        sentence_boundaries: list[(int, int)]

        Returns
        -------
        list[list[str]]
        list[list[bool]]
        list[list[bool]]
        list[list[bool]]
        list[list[bool]]
        list[list[bool]]
        """
        edus_subtoken = [] # subtokens
        edus_token_end = [] # subtoken-level indicators of token end point
        edus_sentence_end = [] # subtoken-level indicators of sentence end point
        edus_edu_begin = [] # subtoken-level indicators of EDU beginning point
        edus_edu_end = [] # subtoken-level indicators of EDU end point
        edus_edu_head = [] # subtoken-level indicators of EDU head beginning point
        for edu, head_pos in zip(edus, edus_head):
            edu_subtoken = []
            edu_token_end = []
            edu_sentence_end = []
            edu_edu_begin = []
            edu_edu_end = []
            edu_edu_head = []
            for token_i, token in enumerate(edu):
                subtokens = self.tokenizer.tokenize(token)
                if len(subtokens) == 0:
                    subtokens = [self.tokenizer.unk_token]
                edu_subtoken.extend(subtokens)
                edu_token_end += [False] * (len(subtokens) - 1) + [True]
                edu_sentence_end += [False] * len(subtokens)
                edu_edu_begin += [False] * len(subtokens)
                edu_edu_end += [False] * len(subtokens)
                if token_i == head_pos:
                    edu_edu_head += [True]  + [False] * (len(subtokens) - 1)
                else:
                    edu_edu_head += [False] * len(subtokens)
            edu_edu_begin[0] = True
            edu_edu_end[-1] = True
            edus_subtoken.append(edu_subtoken)
            edus_token_end.append(edu_token_end)
            edus_sentence_end.append(edu_sentence_end)
            edus_edu_begin.append(edu_edu_begin)
            edus_edu_end.append(edu_edu_end)
            edus_edu_head.append(edu_edu_head)
        for begin_edu_i, end_edu_i in sentence_boundaries:
            edus_sentence_end[1+end_edu_i][-1] = True # Add 1 to the index due to shifting by the ROOT EDU

        return edus_subtoken, edus_token_end, edus_sentence_end, edus_edu_begin, edus_edu_end, edus_edu_head


    def split(self, doc_subtoken, doc_sentence_end, doc_token_end, doc_edu_begin, doc_edu_end, doc_edu_head):
        """
        Parameters
        ----------
        doc_subtoken: list[str]
        doc_sentence_end: list[bool]
        doc_token_end: list[bool]
        doc_edu_begin: list[bool]
        doc_edu_end: list[bool]
        doc_edu_head: list[bool]

        Returns
        -------
        list[list[str]]
        list[int]
        list[int]
        list[int]
        """
        segments = []
        segments_edu_begin = [] # indicators of EDU beginning point
        segments_edu_end = [] # indicators of EDU end point
        segments_edu_head = [] # indicators of EDU head beginning point

        n_subtokens = len(doc_subtoken)
        curr_idx = 0 # Index for subtokens
        while curr_idx < len(doc_subtoken):
            # Try to split at a sentence end point
            end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1) # Inclusive
            while end_idx >= curr_idx and not doc_sentence_end[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                utils.writelog(f"No sentence end found; split at token end")
                # If no sentence end point found, try to split at token end point
                end_idx = min(curr_idx + self.max_seg_len - 1 - 2, n_subtokens - 1)
                while end_idx >= curr_idx and not doc_token_end[end_idx]:
                    end_idx -= 1
                if end_idx < curr_idx:
                    utils.writelog("Cannot split valid segment: no sentence end or token end", error=True)

            segment = [self.cls_token] + doc_subtoken[curr_idx: end_idx + 1] + [self.sep_token]
            segment_edu_begin = [False] + doc_edu_begin[curr_idx: end_idx + 1] + [False]
            segment_edu_end = [False] + doc_edu_end[curr_idx: end_idx + 1] + [False]
            segment_edu_head = [False] + doc_edu_head[curr_idx: end_idx + 1] + [False]

            segments.append(segment)
            segments_edu_begin.append(segment_edu_begin)
            segments_edu_end.append(segment_edu_end)
            segments_edu_head.append(segment_edu_head)

            curr_idx = end_idx + 1

        edu_begin_indices = [i for i,x in enumerate(utils.flatten_lists(segments_edu_begin)) if x]
        edu_end_indices = [i for i,x in enumerate(utils.flatten_lists(segments_edu_end)) if x]
        edu_head_indices = [i for i,x in enumerate(utils.flatten_lists(segments_edu_head)) if x]

        return segments, edu_begin_indices, edu_end_indices, edu_head_indices


    def convert_to_token_ids_with_padding(self, segments):
        """
        Parameters
        ----------
        segments: list[list[str]]

        Returns
        -------
        list[list[int]], list[list[int]]
        """
        n_subtokens = sum([len(s) for s in segments])
        segments_id = []
        segments_mask = []
        for segment in segments:
            segment_id = self.tokenizer.convert_tokens_to_ids(segment)
            segment_mask = [1] * len(segment_id)
            while len(segment_id) < self.max_seg_len:
                segment_id.append(0)
                segment_mask.append(0)
            segments_id.append(segment_id)
            segments_mask.append(segment_mask)
        assert np.sum(np.asarray(segments_mask)) == n_subtokens, (np.sum(np.asarray(segments_mask)), n_subtokens)
        return segments_id, segments_mask


