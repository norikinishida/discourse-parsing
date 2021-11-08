import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer

from . import shared_functions


class ArcStandardModel(nn.Module):

    def __init__(self,
                 device,
                 config,
                 vocab_relation):
        """
        Parameters
        ----------
        device: str
        config: ConfigTree
        vocab_relation: dict[str, int]
        """
        super().__init__()

        ################
        # Hyper parameters
        ################

        self.device = device
        self.config = config
        self.vocab_relation = vocab_relation
        self.ivocab_relation = {i:l for l,i in self.vocab_relation.items()}
        self.n_relations = len(self.vocab_relation)

        self.vocab_action = {"SHIFT": 0}
        for r in self.vocab_relation:
            if r == "<root>":
                continue
            self.vocab_action["LEFT-%s" % r] = len(self.vocab_action)
            self.vocab_action["RIGHT-%s" % r] = len(self.vocab_action)
        self.vocab_action["ROOT-<root>"] = len(self.vocab_action)
        self.ivocab_action = {i:l for l,i in self.vocab_action.items()}
        self.n_actions = len(self.vocab_action)

        self.use_edu_head_information = self.config["use_edu_head_information"]
        self.use_edu_head_attn = self.config["use_edu_head_attn"]

        ################
        # Model components
        ################

        self.dropout = nn.Dropout(p=config["dropout_rate"])

        # BERT
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_tokenizer_pretrained_name_or_path"], additional_special_tokens=["<root>"])
        self.bert = AutoModel.from_pretrained(config["bert_pretrained_name_or_path"], return_dict=False)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # Initialize the Root ("<root>") embedding with the CLS embedding
        root_id = self.tokenizer.convert_tokens_to_ids(["<root>"])[0]
        cls_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.bert.embeddings.word_embeddings.weight.data[root_id] = self.bert.embeddings.word_embeddings.weight.data[cls_id]

        # Dimentionality
        self.bert_emb_dim = self.bert.config.hidden_size
        self.edu_dim = self.bert_emb_dim * 2
        if self.use_edu_head_information:
            self.edu_dim += self.bert_emb_dim
        if self.use_edu_head_attn:
            self.edu_dim += self.bert_emb_dim
        self.state_dim = self.edu_dim * 4

        # MLP for EDU-span head attention
        if self.use_edu_head_attn:
            self.mlp_edu_head_attn = shared_functions.make_mlp(input_dim=self.bert_emb_dim,
                                                               hidden_dims=[config["mlp_edu_head_attn_dim"]],
                                                               output_dim=1,
                                                               dropout=self.dropout)

        # MLP for action classification
        self.mlp = shared_functions.make_mlp(input_dim=self.state_dim,
                                             hidden_dims=[config["mlp_dim"]] * config["mlp_depth"],
                                             output_dim=self.n_actions,
                                             dropout=self.dropout)

    #########################
    # For optimization
    #########################

    def get_params(self, named=False):
        """
        Parameters
        ----------
        named: bool

        Returns
        -------
        list[(str, Param)] or list[Param]
        list[(str, Param)] or list[Param]
        """
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith("bert"):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    #########################
    # Forwarding
    #########################

    def forward_edus(self,
                     edus,
                     segments,
                     segments_id,
                     segments_mask,
                     edu_begin_indices,
                     edu_end_indices,
                     edu_head_indices):
        """
        Parameters
        ----------
        edus: list[list[str]]
        segments: list[list[str]]
        segments_id: Tensor(shape=(n_segments, max_seg_len), dtype=torch.long)
        segments_mask: Tensor(shape=(n_segments, max_seg_len), dtype=torch.long)
        edu_begin_indices: Tensor(shape=(n_edus,), dtype=torch.long)
        edu_end_indices: Tensor(shape=(n_edus,), dtype=torch.long)
        edu_head_indices: Tensor(shape=(n_edus,), dtype=torch.long) or None

        Returns
        -------
        Tensor(shape=(n_edus+1, edu_dim), dtype=np.float32)
        """
        assert len(edus[0]) == 1 # NOTE
        assert edus[0][0] == "<root>" # NOTE

        # Embed tokens using BERT
        token_vectors, _ = self.bert(segments_id, attention_mask=segments_mask) # (n_segments, max_seg_len, bert_emb_dim)
        segments_mask = segments_mask.to(torch.bool) # (n_segments, max_seg_len)
        token_vectors = token_vectors[segments_mask] # (n_tokens, bert_emb_dim)

        # Compute EDU vectors using concatenation
        edu_start_vectors = token_vectors[edu_begin_indices] # (n_edus, bert_emb_dim)
        edu_end_vectors = token_vectors[edu_end_indices] # (n_edus, bert_emb_dim)
        concat_list = [edu_start_vectors, edu_end_vectors]
        if self.use_edu_head_information:
            edu_head_vectors = token_vectors[edu_head_indices] # (n_edus, bert_emb_dim)
            concat_list.append(edu_head_vectors)
        if self.use_edu_head_attn:
            token_attns = torch.squeeze(self.mlp_edu_head_attn(token_vectors), 1) # (n_tokens,)
            doc_range = torch.arange(0, token_vectors.shape[0]).to(self.device) # (n_tokens,)
            doc_range_1 = edu_begin_indices.unsqueeze(1) <= doc_range # (n_edus, n_tokens)
            doc_range_2 = doc_range <= edu_end_indices.unsqueeze(1) # (n_edus, n_tokens)
            edu_span_token_mask = doc_range_1 & doc_range_2 # (n_edus, n_tokens)
            edu_span_token_attns = torch.log(edu_span_token_mask.float()) + torch.unsqueeze(token_attns, 0) # (n_edus, n_tokens); masking for EDU spans (w/ broadcasting)
            edu_span_token_attns = nn.functional.softmax(edu_span_token_attns, dim=1) # (n_edus, n_tokens)
            edu_span_ha_vectors = torch.matmul(edu_span_token_attns, token_vectors) # (n_edus, bert_emb_dim)
            concat_list.append(edu_span_ha_vectors)
        edu_vectors = torch.cat(concat_list, dim=1) # (n_edus, edu_dim)

        # Append a zero vector for padding
        zero_vector = torch.zeros((1, self.edu_dim), device=self.device) # (1, edu_dim)
        edu_vectors = torch.cat([edu_vectors, zero_vector], axis=0) # (n_edus+1, edu_dim)

        return edu_vectors

    def forward_parser_state(self, edu_vectors, parser_state):
        """
        Parameters
        ----------
        edu_vectors: Tensor(shape=(n_edus+1, edu_dim), dtype=np.float32)
        parser_state: ShiftReduceParserState

        Returns
        -------
        Tensor(shape=(1, n_actions), dtype=np.float32)
        """
        # Compute parser-state vector
        stack_top_indices = parser_state.stack[-3:] # s2, s1, s0
        buffer_top_index = parser_state.buffer[0:1] # b0

        pad_index = len(edu_vectors) - 1 # Last vector is pad
        while len(stack_top_indices) < 3:
            stack_top_indices = [pad_index] + stack_top_indices
        if len(buffer_top_index) == 0:
            buffer_top_index = [pad_index]

        stack_top_indices = torch.tensor(stack_top_indices, device=self.device)
        buffer_top_index = torch.tensor(buffer_top_index, device=self.device)

        stack_top_vectors = edu_vectors[stack_top_indices] # (3, edu_dim)
        stack_top_vectors = torch.reshape(stack_top_vectors, (1, -1)) # (1, 3 * edu_dim)
        buffer_top_vector = edu_vectors[buffer_top_index] # (1, edu_dim)
        parser_state_vectors = torch.cat([stack_top_vectors, buffer_top_vector], dim=1) # (1, state_dim)

        # Compute action logits using MLP
        logits = self.mlp(parser_state_vectors) # (1, n_actions)

        return logits



