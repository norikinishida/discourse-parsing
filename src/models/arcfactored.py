import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer

from . import shared_functions


class BiaffineModel(nn.Module):

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

        # MLP for EDU-span head attention
        if self.use_edu_head_attn:
            self.mlp_edu_head_attn = shared_functions.make_mlp(input_dim=self.bert_emb_dim,
                                                               hidden_dims=[config["mlp_edu_head_attn_dim"]],
                                                               output_dim=1,
                                                               dropout=self.dropout)

        # Biaffine for arc scoring and relation classification
        self.mlp_arc_h = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_arc_dim"], dropout=self.dropout)
        self.mlp_arc_d = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_arc_dim"], dropout=self.dropout)
        self.mlp_rel_h = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_rel_dim"], dropout=self.dropout)
        self.mlp_rel_d = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_rel_dim"], dropout=self.dropout)
        self.biaffine_arc = shared_functions.Biaffine(input_dim=config["mlp_arc_dim"], output_dim=1, bias_x=False, bias_y=True)
        self.biaffine_rel = shared_functions.Biaffine(input_dim=config["mlp_rel_dim"], output_dim=self.n_relations, bias_x=True, bias_y=True)

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

    def forward(self,
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
        Tensor(shape=(n_edus, n_edus), dtype=np.float32)
        Tensor(shape=(n_edus, n_edus, n_relations), dtype=np.float32)
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

        # Compute arc scores using biaffine
        arc_head_vectors = self.mlp_arc_h(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_arc_dim)
        arc_dep_vectors = self.mlp_arc_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_arc_dim)
        arc_scores = self.biaffine_arc(arc_head_vectors, arc_dep_vectors).squeeze(0).squeeze(0) # (n_edus, n_edus)

        # Compute relation logits using biaffine
        rel_head_vectors = self.mlp_rel_h(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_rel_dim)
        rel_dep_vectors = self.mlp_rel_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_rel_dim)
        rel_logits = self.biaffine_rel(rel_head_vectors, rel_dep_vectors).permute(0, 2, 3, 1).squeeze(0) # (n_edus, n_edus, n_relations)

        return arc_scores, rel_logits



