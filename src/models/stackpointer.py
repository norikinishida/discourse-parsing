import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer

from . import shared_functions


class StackPointerModel(nn.Module):

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

        # Dimentionality and etc.
        self.bert_emb_dim = self.bert.config.hidden_size
        self.encoder_lstm_input_dim = self.bert_emb_dim * 2
        if self.use_edu_head_information:
            self.encoder_lstm_input_dim += self.bert_emb_dim
        if self.use_edu_head_attn:
            self.encoder_lstm_input_dim += self.bert_emb_dim
        self.edu_dim = config["lstm_dim"] * 2
        self.max_val = 3 # 0, 1, 2, 3
        self.val_emb_dim = 4
        self.n_decoder_lstm_layers = 1
        self.use_valence = config["use_valence"]

        # Valence embedding
        if self.use_valence:
            self.embed_val = self.make_embedding(dict_size=1 + self.max_val, dim=self.val_emb_dim)

        # Encoder LSTM
        self.lstm_enc = nn.LSTM(self.encoder_lstm_input_dim,
                                config["lstm_dim"],
                                num_layers=config["encoder_lstm_layers"],
                                bidirectional=True,
                                batch_first=True,
                                dropout=config["dropout_rate"] if config["encoder_lstm_layers"] > 1 else 0.0)

        # Decoder LSTM
        self.mlp_dec_init = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["lstm_dim"], dropout=self.dropout)
        self.lstm_dec = nn.LSTM(self.edu_dim + self.val_emb_dim if self.use_valence else self.edu_dim,
                                config["lstm_dim"],
                                num_layers=self.n_decoder_lstm_layers,
                                bidirectional=False,
                                batch_first=True)

        # Biaffine for arc scoring and relation classification
        self.mlp_arc_h = shared_functions.make_mlp_hidden(input_dim=config["lstm_dim"], hidden_dim=config["mlp_arc_dim"], dropout=self.dropout)
        self.mlp_arc_d = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_arc_dim"], dropout=self.dropout)
        self.mlp_rel_h = shared_functions.make_mlp_hidden(input_dim=config["lstm_dim"], hidden_dim=config["mlp_rel_dim"], dropout=self.dropout)
        self.mlp_rel_d = shared_functions.make_mlp_hidden(input_dim=self.edu_dim, hidden_dim=config["mlp_rel_dim"], dropout=self.dropout)
        self.biaffine_arc = shared_functions.Biaffine(input_dim=config["mlp_arc_dim"], output_dim=1, bias_x=True, bias_y=True)
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

    def forward_for_training(self,
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
                             use_paragraph_boundaries,
                             #
                             stack_top_indices,
                             valences):
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
        sentence_boundaries: list of (int, int)
        paragraph_boundaries: list of (int, int)
        use_sentence_boundaries: bool
        use_paragraph_boundaries: bool
        stack_top_indices: Tensor(shape=(n_actions,))
        valences: Tensor(shape=(action_length,))

        Returns
        -------
        Tensor(shape=(action_length, n_edus)),
        Tensor(shape=(action_length, n_edus, n_relations))
        """
        assert len(edus[0]) == 1 # NOTE
        assert edus[0][0] == "<root>" # NOTE

        # Compute EDU vectorts using encoder LSTM
        edu_vectors = self.encode(edus=edus,
                                  segments=segments,
                                  segments_id=segments_id,
                                  segments_mask=segments_mask,
                                  edu_begin_indices=edu_begin_indices,
                                  edu_end_indices=edu_end_indices,
                                  edu_head_indices=edu_head_indices) # (n_edus, edu_dim)

        # Compute dependent-side vectors using MLP
        arc_dep_vectors = self.mlp_arc_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_arc_dim)
        rel_dep_vectors = self.mlp_rel_d(edu_vectors).unsqueeze(0) # (1, n_edus, mlp_rel_dim)

        # Initialize decoder state
        decoder_state = self.initialize_decoder_state(edu_vectors=edu_vectors) # STATE

        # Compute decoding states using decoder LSTM
        decoder_state_vectors, _ = self.decode(edu_vectors=edu_vectors,
                                               stack_top_indices=stack_top_indices,
                                               valences=valences,
                                               decoder_state=decoder_state) # (action_length, lstm_dim)

        # Compute head-side vectors using MLP
        arc_head_vectors = self.mlp_arc_h(decoder_state_vectors).unsqueeze(0) # (1, action_length, mlp_arc_dim)
        rel_head_vectors = self.mlp_rel_h(decoder_state_vectors).unsqueeze(0) # (1, action_length, mlp_rel_dim)

        # Compute action and relation logits using biaffine
        pred_actions = self.biaffine_arc(arc_head_vectors, arc_dep_vectors).squeeze(0).squeeze(0) # (action_length, n_edus)
        pred_relations = self.biaffine_rel(rel_head_vectors, rel_dep_vectors).permute(0, 2, 3, 1).squeeze(0) # (action_length, n_edus, n_relations)
        return pred_actions, pred_relations

    def encode(self,
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
        Tensor(shape=(n_edus, edu_dim), dtype=np.float32)
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
        edu_vectors = torch.cat(concat_list, dim=1) # (n_edus, encoder_lstm_input_dim)
        edu_vectors = self.dropout(edu_vectors)

        # Update EDU vectors using encoder LSTM
        edu_vectors, _ = self.lstm_enc(edu_vectors.unsqueeze(0))
        edu_vectors = edu_vectors.squeeze(0) # (n_edus, edu_dim = lstm_dim * 2)
        edu_vectors = self.dropout(edu_vectors)

        return edu_vectors

    def initialize_decoder_state(self, edu_vectors):
        """
        Parameters
        ----------
        edu_vectors: Tensor(shape=(n_edus, edu_dim))

        Returns
        -------
        (Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)), Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)))
        """
        mean_vector = torch.mean(edu_vectors, axis=0).unsqueeze(0) # (1, edu_dim)
        cell = self.mlp_dec_init(mean_vector).unsqueeze(0) # (1, 1, lstm_dim)
        if self.n_decoder_lstm_layers > 1:
            cell = torch.cat([cell, cell.new_zeros(self.n_decoder_lstm_layers - 1, 1, self.config["lstm_dim"])], dim=0)
        # cell.shape: (n_decoder_lstm_layers, 1, lstm_dim)
        state = torch.tanh(cell) # (n_decoder_lstm_layers, 1, lstm_dim)
        return (state, cell)

    def decode(self, edu_vectors, stack_top_indices, valences, decoder_state):
        """
        Parameters
        ----------
        edu_vectors: Tensor(shape=(n_edus, edu_dim))
        stack_top_indices: Tensor(shape=(action_length,))
        valences: Tensor(shape=(action_length,))
        decoder_state: (Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)), Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)))

        Returns
        -------
        Tensor(shape=(action_length, lstm_dim))
        (Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)), Tensor(shape=(n_decoder_lstm_layers, 1, lstm_dim)))
        """
        # Compute parser-state's input vector
        decoder_input_vectors = edu_vectors[stack_top_indices].unsqueeze(0) # (1, action_length, edu_dim)
        if self.use_valence:
            valences = torch.clamp(valences, max=self.max_val) # (action_length,)
            valence_vectors = self.embed_val(valences).unsqueeze(0) # (1, action_length, val_emb_dim)
            decoder_input_vectors = torch.cat([decoder_input_vectors, valence_vectors], axis=2) # (1, action_length, edu_dim + val_emb_dim)

        # Update parser-state vector using decoder LSTM for one step
        decoder_state_vectors, decoder_state = self.lstm_dec(decoder_input_vectors, hx=decoder_state) # (1, action_length, lstm_dim), STATE
        decoder_state_vectors = decoder_state_vectors.squeeze(0) # (action_length, lstm_dim)
        decoder_state_vectors = self.dropout(decoder_state_vectors)

        return decoder_state_vectors, decoder_state

