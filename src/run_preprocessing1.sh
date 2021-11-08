#!/usr/bin/env sh

# Path to downloaded datasets
RSTDT=/home/nishida/storage/dataset/RST-DT/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0
PTBWSJ=/home/nishida/storage/dataset/Penn-Treebank/LDC99T42/treebank_3/raw/wsj
STAC=/home/nishida/storage/dataset/STAC/stac-linguistic-2018-05-04/data
STAC_TEST=/home/nishida/storage/dataset/STAC/TEST_spect
MOLWENI=/home/nishida/storage/dataset/Molweni/DP
AASC=/home/nishida/storage/dataset/AASC-v4/AASC_v4.sentence
CORD19=/home/nishida/storage/dataset/CORD-19-Snapshot-20200928/document_parses
UDC=/home/nishida/storage/dataset/Ubuntu-Dialogue-Corpus-v2/ubuntu-ranking-dataset-creator/src

STORAGE=/home/nishida/storage/projects/discourse/discourse-parsing
STORAGE_DATA=${STORAGE}/data


################################
# SpanBERT
################################


download_spanbert(){
    model=$1
    wget -P ${STORAGE} https://dl.fbaipublicfiles.com/fairseq/models/${model}.tar.gz
    mkdir ${STORAGE}/${model}
    tar zxvf ${STORAGE}/${model}.tar.gz -C ${STORAGE}/${model}
    rm ${STORAGE}/${model}.tar.gz
}

download_spanbert spanbert_hf_base
download_spanbert spanbert_hf


################################
# RST-DT
################################


# Step 1. Extract EDUs and gold annotations from RST-DT.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.edus.tokens (original),
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.*.*.ctree,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.arcs
python ./preprocessing1/prepare_rstdt.py --input ${RSTDT}/TRAINING --output ${STORAGE_DATA}/rstdt/wsj/train
python ./preprocessing1/prepare_rstdt.py --input ${RSTDT}/TEST --output ${STORAGE_DATA}/rstdt/wsj/test
python ./preprocessing1/convert_ctrees_to_dtrees_rstdt.py --path ${STORAGE_DATA}/rstdt/wsj/train
python ./preprocessing1/convert_ctrees_to_dtrees_rstdt.py --path ${STORAGE_DATA}/rstdt/wsj/test

# Step 2. Extract raw texts (corresponding to RST-DT) from PTB-WSJ for determining paragraph boundaries.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.doc.tokens
python ./preprocessing1/prepare_ptbwsj.py --ptbwsj_in ${PTBWSJ} --rstdt_train ${STORAGE_DATA}/rstdt/wsj/train --rstdt_test ${STORAGE_DATA}/rstdt/wsj/test --inside_rstdt

# Step 3. Fix noisy mismatched characters (e.g., "Oct. 5" (in PTB-WSJ) vs. "Oct. 17" (in RST-DT)) between RST-DT and PTB-WSJ to determine paragraph boundaries.
# I don't know why such minor character changes happen during the RST-DT annotation.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.edus.tokens (fixed version),
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.doc.tokens (fixed version)
#./preprocessing1/create_patch_data.sh
cp ./preprocessing1/patch_data/train/*.tokens ${STORAGE_DATA}/rstdt/wsj/train/
cp ./preprocessing1/patch_data/test/*.tokens ${STORAGE_DATA}/rstdt/wsj/test/
python ./preprocessing1/find_conflictions_btw_goldedus_and_document.py --path ${STORAGE_DATA}/rstdt/wsj/train --check_token --check_char --check_boundary
python ./preprocessing1/find_conflictions_btw_goldedus_and_document.py --path ${STORAGE_DATA}/rstdt/wsj/test --check_token --check_char --check_boundary

# Step 4. Perform sentence splitting, tokenization, POS tagging, and syntactic dependency parsing and obtain paragraph boundaries.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.sents.tokens,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.sents.postags,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.sents.arcs,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.paragraph_boundaries
python ./preprocessing1/doc2sents.py --path ${STORAGE_DATA}/rstdt/wsj/train --with_gold_edus
python ./preprocessing1/doc2sents.py --path ${STORAGE_DATA}/rstdt/wsj/test --with_gold_edus

# Step 5. Split the sentence-level features into EDU-level ones and obtain sentence boundaries.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.edus.postags,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.edus.arcs,
#       STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.sentence_boundaries
python ./preprocessing1/sents2edus.py --path ${STORAGE_DATA}/rstdt/wsj/train
python ./preprocessing1/sents2edus.py --path ${STORAGE_DATA}/rstdt/wsj/test

# Step 6. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/rstdt/wsj/{train,test}/wsj_*.edus.heads,
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/rstdt/wsj/train
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/rstdt/wsj/test

# Step 7. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.args, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries, *.ctree, *.arcs) into JSON files.
#   -> STORAGE_DATA/rstdt-compiled/wsj/{train,test}/wsj_*.json
python ./preprocessing1/compile_to_jsons_rstdt.py --input ${STORAGE_DATA}/rstdt/wsj/train --output ${STORAGE_DATA}/rstdt-compiled/wsj/train
python ./preprocessing1/compile_to_jsons_rstdt.py --input ${STORAGE_DATA}/rstdt/wsj/test --output ${STORAGE_DATA}/rstdt-compiled/wsj/test


################################
# Unlabeled dataset
# PTB-WSJ
################################


# Step 1. Extract raw texts (NOT corresponding to RST-DT) from PTB-WSJ.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.doc.tokens
python ./preprocessing1/prepare_ptbwsj.py --ptbwsj_in ${PTBWSJ} --rstdt_train ${STORAGE_DATA}/rstdt/wsj/train --rstdt_test ${STORAGE_DATA}/rstdt/wsj/test --ptbwsj_out ${STORAGE_DATA}/ptbwsj-wo-rstdt --outside_rstdt

# Step 2. Perform sentence splitting, tokenization, POS tagging, and syntactic dependency parsing and obtain paragraph boundaries.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.sents.tokens,
#       STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.sents.postags,
#       STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.sents.arcs,
#       STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.paragraph_boundaries
python ./preprocessing1/doc2sents.py --path ${STORAGE_DATA}/ptbwsj-wo-rstdt

# Step 3. Segment the documents (*.sents.tokens) into EDUs.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.edus.tokens
# NOTE: We used https://github.com/PKU-TANGENT/NeuralEDUSeg to segment the documents into EDUs. (c.f., ./preprocessing1/segment.sh)

# Step 4. Split the sentence-level features into EDU-level ones and obtain sentence boundaries.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.edus.postags,
#       STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.edus.arcs,
#       STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.sentence_boundaries
python ./preprocessing1/sents2edus.py --path ${STORAGE_DATA}/ptbwsj-wo-rstdt

# Step 5. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt/wsj_*.edus.heads,
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/ptbwsj-wo-rstdt

# Step 6. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries) into JSON files.
#   ->  STORAGE_DATA/ptbwsj-wo-rstdt-compiled/wsj_*.json
python ./preprocessing1/compile_to_jsons_unannotated_corpus.py --input ${STORAGE_DATA}/ptbwsj-wo-rstdt --output ${STORAGE_DATA}/ptbwsj-wo-rstdt-compiled


################################
# SciDTB
################################


# Step 0: Download dataset.
mkdir -p ${STORAGE_DATA}/scidtb/preprocessed
git clone https://github.com/PKU-TANGENT/SciDTB.git ${STORAGE_DATA}/scidtb/download

# Step 1. (1) Extract tokenized EDUs, sentence/paragraph boundaries, and gold annotations from the dataset; (2) perform POS tagging and syntactic dependency parsing.
#   ->  STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.edus.tokens,
#       STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.edus.postags,
#       STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.edus.arcs,
#       STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.sentence_boundaries,
#       STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.paragraph_boundaries,
#       STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.arcs
python ./preprocessing1/prepare_scidtb.py --input ${STORAGE_DATA}/scidtb/download/dataset/train --output ${STORAGE_DATA}/scidtb/preprocessed/train
python ./preprocessing1/prepare_scidtb.py --input ${STORAGE_DATA}/scidtb/download/dataset/dev/gold --output ${STORAGE_DATA}/scidtb/preprocessed/dev/gold
python ./preprocessing1/prepare_scidtb.py --input ${STORAGE_DATA}/scidtb/download/dataset/dev/second_annotate --output ${STORAGE_DATA}/scidtb/preprocessed/dev/second_annotate
python ./preprocessing1/prepare_scidtb.py --input ${STORAGE_DATA}/scidtb/download/dataset/test/gold --output ${STORAGE_DATA}/scidtb/preprocessed/test/gold
python ./preprocessing1/prepare_scidtb.py --input ${STORAGE_DATA}/scidtb/download/dataset/test/second_annotate --output ${STORAGE_DATA}/scidtb/preprocessed/test/second_annotate

# Step 2. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/scidtb/preprocessed/{train,dev/*,test/*}/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/scidtb/preprocessed/train
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/scidtb/preprocessed/dev/gold
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/scidtb/preprocessed/dev/second_annotate
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/scidtb/preprocessed/test/gold
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/scidtb/preprocessed/test/second_annotate

# Step 3. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries, *.arcs) into JSON files.
#   ->  STORAGE_DATA/scidtb-compiled/*.json
python ./preprocessing1/compile_to_jsons_scidtb.py --input ${STORAGE_DATA}/scidtb/preprocessed/train --output ${STORAGE_DATA}/scidtb-compiled/train
python ./preprocessing1/compile_to_jsons_scidtb.py --input ${STORAGE_DATA}/scidtb/preprocessed/dev/gold --output ${STORAGE_DATA}/scidtb-compiled/dev/gold
python ./preprocessing1/compile_to_jsons_scidtb.py --input ${STORAGE_DATA}/scidtb/preprocessed/dev/second_annotate --output ${STORAGE_DATA}/scidtb-compiled/dev/second_annotate
python ./preprocessing1/compile_to_jsons_scidtb.py --input ${STORAGE_DATA}/scidtb/preprocessed/test/gold --output ${STORAGE_DATA}/scidtb-compiled/test/gold
python ./preprocessing1/compile_to_jsons_scidtb.py --input ${STORAGE_DATA}/scidtb/preprocessed/test/second_annotate --output ${STORAGE_DATA}/scidtb-compiled/test/second_annotate


################################
# COVID19-DTB
################################


# Step 0: Download dataset.
mkdir -p ${STORAGE_DATA}/covid19-dtb/preprocessed
git clone https://github.com/norikinishida/biomedical-discourse-treebanks.git ${STORAGE_DATA}/covid19-dtb/download

# Step 1. (1) Extract tokenized EDUs, sentence/paragraph boundaries, and gold annotations from the dataset; (2) perform POS tagging and syntactic dependency parsing.
#   ->  STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.edus.tokens,
#       STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.edus.postags,
#       STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.edus.arcs,
#       STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.sentence_boundaries
#       STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.paragraph_boundaries,
#       STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.arcs
python ./preprocessing1/prepare_covid19_dtb.py --input ${STORAGE_DATA}/covid19-dtb/download/covid19-dtb/dataset/v1/dev --output ${STORAGE_DATA}/covid19-dtb/preprocessed/dev
python ./preprocessing1/prepare_covid19_dtb.py --input ${STORAGE_DATA}/covid19-dtb/download/covid19-dtb/dataset/v1/test --output ${STORAGE_DATA}/covid19-dtb/preprocessed/test

# Step 2. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/covid19-dtb/preprocessed/{dev,test}/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/covid19-dtb/preprocessed/dev
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/covid19-dtb/preprocessed/test

# Step 3. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries, *.arcs) into JSON files.
#   ->  STORAGE_DATA/covid19-dtb-compiled/*.json
python ./preprocessing1/compile_to_jsons_covid19_dtb.py --input ${STORAGE_DATA}/covid19-dtb/preprocessed/dev --output ${STORAGE_DATA}/covid19-dtb-compiled/dev
python ./preprocessing1/compile_to_jsons_covid19_dtb.py --input ${STORAGE_DATA}/covid19-dtb/preprocessed/test --output ${STORAGE_DATA}/covid19-dtb-compiled/test


################################
# STAC
################################


# Step 1. (1) Extract tokenized EDUs, speaker information, sentence/paragraph boundaries, and gold annotations from the dataset; (2) perform POS tagging and syntactic dependency parsing.
#   ->  STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.edus.tokens,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.edus.postags,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.edus.arcs,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.speakers,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.sentence_boundaries,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.paragraph_boundaries,
#       STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.arcs
python ./preprocessing1/prepare_stac.py --input ${STAC}/pilot_spect --output ${STORAGE_DATA}/stac/pilot_spect
python ./preprocessing1/prepare_stac.py --input ${STAC}/socl-season1_spect --output ${STORAGE_DATA}/stac/socl-season1_spect
python ./preprocessing1/prepare_stac.py --input ${STAC}/socl-season2_spect --output ${STORAGE_DATA}/stac/socl-season2_spect
python ./preprocessing1/prepare_stac.py --input ${STAC_TEST} --output ${STORAGE_DATA}/stac/test

# Step 2. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/stac/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/stac/pilot_spect
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/stac/socl-season1_spect
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/stac/socl-season2_spect
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/stac/test

# Step 3. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.speakers, *.sentence_boundaries, *.paragraph_boundaries, *.arcs) into JSON files.
#   ->  STORAGE_DATA/stac-compiled/{pilot_spect,socl-season1_spect,socl-season2_spect,test}/*.json
python ./preprocessing1/compile_to_jsons_stac.py --input ${STORAGE_DATA}/stac/pilot_spect --output ${STORAGE_DATA}/stac-compiled/pilot_spect
python ./preprocessing1/compile_to_jsons_stac.py --input ${STORAGE_DATA}/stac/socl-season1_spect --output ${STORAGE_DATA}/stac-compiled/socl-season1_spect
python ./preprocessing1/compile_to_jsons_stac.py --input ${STORAGE_DATA}/stac/socl-season2_spect --output ${STORAGE_DATA}/stac-compiled/socl-season2_spect
python ./preprocessing1/compile_to_jsons_stac.py --input ${STORAGE_DATA}/stac/test ${STORAGE_DATA}/stac-compiled/test


################################
# Molweni
################################


# Step 1. (1) Extract tokenized EDUs, speaker information, sentence/paragraph boundaries, and gold annotations from the dataset; (2) perform POS tagging and syntactic dependency parsing.
#   ->  STORAGE_DATA/molweni/{train,dev,test}/*.edus.tokens,
#       STORAGE_DATA/molweni/{train,dev,test}/*.edus.postags,
#       STORAGE_DATA/molweni/{train,dev,test}/*.edus.arcs,
#       STORAGE_DATA/molweni/{train,dev,test}/*.speakers,
#       STORAGE_DATA/molweni/{train,dev,test}/*.sentence_boundaries,
#       STORAGE_DATA/molweni/{train,dev,test}/*.paragraph_boundaries,
#       STORAGE_DATA/molweni/{train,dev,test}/*.arcs
python ./preprocessing1/prepare_molweni.py --input ${MOLWENI}/train.json --output ${STORAGE_DATA}/molweni/train --output_filename_prefix train
python ./preprocessing1/prepare_molweni.py --input ${MOLWENI}/dev.json --output ${STORAGE_DATA}/molweni/dev --output_filename_prefix dev
python ./preprocessing1/prepare_molweni.py --input ${MOLWENI}/test.json --output ${STORAGE_DATA}/molweni/test --output_filename_prefix test

# Step 2. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/molweni/{train,dev,test}/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/molweni/train
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/molweni/dev
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/molweni/test

# Step 3. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.speakers, *.sentence_boundaries, *.paragraph_boundaries, *.arcs) into JSON files.
#   ->  STORAGE_DATA/molweni-compiled/{train,dev,test}/*.json
python ./preprocessing1/compile_to_jsons_molweni.py --input ${STORAGE_DATA}/molweni/train --output ${STORAGE_DATA}/molweni-compiled/train
python ./preprocessing1/compile_to_jsons_molweni.py --input ${STORAGE_DATA}/molweni/dev --output ${STORAGE_DATA}/molweni-compiled/dev
python ./preprocessing1/compile_to_jsons_molweni.py --input ${STORAGE_DATA}/molweni/test --output ${STORAGE_DATA}/molweni-compiled/test


################################
# Unlabeled dataset
# ACL Anthology Sentence Corpus (AASC; only abstract)
################################


# Step 1. Extract raw texts (only abstracts) from the dataset.
#   ->  STORAGE_DATA/aasc-abst/*.doc.tokens
python ./preprocessing1/prepare_aasc_abst.py --input ${AASC} --output ${STORAGE_DATA}/aasc-abst

# Step 2. Perform sentence splitting, tokenization, POS tagging, and syntactic dependency parsing and obtain paragraph boundaries.
#   ->  STORAGE_DATA/aasc-abst/*.sents.tokens,
#       STORAGE_DATA/aasc-abst/*.sents.postags,
#       STORAGE_DATA/aasc-asbt/*.sents.arcs,
#       STORAGE_DATA/aasc-abst/*.paragraph_boundaries
python ./preprocessing1/doc2sents.py --path ${STORAGE_DATA}/aasc-abst

# Step 3. Segment the documents (*.sents.tokens) into EDUs.
#   ->  STORAGE_DATA/aasc-abst/*.edus.tokens
# NOTE: We used https://github.com/PKU-TANGENT/NeuralEDUSeg to segment the documents into EDUs. (c.f., ./preprocessing1/segment.sh)

# Step 4. Split the sentence-level features into EDU-level ones and obtain sentence boundaries.
#   ->  STORAGE_DATA/aasc-abst/*.edus.postags,
#       STORAGE_DATA/aasc-abst/*.edus.arcs,
#       STORAGE_DATA/aasc-abst/*.sentence_boundaries
python ./preprocessing1/sents2edus.py --path ${STORAGE_DATA}/aasc-abst

# Step 5. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/aasc-abst/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/aasc-abst

# Step 6. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries) into JSON files.
#   ->  STORAGE_DATA/aasc-abst-compiled/*.json
python ./preprocessing1/compile_to_jsons_unannotated_corpus.py --input ${STORAGE_DATA}/aasc-abst --output ${STORAGE_DATA}/aasc-abst-compiled


###############################
# Unlabeled dataset
# CORD-19 (only abstract)
################################


# Step 1. Extract raw texts (only abstracts) from the dataset.
#   ->  STORAGE_DATA/cord19-abst/*.doc.tokens
python ./preprocessing1/prepare_cord19_abst.py --input ${CORD19} --output ${STORAGE_DATA}/cord19-abst

# Step 2. Perform sentence splitting, tokenization, POS tagging, and syntactic dependency parsing and obtain paragraph boundaries.
#   ->  STORAGE_DATA/cord19-abst/*.sents.tokens,
#       STORAGE_DATA/cord19-abst/*.sents.postags,
#       STORAGE_DATA/cord19-abst/*.sents.arcs,
#       STORAGE_DATA/cord19-abst/*.paragraph_boundaries
python ./preprocessing1/doc2sents.py --path ${STORAGE_DATA}/cord19-abst

# Step 3. Segment the documents (*.sents.tokens) into EDUs.
#   ->  STORAGE_DATA/cord19-abst/*.edus.tokens
# NOTE: We used https://github.com/PKU-TANGENT/NeuralEDUSeg to segment the documents into EDUs. (c.f., ./preprocessing1/segment.sh)

# Step 4. Split the sentence-level features into EDU-level ones and obtain sentence boundaries.
#   ->  STORAGE_DATA/cord19-abst/*.edus.postags,
#       STORAGE_DATA/cord19-abst/*.edus.arcs,
#       STORAGE_DATA/cord19-abst/*.sentence_boundaries
python ./preprocessing1/sents2edus.py --path ${STORAGE_DATA}/cord19-abst

# Step 5. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/cord19-abst/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/cord19-abst

# Step 6. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.sentence_boundaries, *.paragraph_boundaries) into JSON files.
#   ->  STORAGE_DATA/cord19-abst-compiled/*.json
python ./preprocessing1/compile_to_jsons_unannotated_corpus.py --input ${STORAGE_DATA}/cord19-abst --output ${STORAGE_DATA}/cord19-abst-compiled


################################
# Unlabeled dataset
# Ubuntu Dialogue Corpus
################################


# Step 0. Download Ubuntu Dialogue Corpus and extract information.
#   ->  STORAGE_DATA/ubuntu-dialogue-corpus/extracted/*.tsv
python ./preprocessing1/download_ubuntu_dialogue_corpus.py \
    --input ${UDC} \
    --output ${STORAGE_DATA}/ubuntu-dialogue-corpus/extracted \
    --n_dialogues 100000 \
    --min_dialogue_length 7 \
    --max_dialogue_length 16 \
    --max_utterance_length 20 \
    --max_speakers 9

# Step 1. (1) Extract utterances (EDUs), speaker/listener information, and sentence/paragraph boundaries; (2) perform tokenization, POS tagging, and syntactic dependency parsing.
#   ->  STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.edus.tokens,
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.edus.postags,
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.edus.arcs,
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.speakers,
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.listeners
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.sentence_boundaries,
#       STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.paragraph_boundaries
python ./preprocessing1/prepare_ubuntu_dialogue_corpus.py --input ${STORAGE_DATA}/ubuntu-dialogue-corpus/extracted --output ${STORAGE_DATA}/ubuntu-dialogue-corpus/preprocessed

# Step 2. Extract EDU-level syntactic head information from the dependency arcs.
#   ->  STORAGE_DATA/ubuntu-dialogue-corpus/preprocessed/*.edus.heads
python ./preprocessing1/extract_head_from_arcs.py --path ${STORAGE_DATA}/ubuntu-dialogue-corpus/preprocessed

# Step 3. Compile the extacted information (*.edus.tokens, *.edus.postags, *.edus.arcs, *.edus.heads, *.speakers, *.listeners, *.sentence_boundaries, *.paragraph_boundaries) into JSON files.
#   ->  STORAGE_DATA/ubuntu-dialogue-corpus-compiled/*.json
python ./preprocessing1/compile_to_jsons_ubuntu_dialogue_corpus.py --input ${STORAGE_DATA}/ubuntu-dialogue-corpus/preprocessed --output ${STORAGE_DATA}/ubuntu-dialogue-corpus-compiled


