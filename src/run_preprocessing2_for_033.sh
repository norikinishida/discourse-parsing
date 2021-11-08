#!/usr/bin/env sh


################################
# SciDTB
################################


# Results (for 033):
#   ->  /path/to/caches-033/mapped-scidtb.{train,dev,test}-{"",gold,second_annotate}.scibert_scivocab_uncased.npy
#       /path/to/caches-033/mapped-scidtb.{train,dev,test}-{"",gold,second_annotate}.gold.arcs
python ./preprocessing2/preprocess_scidtb_for_033.py


################################
# COVID19-DTB
################################


# Results (for 033):
#   ->  /path/to/caches-033/mapped-covid19-dtb.{dev,test}.scibert_scivocab_uncased.npy
#       /path/to/caches-033/mapped-covid19-dtb.{dev,test}.gold.arcs
#       /path/to/caches-033/mapped-covid19-dtb.relations.vocab.txt
python ./preprocessing2/preprocess_covid19_dtb_for_033.py


################################
# STAC
################################


# Results (for 033):
#   ->  /path/to/caches-033/mapped-stac.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-033/mapped-stac.{train,dev,test}.gold.arcs
python ./preprocessing2/preprocess_stac_for_033.py


################################
# Molweni
################################


# Results (for 033):
#   ->  /path/to/caches-033/molweni.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-033/molweni.{train,dev,test}.gold.arcs
#       /path/to/caches-033/molweni.relations.vocab.txt
python ./preprocessing2/preprocess_molweni_for_033.py


################################
# Unlabeled dataset
# ACL Anthology Sentence Corpus (AASC; only abstract)
################################


# Results (for 033):
#   ->  /path/to/caches-033/aasc-abst.scibert_scivocab_uncased.npy
python ./preprocessing2/preprocess_aasc_abst_for_033.py


###############################
# Unlabeled dataset
# CORD-19 (only abstract)
################################


# Results (for 033):
#   ->  /path/to/caches-033/cord19-abst.scibert_scivocab_uncased.npy
python ./preprocessing2/preprocess_cord19_abst_for_033.py


################################
# Unlabeled dataset
# Ubuntu Dialogue Corpus
################################


# Results (for 033):
#   ->  /path/to/caches-033/ubuntu-dialogue-corpus.bert-base-cased.npy
python ./preprocessing2/preprocess_ubuntu_dialogue_corpus_for_033.py

