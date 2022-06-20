#!/usr/bin/env sh


################################
# SciDTB
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/mapped-scidtb.{train,dev,test}-{"",gold,second_annotate}.scibert_scivocab_uncased.npy
#       /path/to/caches-tacl2022/mapped-scidtb.{train,dev,test}-{"",gold,second_annotate}.gold.arcs
python ./preprocessing2/preprocess_scidtb_for_tacl2022.py


################################
# COVID19-DTB
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/mapped-covid19-dtb.{dev,test}.scibert_scivocab_uncased.npy
#       /path/to/caches-tacl2022/mapped-covid19-dtb.{dev,test}.gold.arcs
#       /path/to/caches-tacl2022/mapped-covid19-dtb.relations.vocab.txt
python ./preprocessing2/preprocess_covid19_dtb_for_tacl2022.py


################################
# STAC
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/mapped-stac.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-tacl2022/mapped-stac.{train,dev,test}.gold.arcs
python ./preprocessing2/preprocess_stac_for_tacl2022.py


################################
# Molweni
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/molweni.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-tacl2022/molweni.{train,dev,test}.gold.arcs
#       /path/to/caches-tacl2022/molweni.relations.vocab.txt
python ./preprocessing2/preprocess_molweni_for_tacl2022.py


################################
# Unlabeled dataset
# ACL Anthology Sentence Corpus (AASC; only abstract)
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/aasc-abst.scibert_scivocab_uncased.npy
python ./preprocessing2/preprocess_aasc_abst_for_tacl2022.py


###############################
# Unlabeled dataset
# CORD-19 (only abstract)
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/cord19-abst.scibert_scivocab_uncased.npy
python ./preprocessing2/preprocess_cord19_abst_for_tacl2022.py


################################
# Unlabeled dataset
# Ubuntu Dialogue Corpus
################################


# Results (for tacl2022):
#   ->  /path/to/caches-tacl2022/ubuntu-dialogue-corpus.bert-base-cased.npy
python ./preprocessing2/preprocess_ubuntu_dialogue_corpus_for_tacl2022.py

