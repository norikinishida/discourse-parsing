#!/usr/bin/env sh


################################
# RST-DT
################################


# Results:
#   ->  /path/to/caches-dep/rstdt.{train,test}.bert-base-cased.npy
#       /path/to/caches-dep/rstdt.{train,test}.gold.labeled.nary.ctrees
#       /path/to/caches-dep/rstdt.{train,test}.gold.labeled.bin.ctrees
#       /path/to/caches-dep/rstdt.{train,test}.gold.arcs
#       /path/to/caches-dep/rstdt.constituency_relations.vocab.txt
#       /path/to/caches-dep/rstdt.constituency_nuclearities.vocab.txt
#       /path/to/caches-dep/rstdt.dependency_relations.vocab.txt
python ./preprocessing2/preprocess_rstdt.py


################################
# Unlabeled dataset
# PTB-WSJ
################################


# Results:
#       /path/to/caches-dep/ptbwsj-wo-rstdt.bert-base-cased.npy
python ./preprocessing2/preprocess_ptbwsj_wo_rstdt.py


################################
# SciDTB
################################


# Results:
#   ->  /path/to/caches-dep/scidtb.{train,dev,test}-{"",gold,second_annotate}.scibert_scivocab_uncased.npy
#       /path/to/caches-dep/scidtb.{train,dev,test}-{"",gold,second_annotate}.gold.arcs
#       /path/to/caches-dep/scidtb.relations.vocab.txt
python ./preprocessing2/preprocess_scidtb.py


################################
# COVID19-DTB
################################


# Results:
#   ->  /path/to/caches-dep/covid19-dtb.{dev,test}.scibert_scivocab_uncased.npy
#       /path/to/caches-dep/covid19-dtb.{dev,test}.gold.arcs
#       /path/to/caches-dep/covid19-dtb.relations.vocab.txt
python ./preprocessing2/preprocess_covid19_dtb.py


################################
# STAC
################################


# Results:
#   ->  /path/to/caches-dep/stac.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-dep/stac.{train,dev,test}.gold.arcs
#       /path/to/caches-dep/stac.relations.vocab.txt
python ./preprocessing2/preprocess_stac.py


################################
# Molweni
################################


# Results:
#   ->  /path/to/caches-dep/molweni.{train,dev,test}.bert-base-cased.npy
#       /path/to/caches-dep/molweni.{train,dev,test}.gold.arcs
#       /path/to/caches-dep/molweni.relations.vocab.txt
python ./preprocessing2/preprocess_molweni.py
