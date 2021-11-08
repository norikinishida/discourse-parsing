#!/usr/bin/env sh

STORAGE=/home/nishida/storage/projects/discourse/discourse-parsing/data

#####################
# PTB-WSJ
#####################

mkdir ../data/ptbwsj-wo-rstdt.out

python run.py --segment --input_files ${STORAGE}/ptbwsj-wo-rstdt/*.sents.tokens --result_dir ../data/ptbwsj-wo-rstdt.out/

python rename.py --input_files ../data/ptbwsj-wo-rstdt.out/*.sents.tokens --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

cp ../data/ptbwsj-wo-rstdt.out/*.edus.tokens ${STORAGE}/ptbwsj-wo-rstdt

#####################
# ACL Anthology Sentence Corpus (AASC; only abstract)
#####################

mkdir ../data/aasc-abst.out

find ${STORAGE}/aasc-abst -name *.sents.tokens > ../data/aasc-abst.out/filelist.txt
python run.py --segment --input_file_list ../data/aasc-abst.out/filelist.txt --result_dir ../data/aasc-abst.out/

find ../data/aasc-abst.out -name *.sents.tokens > ../data/aasc-abst.out/filelist2.txt
python rename.py --input_file_list ../data/aasc-abst.out/filelist2.txt --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

find ../data/aasc-abst.out -name *.edus.tokens -print0 | xargs -0 -I {} cp {} ${STORAGE}/aasc-abst

#####################
# CORD-19 (only abstract)
#####################

mkdir ../data/cord19-abst.out

find ${STORAGE}/cord19-abst -name *.sents.tokens > ../data/cord19-abst.out/filelist.txt
python run.py --segment --input_file_list ../data/cord19-abst.out/filelist.txt --result_dir ../data/cord19-abst.out/

find ../data/cord19-abst.out -name *.sents.tokens > ../data/cord19-abst.out/filelist2.txt
python rename.py --input_file_list ../data/cord19-abst.out/filelist2.txt --src_ext ".sents.tokens" --dst_ext ".edus.tokens"

find ../data/cord19-abst.out -name *.edus.tokens -print0 | xargs -0 -I {} cp {} ${STORAGE}/cord19-abst

