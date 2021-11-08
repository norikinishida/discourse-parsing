# discourse-parsing

This repository is an implementation of discourse parsers:

- Arc-factored discourse dependency parser using a BERT-based biaffine model and multi-stage Eisner decoding
- Shift-reduce discourse dependency parser using a BERT-based arc-standard model
- Stack-pointer discourse dependency parser using a BERT-based pointer network
- Bootstrapping algorithms (self-training, co-training, tri-training, and assymmetric tri-training) of discourse dependency parsers for unsupervised domain adaptation (Nishida and Matsumoto, 2021) (to appear in TACL)

## Requirements

- numpy
- pytorch
- huggingface
- spacy
- jsonlines
- pyprind
- https://github.com/norikinishida/utils
- https://github.com/norikinishida/treetk

## Configuration

The following files need to be editted according to your environment.

- `./config/path.conf`
- `./run_preprocessing1.sh`

## Preprocessing

Overall steps:

1. Download dataset, e.g., RST-DT, SciDTB, COVID19-DTB, STAC, Molweni, etc.
1. Extract (or predict) EDU segments, syntactic features (POS tags, syntactic dependency arcs, and EDU-level syntactic heads), sentence/paragraph boundaries, and gold constituent/dependency structures.
1. Compile the extracted (or predicted) information as JSON files. Each JSON file corresponds to each data instance in the dataset.
1. Preprocess the JSON files using the BERT tokenizer and save the preprocessed datasets (`*.npy`), gold structures (`*.ctrees`, `*.arcs`), and discourse relation class list (`*.vocab.txt`) to the `<caches-*>` directory specified in the configuration file (`./config/path.conf`).

See `./run_preprocessing1.sh` and `./run_preprocessing2.sh` for more detailed examples on Step 1-3 and Step 4, respectively.

### JSON format (RST-DT, wsj\_1185)

```
{
    "edus": [
        {
            "tokens": "A group",
            "postags": "DT NN",
            "arcs": "2-1-det 25-2-nsubj",
            "head": "1"
        },
        {
            "tokens": "including Gene E. Phillips , former chairman of Southmark Corp. , and William S. Friedman , former vice chairman of Southmark ,",
            "postags": "VBG NNP NNP NNP , JJ NN IN NNP NNP , CC NNP NNP NNP , JJ NN NN IN NNP ,",
            "arcs": "2-3-prep 6-4-compound 6-5-compound 3-6-pobj 6-7-punct 9-8-amod 6-9-appos 9-10-prep 12-11-compound 10-12-pobj 6-13-punct 6-14-cc 17-15-compound 17-16-compound 6-17-conj 17-18-punct 21-19-amod 21-20-compound 17-21-appos 21-22-prep 22-23-pobj 17-24-punct",
            "head": "0"
        },
        {
            "tokens": "lowered its stake in the Dallas real estate concern to 7.7 % ,",
            "postags": "VBD PRP$ NN IN DT NNP JJ NN NN IN CD NN ,",
            "arcs": "0-25-ROOT 27-26-poss 25-27-dobj 27-28-prep 33-29-det 33-30-nmod 32-31-amod 33-32-compound 28-33-pobj 25-34-prep 36-35-nummod 34-36-pobj 25-37-punct",
            "head": "0"
        },
        ...
    ],
    "sentence_boundaries": [
        "0 3",
        "4 5",
        "6 7"
    ],
    "paragraph_boundaries": [
        "0 0",
        "1 1",
        "2 2"
    ],
    "nary_sexp": "( <elaboration-additional,N/S> ( <elaboration-additional,N/S> ( <attribution,N/S> ( <Same-Unit,N/N> ( <elaboration-set-member-e,N/S> 0 1 ) 2 ) 3 ) ( <attribution,S/N> 4 5 ) ) ( <attribution,S/N> 6 7 ) )",
    "bin_sexp": "( <elaboration-additional,N/S> ( <elaboration-additional,N/S> ( <attribution,N/S> ( <Same-Unit,N/N> ( <elaboration-set-member-e,N/S> 0 1 ) 2 ) 3 ) ( <attribution,S/N> 4 5 ) ) ( <attribution,S/N> 6 7 ) )",
    "arcs": "0-1-<root> 1-2-elaboration-set-member-e 1-3-Same-Unit 1-4-attribution 6-5-attribution 1-6-elaboration-additional 8-7-attribution 1-8-elaboration-additional"
}
```

### Preprocessing without syntactic features and paragraph boundaries

You can use JSON files without syntactic features (i.e., "postags", "arcs", and "head" keys in each EDU entry) and paragraph boundaries ("paragraph_boundaries" key), as follows:

```
{
    "edus": [
        {
            "tokens": "A group"
        },
        {
            "tokens": "including Gene E. Phillips , former chairman of Southmark Corp. , and William S. Friedman , former vice chairman of Southmark ,"
        },
        {
            "tokens": "lowered its stake in the Dallas real estate concern to 7.7 % ,"
        },
        ...
    ],
    "sentence_boundaries": [
        "0 3",
        "4 5",
        "6 7"
    ],
    "nary_sexp": "( <elaboration-additional,N/S> ( <elaboration-additional,N/S> ( <attribution,N/S> ( <Same-Unit,N/N> ( <elaboration-set-member-e,N/S> 0 1 ) 2 ) 3 ) ( <attribution,S/N> 4 5 ) ) ( <attribution,S/N> 6 7 ) )",
    "bin_sexp": "( <elaboration-additional,N/S> ( <elaboration-additional,N/S> ( <attribution,N/S> ( <Same-Unit,N/N> ( <elaboration-set-member-e,N/S> 0 1 ) 2 ) 3 ) ( <attribution,S/N> 4 5 ) ) ( <attribution,S/N> 6 7 ) )",
    "arcs": "0-1-<root> 1-2-elaboration-set-member-e 1-3-Same-Unit 1-4-attribution 6-5-attribution 1-6-elaboration-additional 8-7-attribution 1-8-elaboration-additional"
}
```

In training (and evaluation), please set ```use_edu_head_information = false``` in an appropriate configuration file (e.g., `./config/arcfactored.conf`).

## Training

Experiment configurations are found in `./config` (e.g., `arcfactored.conf` and `shiftreduce.conf`).
You can also add your own configuration.
Choose a configuration name (e.g., `biaffine_scibert_scidtb`), and run

```
python main_<parser_type>.py --gpu <gpu_id> --config <config_name> --actiontype train
```

The following command is an example to train an arc-factored discourse dependency parser using a biaffine model and SciBERT on SciDTB:

```
python main_arcfactored.py --gpu 0 --config biaffine_scibert_scidtb --actiontype train
```

Results are stored in the `<results>/<parser_type>.<config_name>` directory, where `<results>` is specified in `./config/path.conf`.

Outputs:
- Log: `<results>/<parser_type>.<config_name>/<prefix>.training.log`
- Training losses: `<results>/<parser_type>.<config_name>/<prefix>.training.jsonl`
- Model parameters: `<results>/<parser_type>.<config_name>/<prefix>.model`
- Validation outputs: `<results>/<parser_type>.<config_name>/<prefix>.validation.arcs`
- Validation scores: `<results>/<parser_type>.<config_name>/<prefix>.validation.jsonl`

`<prefix>` is automatically determined based on the execution time, .e.g, `Jun09_01-23-45`.

## Evaluation

The trained model can be evaluated on the test dataset using the following command:

```
python main_<parser_type>.py --gpu <gpu_id> --config <config_name> --prefix <prefix> --actiontype evaluate
```

The following command is an example to evaluate the above model on the SciDTB test set:

```
python main_arcfactored.py --gpu 0 --config biaffine_scibert_scidtb --prefix Jun09_01-23-45 --actiontype evaluate
```

Results are stored in the `<results>/<parser_type>.<config_name>` directory.

Outputs:

- Log: `<results>/<parser_type>.<config_name>/<prefix>.evaluation.log`
- Evaluation outputs: `<results>/<parser_type>.<config_name>/<prefix>.evaluation.arcs`
- Evaluation outputs (SciDTB format): `<results>/<parser_type>.<config_name>/<prefix>.evaluation.arcs.dep/*.dep`
- Evaluation scores: `<results>/<parser_type>.<config_name>/<prefix>.evaluation.json`


