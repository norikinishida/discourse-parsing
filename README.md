# discourse-parsing

This repository is an implementation of discourse parsers:

- Arc-factored discourse dependency parser using a BERT-based biaffine model and multi-stage Eisner decoding
- Shift-reduce discourse dependency parser using a BERT-based arc-standard model
- Stack-pointer discourse dependency parser using a BERT-based pointer network
- Bootstrapping algorithms (self-training, co-training, tri-training, and assymmetric tri-training) of discourse dependency parsers for unsupervised domain adaptation ([Nishida and Matsumoto, 2022](https://doi.org/10.1162/tacl_a_00451))

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

See `./run_preprocessing1.sh` and `./run_preprocessing2.sh` (or `./run_preprocessing2_for_tacl2022`) for more detailed examples on Step 1-3 and Step 4, respectively.

### JSON format (RST-DT, wsj\_0644)

```json
{
    "edus": [
        {
            "tokens": "Bruce W. Wilkinson , president and chief executive officer , was named to the additional post of chairman of this architectural and design services concern .",
            "postags": "NNP NNP NNP , NN CC JJ JJ NN , VBD VBN IN DT JJ NN IN NN IN DT JJ CC NN NNS NN .",
            "arcs": "3-1-compound 3-2-compound 12-3-nsubjpass 3-4-punct 3-5-appos 5-6-cc 9-7-amod 9-8-amod 5-9-conj 3-10-punct 12-11-auxpass 0-12-ROOT 12-13-prep 16-14-det 16-15-amod 13-16-pobj 16-17-prep 17-18-pobj 18-19-prep 25-20-det 25-21-amod 21-22-cc 24-23-compound 21-24-conj 19-25-pobj 12-26-punct",
            "head": "11"
        },
        {
            "tokens": "Mr. Wilkinson , 45 years old , succeeds Thomas A. Bullock , 66 ,",
            "postags": "NNP NNP , CD NNS JJ , VBZ NNP NN NNP , CD ,",
            "arcs": "2-1-compound 8-2-nsubj 2-3-punct 5-4-nummod 6-5-npadvmod 2-6-amod 2-7-punct 0-8-ROOT 11-9-compound 11-10-compound 8-11-dobj 11-12-punct 11-13-appos 11-14-punct",
            "head": "7"
        },
        {
            "tokens": "who is retiring as chairman",
            "postags": "WP VBZ VBG IN NN",
            "arcs": "17-15-nsubj 17-16-aux 11-17-relcl 17-18-prep 18-19-pobj",
            "head": "2"
        },
        {
            "tokens": "but will continue as a director and chairman of the executive committee .",
            "postags": "CC MD VB IN DT NN CC NN IN DT JJ NN .",
            "arcs": "17-20-cc 22-21-aux 17-22-conj 22-23-prep 25-24-det 23-25-pobj 25-26-cc 25-27-conj 25-28-prep 31-29-det 31-30-amod 28-31-pobj 8-32-punct",
            "head": "0"
        }
    ],
    "sentence_boundaries": "0-0 1-3",
    "paragraph_boundaries": "0-1",
    "nary_sexp": "( <circumstance,N/S> 0 ( <elaboration-additional-e,N/S> 1 ( <concession,S/N> 2 3 ) ) )",
    "bin_sexp": "( <circumstance,N/S> 0 ( <elaboration-additional-e,N/S> 1 ( <concession,S/N> 2 3 ) ) )",
    "arcs": "0-1-<root> 1-2-circumstance 4-3-concession 2-4-elaboration-additional-e"
}
```

### Preprocessing without syntactic features and paragraph boundaries

You can use JSON files without syntactic features (i.e., "postags", "arcs", and "head" keys in each EDU entry) and paragraph boundaries ("paragraph_boundaries" key), as follows:

```json
{
    "edus": [
        {
            "tokens": "Bruce W. Wilkinson , president and chief executive officer , was named to the additional post of chairman of this architectural and design services concern .",
        },
        {
            "tokens": "Mr. Wilkinson , 45 years old , succeeds Thomas A. Bullock , 66 ,",
        },
        {
            "tokens": "who is retiring as chairman",
        },
        {
            "tokens": "but will continue as a director and chairman of the executive committee .",
        }
    ],
    "sentence_boundaries": "0-0 1-3",
    "nary_sexp": "( <circumstance,N/S> 0 ( <elaboration-additional-e,N/S> 1 ( <concession,S/N> 2 3 ) ) )",
    "bin_sexp": "( <circumstance,N/S> 0 ( <elaboration-additional-e,N/S> 1 ( <concession,S/N> 2 3 ) ) )",
    "arcs": "0-1-<root> 1-2-circumstance 4-3-concession 2-4-elaboration-additional-e"
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
- Training losses: `<results>/<parser_type>.<config_name>/<prefix>.train.losses.jsonl`
- Model parameters: `<results>/<parser_type>.<config_name>/<prefix>.model`
- Validation outputs: `<results>/<parser_type>.<config_name>/<prefix>.dev.pred.arcs`
- Validation scores: `<results>/<parser_type>.<config_name>/<prefix>.dev.eval.jsonl`

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
- Evaluation outputs: `<results>/<parser_type>.<config_name>/<prefix>.test.pred.arcs`
- Evaluation outputs (SciDTB format): `<results>/<parser_type>.<config_name>/<prefix>.test.pred.arcs.dep/*.dep`
- Evaluation scores: `<results>/<parser_type>.<config_name>/<prefix>.test.eval.json`


