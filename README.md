# Contextual Abuse Dataset

This is the repository for:

 **Introducing CAD: the Contextual Abuse Dataset** 
 Bertie Vidgen, Dong Nguyen, Helen Margetts, Patricia Rossini, Rebekah Tromble,
 NAACL 2021
  [Paper (ACL Anthology)](https://www.aclweb.org/anthology/2021.naacl-main.182/)

The data + models have been uploaded to Zenodo: http://doi.org/10.5281/zenodo.4881008

## Install

See the `environments` folder for the exact environments that were used. The LR experiments were performed on a local machine. The experiments with BERT and DistilBERT were performed on a computing cluster.

* scikit-learn : 0.22.1
* mlflow:  1.11.0
* nlp: 0.4.0 
* pandas
* transformers 3.2.0

## Directory structure
 
Directories:

* `environments`:  Environments used for performing the experiments (Conda).
* `experiments`: The models and model outputs (*not included on Github, get this on Zenodo*)
* `src`: Python code
* `data`: data (*not included on Github, get this on Zenodo*)

**Note about the dataset (v1 vs. v1.1)**
`cad_v1` was used to produce the results in the NAACL 2021 paper.
We identified some minor issues later. This affects the primary and secondary categories of 95 entries. The new version CAD v1.1 is also provided, based on the changes recorded in `data/errata_v1_to_v1_1`.


**Overview**

```
.
├── data
│   ├── cad_v1.tsv
│   ├── cad_v1_train.tsv
│   ├── cad_v1_dev.tsv
│   ├── cad_v1_test.tsv
│   ├── cad_v1_1.tsv
│   ├── cad_v1_1_train.tsv
│   ├── cad_v1_1_dev.tsv
│   ├── cad_v1_1_test.tsv
│   ├── test
│   │   ├──  test_data.tsv
│   │   ├──  test_data_test.tsv
│   │   ├──  test_predictions.tsv
│   ├── errata_v1_to_v1_1
│   ├── cad_codebook_april2021.pdf
├── experiments
│   ├── lr 
│   ├── test_dbert
│   │   ├── cad_final_601_DistilBERT_0_0.00005_False
│   │   ├── cad_final_78_DistilBERT_0_0.00005_False
│   │   ├── cad_final_8923_DistilBERT_0_0.00005_False
│   │   ├── cad_final_92064_DistilBERT_0_0.00005_False
│   │   ├── cad_final_971242_DistilBERT_0_0.00005_False
│   │   ├── test_output_label_analysis_dbert_aggregated.txt
│   │   ├── test_output_metrics_dbert_aggregated.txt
│   ├── test_bert
│   │   ├── cad_final_601_BERT_0.01_0.00003_False
│   │   ├── cad_final_78_BERT_0.01_0.00003_False
│   │   ├── cad_final_8923_BERT_0.01_0.00003_False
│   │   ├── cad_final_92064_BERT_0.01_0.00003_False
│   │   ├── cad_final_971242_BERT_0.01_0.00003_False
│   │   ├── test_output_label_analysis_bert_aggregated.txt
│   │   ├── test_output_metrics_bert_aggregated.txt
│   │   ├── bert_error_analysis.txt
│   ├── tune_dbert_summary.txt
│   ├── test_dbert_summary.txt
│   ├── tune_bert_summary.txt
│   ├── test_bert_summary.txt
├── environments
├── src

```

# Data


**Data exploration:**

 `data_analysis.ipynb` A notebook that explores the data and prints out statistics.
 

**Writing the splits:**


Given the full file, we can generate individual files for each split (containing id, text, labels) using:

```
python classification_util.py write_splits --input_file="../data/cad_v1.tsv"
```

The generated files are also provided as `../data/cad_v1_train.tsv`,  `../data/cad_v1_dev.tsv` and  `../data/cad_v1_test.tsv`

**Data checks:**

``` 
python check_data.py --input_file "../data/cad_v1.tsv"
```

# Logistic regression

Logistic Regression is implemented using scikit learn (0.22.1).
The code makes use of the `mlflow` library. 
All results are initially stored in a `mlflow` folder (in the current working directory). 

The final model and results are provided in `experiments/lr`

**Tune parameters**

```
python lr.py tune
```

To view results:

Let's assume the results (the `mlflow` folder) is stored in `experiments/lr/tuning`. Go to this folder and run:

```
mlflow ui 
```

See results at `http://localhost:5000/`

**Testing**

Based on tuning, the following hyperparameters were used: `max_df=0.95, min_df=5, C=0.25`
 
```
python lr.py test
```


**Analysis**

Both calls below work in the same way. Based on the file with the predictions for each test instance ( `model_outputs.txt`), they will compute additional evaluation statistics and save the results to the same directory of `model_outputs.txt`.


Calculate some additional evaluation metrics, such as recall for labels based on previous or current content

```
python classification_analysis.py general_analysis --predictions_file "../experiments/lr/test/mlruns/0/d0a17039a5104dd8a32b6f8980fe1ace/artifacts/model_outputs.txt"
```

Do an analysis of the predicted labels (e.g. does the model often underpredict the number of labels?

```
python classification_analysis.py label_analysis --predictions_file "../experiments/lr/test/mlruns/0/d0a17039a5104dd8a32b6f8980fe1ace/artifacts/model_outputs.txt"
```



# DistilBERT

The implementation uses the Transformers library (3.2.0).
The `experiments/test_dbert` contains the predictions and analyses files for the five final models (because we used 5 different random seeds). 
The folder of the best performing model `experiments/test_dbert/cad_final_8923_DistilBERT_0_0.00005_False` also contains the trained model.

**Finetuning with different hyperparameters on train and dev**


```
python hatespeech_finetuning_multilabel.py tune DistilBERT [output_dir]
```

*Analyze results*

Print out the results (macro and micro F1 scores) for each parameter setting. The method assumes each subdir in the specified directory (replace with the correct path) contains one model:

```
python analyze_transformers.py analyze_runs --input_pattern "../experiments/tune_dbert/*"
```

The results are printed to standard output. We have provided the output in `experiments/tune_dbert_summary.txt`

 
*Best setting:* learning rate 5e-5, weight decay 0

**Finetune on train with selected hyper parameters, apply model on test**

First train and test DistilBERT (5 random seeds) on the train and test set with the selected parameters

```
python hatespeech_finetuning_multilabel.py test DistilBERT [output_dir]
```

Each model is stored in a directory. The directory names contain the different parameters used, including the random seed.

**Analysis of the results on the set set**


- Print out average micro and macro scores for all models who match the specified input pattern (in this case, same settings for DistilBERT runs but with different random seeds). 

```
python analyze_transformers.py analyze_runs --input_pattern "../experiments/test_dbert/cad_final_*_DistilBERT_0_0.00005_False"
```
 
 The results are provided in `experiments/test_dbert_summary.txt`
 
- To do more fine-grained analyses, run model again on test set and print out predictions:
 
```
python analyze_transformers.py write_predictions --input_pattern "../experiments/test_dbert/cad_final_*_DistilBERT_0_0.00005_False" --model_type DistilBERT
```
 
In each folder, the predictions are stored in a file called `test_output.txt`, with the format `[postid]\t[multi hot encoding of predictions]`
 
 
 - For each run, read in the predictions file and write out all the metrics:
 
```
python analyze_transformers.py print_metrics --input_pattern "../experiments/test_dbert/cad_final_*_DistilBERT_0_0.00005_False" --analysis_type "general_analysis"
```

This generates two files in each model folder: `test_output_analysis.txt` and `test_output_metrics.txt`.
Also run this once with `analysis_type="label_analysis"`, this will output statistics related to the number of labels predicted, and will generate `test_output_label_analysis.txt`


For each DistilBERT model, we now have files with evaluation metrics. 
The following reads in the specified file for each model, and writes out mean and stds for each metric to standard output:

```
python analyze_transformers.py print_aggr_analysis --input_pattern "../experiments/test_dbert/cad_final_*_DistilBERT_0_0.00005_False" --metrics_file "test_output_metrics.txt"
```

The output is provided in `../experiments/test_dbert/test_output_metrics_dbert_aggregated.txt`

# BERT

The `experiments/test_bert` contains the predictions and analyses files for the five final models (because we used 5 different random seeds). The folder of the best performing model `experiments/test_bert/cad_final_971242_BERT_0.01_0.00003_False` also contains the trained model.


**Finetuning with different hyperparameters on train and dev**

Used command:

```
python hatespeech_finetuning_multilabel.py tune BERT [output_dir]
```


Summary results are in: `../experiments/tune_bert_summary.txt`


*Best setting:* learning rate 3e-5, weight decay 0.01

**Finetune on train with selected hyper parameters, apply model on test**

Run the models on train and test with the selected parameters

```
python hatespeech_finetuning_multilabel.py test BERT [output_dir]
```

**Analysis of the results on the set set**

Assuming that ``"../experiments/test_bert`` is the output directory, read the log files and print macro/micro averages.

```
python analyze_transformers.py analyze_runs --input_pattern "../experiments/test_bert/cad_final_*_BERT_0.01_0.00003_False"
```

 The results are provided in `experiments/test_bert_summary.txt`
 
 
Apply BERT again on the test set, now write predictions for each instance to an output file (`test_output.txt`) in the model directory.

```
python analyze_transformers.py write_predictions --input_pattern "../experiments/test_bert/cad_final_*_BERT_0.01_0.00003_False" --model_type BERT
```

For each run, read in the predictions file and write out all the metrics:

```
python analyze_transformers.py print_metrics --input_pattern "../experiments/test_bert/cad_final_*_BERT_0.01_0.00003_False" --analysis_type "general_analysis"
```

This generates two files in each model folder: `test_output_analysis.txt` and `test_output_metrics.txt`.
Also run this once with `analysis_type="label_analysis"`, this will output statistics related to the number of labels predicted and will generate `test_output_label_analysis.txt`


Now, by reading the file with evaluation metrics for each BERT model (we have 5, because we used 5 different random seeds), we can now write out mean and stds for each metric to standard output:

```
python analyze_transformers.py print_aggr_analysis --input_pattern "../experiments/test_bert/cad_final_*_BERT_0.01_0.00003_False" --metrics_file "test_output_metrics.txt"
```

The output is provided in `../experiments/test_bert/test_output_metrics_bert_aggregated.txt`


Error analysis:

```
python analyze_transformers.py print_error_analysis --input_pattern "../experiments/test_bert/cad_final_*_BERT_0.01_0.00003_False" 
```

# Citation

```
@inproceedings{vidgen-etal-2021-introducing,
    title = "Introducing {CAD}: the Contextual Abuse Dataset",
    author = "Vidgen, Bertie  and
      Nguyen, Dong  and
      Margetts, Helen  and
      Rossini, Patricia  and
      Tromble, Rebekah",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.182",
    pages = "2289--2303",
    abstract = "Online abuse can inflict harm on users and communities, making online spaces unsafe and toxic. Progress in automatically detecting and classifying abusive content is often held back by the lack of high quality and detailed datasets.We introduce a new dataset of primarily English Reddit entries which addresses several limitations of prior work. It (1) contains six conceptually distinct primary categories as well as secondary categories, (2) has labels annotated in the context of the conversation thread, (3) contains rationales and (4) uses an expert-driven group-adjudication process for high quality annotations. We report several baseline models to benchmark the work of future researchers. The annotated dataset, annotation guidelines, models and code are freely available.",
}
```
 [Paper (ACL Anthology)](https://www.aclweb.org/anthology/2021.naacl-main.182/)

# Notes

**Data**

* `contextual_abuse_dataset.py`: Loads in the dataset. Does a few preprocessing steps.
* `check_data.py`: Contains methods to perform a few checks on the data. 
* `data_analysis.ipynb:` A notebook that explores the data and prints out statistics. 

*Notes:*

* check_thread_structure: missing: 618, 02, 03
* Slurs are excluded from the experiments. If a post is only annotated as Slur, it becomes Neutral. Otherwise, the Slur label is just ignored. 
* cad_27367 was excluded from the experiments because only the image is abusive.
* First time loading the dataset can give a `TypeError: argument of type 'Value' is not iterable` error. 

**Logistic Regression**

* `lr.py`: Runs the logistic regression classifier (both tuning and testing). Writes the models and outputs to files.

**Transformers**

* `hatespeech_finetuning_multilabel.py` Script to finetune BERT and DistilBERT. 
* `distillbert_multilabel.py` adaptation of DistilBert to support multilabel classification.
* `bert_multilabel.py` adaptation of Bert to support multilabel classification.
* `analyze_transformers.py` analyze the results of the transformer models. 

**Utilities**
* `classification_util.py`: Utilities for classification (writing output and evaluation metrics to file).
* `classification_analysis.py`: Analyze classification results.

**Tests**
* `tests.py`: Checks a few processing methods.


