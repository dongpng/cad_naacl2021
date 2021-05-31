import argparse
import csv
import json
import os

from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd


import classification_analysis
import classification_util
import contextual_abuse_dataset


###############################
# Helper functions to analyze the outputs of the models
###############################

def print_aggr_analysis(input_pattern, test_file):
    """
    Multiple models were trained (with different random seeds).
    This method aggregates across the individual output metrics
    and prints (to command line) averages and standard deviations

    args:
       input_pattern: input pattern to the model files,
                      for example "../cad_final_*_DistilBERT_0.01_0.00004_False"
       test_file: name of the file containing the output metrics (tab separated)
                      for example test_output_metrics.txt
                      first lines should contain info about the data files
                      remaining lines contain the metric values
    """
    
    # dictionary with values for each metric
    # <metric, [values]>
    results = {}

    # to ensure each metrics file used the same data file
    data_file = None

    num_models = 0

    for model_dir in glob(input_pattern):
        # read in metrics file
        metrics_file = model_dir + "/" + test_file
        print(metrics_file)
        if not os.path.isfile(metrics_file):
            print("error, metrics file is missing")

        with open(metrics_file) as input_file:
            num_models += 1

            lines = input_file.readlines()
            predictions_file = lines[0].strip().split(" ")[-1]
            data_file_current = lines[1].strip().split(" ")[-1]
            label_file_current = lines[2].strip().split(" ")[-1]

            # to ensure each metrics file used the same data file
            if not data_file:
                data_file = data_file_current
            else:
                assert data_file == data_file_current
           
            assert lines[3].strip() == ''

            # read in the metrics
            for line in lines[4:]:
                metric, value = line.strip().split("\t")
                if metric not in results:
                    results[metric] = []
                results[metric].append(float(value))

    # write out averages and stds
    for metric, values in results.items():
        values = np.array(values)
        assert len(values) == num_models
        print("%s\t%.3f\t%.3f" % (metric, values.mean(), values.std()))


def print_metrics_all_runs(input_pattern, analysis_type):
    """ 
    Iterate over all the models and perform 
    the indicated analysis

    args: 
       input_pattern: input pattern to the model files
       analysis_type: {print_predictions_file,label_analysis} 
    """
    print("print metrics")
    for model_dir in glob(input_pattern):
        if analysis_type == "general_analysis":
            classification_analysis.print_analysis_predictions_file(
                            model_dir + "/test_output.txt",
                            contextual_abuse_dataset.DATASET_FULL_PATH,
                            contextual_abuse_dataset.DATASET_TEST_PATH
            )
        elif analysis_type == "label_analysis":
            classification_analysis.print_label_analysis(
                            model_dir + "/test_output.txt",
                            contextual_abuse_dataset.DATASET_FULL_PATH,
                            contextual_abuse_dataset.DATASET_TEST_PATH
            )


def analyze_runs(input_pattern):
    """ 
    Given a directory where each input directory is the output
    of one model, read in the evaluation results
    and print out the results (average micro and macro scores) 
    together with the used parameters.
    
    args:
        input_pattern: this should be a pattern that points to a folder
                       where each directory is the output of one finetuning run,
                       e.g ..."finetuning_bert/*"

    """

    # keep track of the scores
    f1_macros = []
    f1_micros = []


    for input_dir in glob(input_pattern):

        if not os.path.isdir(input_dir):
            continue

        # the name of the directory encodes different settings
        run = os.path.basename(input_dir)

        log_file = input_dir + "/log_history.json"
        if not os.path.isfile(log_file):
            print("Log file is missing: %s" % log_file)
            continue  

        # read in the file
        with open(log_file, 'r') as input_file:
            # results are at the end
            eval_results = json.load(input_file)[-1]

            # print results for the individual runs
            print("%s\t%.4f\t%.4f" % ("\t".join([str(s) for s in run.split("_")]), 
                eval_results['eval_fbeta_macro'], eval_results['eval_fbeta_micro']))

            f1_macros.append(float(eval_results['eval_fbeta_macro']))
            f1_micros.append(float(eval_results['eval_fbeta_micro']))

        
    print("Averages")
    print("F1 macro: %.3f (%.3f)" % (np.array(f1_macros).mean(), 
                                     np.array(f1_macros).std()))
    print("F1 micro: %.3f (%.3f)" % (np.array(f1_micros).mean(), 
                                     np.array(f1_micros).std()))



def error_analysis(input_pattern, num_instances=20):
    """ Do an error analysis by printing instances 
    that many of the models predicted incorrectly 
    
    args: 
       input_pattern: input pattern to the model files
       num_instances: number of instances to print per type
    """
    
    # get gold labels
    gold_label_map = classification_analysis.get_gold_label_map(
                        contextual_abuse_dataset.DATASET_TEST_PATH
    )
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()
   
    # read in the test set
    df = pd.read_csv(contextual_abuse_dataset.DATASET_TEST_PATH, delimiter="\t", 
                    quoting=csv.QUOTE_NONE)
    
     # Keep a count of errors per instance, so we can print the instances
    # that many models do wrong
    # <label, <correct/incorrect, <post_id, count>>>
    count_map = {}
    for label in label_map.keys():
        count_map[label] = {'correct': defaultdict(int),
                            'incorrect': defaultdict(int)}
 
   
    for model_dir in glob(input_pattern):
        
        predictions_file =  model_dir + "/test_output.txt"
        print("Predictions file: %s" % predictions_file)

        predictions_map = classification_util.read_predictions_file(predictions_file)
        
        # update stats
        for post_id, predictions in predictions_map.items():

            for label, label_id in label_map.items():

                if gold_label_map[post_id][label_id]:

                    if predictions[label_id]:
                        count_map[label]['correct'][post_id] += 1
                    
                    # it should have predicted this label, but was missed
                    else:
                        count_map[label]['incorrect'][post_id] += 1

                   
    # now print the instances
    for label, d in count_map.items():

        print("\n**Post that are %s but not predicted to be %s**" % (label, label))
        error_map = count_map[label]['incorrect']
        for post_id in sorted(error_map, key=error_map.get, reverse=True)[:num_instances]:
            text = df.loc[df['id'] == post_id]['text'].values[0]
            print("%s\t%s\t%s" % (post_id, error_map[post_id], text))

        print("\n**Post that are %s and predicted to be %s**" % (label, label))
        correct_map = count_map[label]['correct']
        for post_id in sorted(correct_map, key=correct_map.get, reverse=True)[:num_instances]:
            text = df.loc[df['id'] == post_id]['text'].values[0]
            print("%s\t%s\t%s" % (post_id, correct_map[post_id], text))

        
###############################
# Run a model on a test set
###############################
    
def write_predictions(input_pattern, model_type):
    """ 
    Apply each model (indicated by input_pattern)
    to the test data and write output
    """
    import torch
    from nlp import load_dataset
    from transformers import BertTokenizer, DistilBertTokenizer

    import bert_multilabel
    import distillbert_multilabel


    # read in test data
    test_dataset = load_dataset('contextual_abuse_dataset.py', split="test")

    for model_dir in glob(input_pattern):
    
        print("Read model from: %s" % model_dir)

        model = None
        if model_type == "DistilBERT":
            model = distillbert_multilabel.DistilBertForMultiLabelSequenceClassification.from_pretrained(
                            model_dir, num_labels=contextual_abuse_dataset.NUM_LABELS) 
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        elif model_type == "BERT":
            model = bert_multilabel.BertForMultiLabelSequenceClassification.from_pretrained(
                            model_dir, num_labels=contextual_abuse_dataset.NUM_LABELS) 
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        else:
            print("Error, unknown model")
            sys.exit(0)
        
        # set the model to evaluation mode
        model.eval()

        output_file_path = model_dir + "/test_output.txt"
        print("Output file: %s" % output_file_path)

        if os.path.isfile(output_file_path):
            print("Output file already exists")
            return

        with open(output_file_path, 'w') as output_file:
            with torch.no_grad():
                for row in test_dataset:
                    tokens = tokenizer(row['text'], padding=True, truncation=True, return_tensors="pt")
                    predictions = model(**tokens)[0]
                    s = torch.sigmoid(predictions)
                    pred_labels_multi = (s >= 0.5).type(torch.uint8)
                    output_file.write("%s\t%s\n" % (row['id'], pred_labels_multi[0].numpy()))
                


if __name__ == "__main__":
    
     
    my_parser = argparse.ArgumentParser(description='Analyze transformer results')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: analyze_runs, print_metrics, print_aggr_analysis, write_predictions, print_error_analysis')

    my_parser.add_argument('--input_pattern',
                        metavar='input_pattern',
                        type=str,
                        help='Input pattern',
                        required=False)

    my_parser.add_argument('--analysis_type',
                        metavar='analysis_type',
                        type=str,
                        help='Analysis type',
                        required=False)

    my_parser.add_argument('--metrics_file',
                        metavar='metrics_file',
                        type=str,
                        help='Name of the file with the evaluation metrics',
                        required=False)

    my_parser.add_argument('--model_type',
                        metavar='model_type',
                        type=str,
                        help='Model type',
                        required=False)

    args = my_parser.parse_args()
    args = vars(args)
    
    print("Analyze the results of the Transformer models")

    if args["action"] == "analyze_runs":
        print("Analyze the results of %s" % args["input_pattern"])
        analyze_runs(args["input_pattern"])

    elif args["action"] == "print_metrics":
        print("Print the metrics for %s" % args["input_pattern"])
        print_metrics_all_runs(args["input_pattern"], args["analysis_type"])

    elif args["action"] == "print_aggr_analysis":
        print("Print aggregate metrics for %s" % args["input_pattern"])
        print_aggr_analysis(args["input_pattern"], args["metrics_file"])

    elif args["action"] == "write_predictions":
        print("Write predictions for %s (a %s model)" % (args["input_pattern"], args["model_type"]))
        write_predictions(args["input_pattern"], args["model_type"])

    elif  args["action"] == "print_error_analysis":
        print("Print error analysis for %s" % args["input_pattern"])
        error_analysis(args["input_pattern"])

    
