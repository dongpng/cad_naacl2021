import argparse
import csv
import datetime
import math
import numpy as np
import pandas as pd 

from sklearn.metrics import classification_report, coverage_error, hamming_loss, precision_recall_fscore_support, confusion_matrix, jaccard_score

import contextual_abuse_dataset


###############################
# A few preprocessing methods to process the full dataset file into
# files for the classification task.
###############################

def ignore_entry(s):
    """Return true if we want to ignore this entry"""
    return (s == "nan" or len(s.strip()) == 0 or 
            s == "[removed]" or s == "[deleted]")


def get_raw_texts_and_labels(data, selected_set):
    """
    Given the data (map), the selected set (train/dev/test)
    and a label_map, return:
    
    texts_raw: list with texts in the selected set
    labels_raw: list with corresponding labels in the selected set 
    ids_raw: list with postIDs
    """

    texts_raw = []
    labels_raw = []
    ids_raw = [] 
    
    for id, d in data.items():
        if d['split'] != selected_set:
            continue
        
        # we only want the unique set of labels
        labels_raw.append(set(d['labels']))
        texts_raw.append(d['text'])
        ids_raw.append(id)

    return texts_raw, labels_raw, ids_raw


def write_train_dev_test_splits(input_file):
    """
    Reads in the full dataset file and splits it into
    a train/dev/test file (excluding the rows with 'exclude')
    and aggregating the labels per post.
    Writes out three files, one for each split.
    Slurs are removed, because they are very infrequent

    args:
        input_file: path to the input file

    returns:
        None, but writes three files in the current directory
    """

    # read in the data from the csv file
    df = pd.read_csv(input_file, delimiter="\t", quoting=csv.QUOTE_NONE, 
                     keep_default_na=False)

    # Read in a map with the data
    # each entry stores: text, labels,
    #                    threadID, set
    data = {} # <id, {data about the entry}>
    
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()

    # Iterate over the rows
    for index, row in df.iterrows():

        if ('train' in row['split'] or 'dev' in row['split']
            or 'test' in row['split']):

            # If it's the first time we're seeing this
            # entry, add it to the map
            if row['info_id'] not in data:

                # store basic data
                data[row['info_id']] = {
                    'text': str(row['meta_text']).strip(),
                    'labels': [],
                    'thread_id': row['info_thread.id'],
                    'split': row['split']
                }

            # add the label
            data[row['info_id']]['labels'].append(row['annotation_Primary'])

    # Now, write the data
    now = datetime.datetime.now()

    splits = ["train", "dev", "test"]

    for split in splits:

        texts_raw, labels, ids = get_raw_texts_and_labels(data, split)
        
        # convert the labels
        # e.g. Slurs are not used in the classification experiments.
        labels_converted = []

        for i, id in enumerate(ids):

            # we exclude Slurs, as they are infrequent.
            if 'Slur' in labels[i]:
                # if the only label is Slur, relabel the post as Neutral
                if len(labels[i]) == 1:
                    labels[i] = set(['Neutral'])
                # else just remove Slur
                else:
                    labels[i].remove('Slur')
                
            tmp = list(labels[i])
            tmp.sort()
            labels_converted.append(",".join(tmp))
           
        # now write to file
        df_to_write = pd.DataFrame(list(zip(ids, texts_raw, labels_converted)), 
                                columns =['id', 'text', 'labels']) 

        csv_output_file = ("cad_%s_%s_%s_%s.tsv" % 
                                (now.day, now.month, now.year, split))

        df_to_write.to_csv(csv_output_file, index=False, sep="\t", 
                            quoting=csv.QUOTE_NONE, quotechar="",  escapechar="")
                


###############################
# Input / output functions
###############################


def save_output(ids, preds, output_file="classification_output.txt"):
    """ Save the output to file

    args:
       ids: list with IDs of the instances
       preds: list with the predictions
       output_file: path to output file
    """
    assert len(ids) == len(preds)

    with open(output_file, 'w') as output_file:
        for i, id in enumerate(ids):
            output_file.write("%s\t%s\n" % (ids[i], preds[i]))

###############################
# Evaluation
###############################

def multilabel_accuracy(gold_labels, predictions):
    """ 
    Calculates exact match accuracy for multi label
    classification problems.

    Each instance is a multi hot encoding of the labels
    
    args:
       gold_labels: list with gold labels
       predictions: list with predictions

    returns:
       the exact match accuracy (fraction)
    """
    correct = 0
    for i, labels in enumerate(gold_labels):
        if np.array_equal(labels, predictions[i]):
            correct += 1

    return correct/float(len(gold_labels))


def get_multilabel_results(y_true, y_pred, inv_label_map=None):
    """ Return multilabel metrics in a dictionary
    and a result string to print 
   
    args:
       y_true: the gold labels (list)
       y_pred: the predictions (list)
       inv_label_map: {index, label}

    returns:
       a dictionary with metrics and values
       a string with evaluation output
    """

    # create a dictionary with results
    result = {
        "coverage_error": coverage_error(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "multilabel_accuracy": multilabel_accuracy(y_true, y_pred),
        'jaccard_score': jaccard_score(y_true, y_pred, average='samples')
    }

    # macro results
    precision_macro, recall_macro, fbeta_macro, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                                                    average="macro")
    result['precision_macro'] = precision_macro
    result['recall_macro'] = recall_macro
    result['fbeta_macro'] = fbeta_macro

    precision_micro, recall_micro, fbeta_micro, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                                                    average="micro")
    result['precision_micro'] = precision_micro
    result['recall_micro'] = recall_micro
    result['fbeta_micro'] = fbeta_micro

    # also print the metrics
    result_str = ""
    for metric, score in result.items():
        result_str += "%s\t%.3f\n" % (metric, score)

    # now calculate precision and recall for each class (no aggregation)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred)

    # print header
    if inv_label_map:
        classes_header = "\t"
        for i in range(len(inv_label_map)):
            classes_header += inv_label_map[i] + "\t"
            result["precision_" + inv_label_map[i]] = precision[i]
            result["recall_" + inv_label_map[i]] = recall[i]
            result["fbeta_" + inv_label_map[i]] = fbeta[i]
            result["support_" + inv_label_map[i]] = int(support[i]) #otherwise, it's int64?
            
        result_str += classes_header + "\n"

    result_str += "precision\t%s\n" % " ".join(["{:.3f}".format(s) for s in precision])
    result_str += "recall\t%s\n" % " ".join(["{:.3f}".format(s) for s in recall])
    result_str += "fbeta\t%s\n" % " ".join(["{:.3f}".format(s) for s in fbeta])
    result_str += "support\t%s\n" % " ".join([str(s) for s in support])

    return result, result_str


def read_predictions_file(predictions_file):
    """ 
    Read in a predictions file (id \t predictions)

    args:
        predictions_file: path to file with predictions
    
    returns:
        map with info_id, [predictions] (multi hot encoded)
    """
    predictions_map = {}
    with open(predictions_file, 'r') as input_file:
        for line in input_file.readlines():
            post_id, predictions = line.strip().split("\t")
            predictions = [int(i) for i in predictions[1:-1].split(' ')]
            predictions_map[post_id] = predictions

    return predictions_map

###############################
# Helper functions for transformers
###############################


def calculate_class_weights(dataset_labels):
    """
    Set weights as instructions in BCEwithLogitsLoss doc

    args:
        dataset_labels: list of lists with labels
                        [[0], [1,2,3], [2,3], ...]

    returns:
        a dictionary with for each label the weight
    """
    # First get the counts of the labels
    # label, [pos, neg count]
    # first get the positive count
    label_count = {}
    for l in dataset_labels:
        for label in l:
            if label not in label_count:
                label_count[label] = [1,0]
            else: 
                label_count[label][0] += 1   # increase pos count
    
    # now update negative counts and 
    # calculate the class weights for the loss function
    weights = [0] * len(label_count)
    for label, count in label_count.items():
        # calculate the negative counts
        count[1] = len(dataset_labels) - count[0]
        # set weights as instructions in BCEwithLogitsLoss doc
        weights[label] = count[1]/float(count[0])

    return weights


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Classification util')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: write_splits')


    my_parser.add_argument('--input_file',
                        metavar='input file',
                        type=str,
                        help='path to input file (tsv)',
                        required=False)

    args = my_parser.parse_args()
    args = vars(args)
    
    print("Classification util")
    if args["action"] == "write_splits":
        print("Write splits")
        print("Input file: %s" % args["input_file"])
        write_train_dev_test_splits(args["input_file"])

