import argparse
import collections
import csv
import os

from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

import classification_util
import contextual_abuse_dataset

################## Utility functions ##################

def get_gold_label_map(input_file):
    """"
    args:
        input_file: tab separated file with id, text, labels

    Returns:
       the gold label map <info_id, labels> with labels 
       encoded as a multi hot vector

    """
    df = pd.read_csv(input_file, delimiter="\t")

    label_map,_ = contextual_abuse_dataset.get_label_map()
    
    # <id, labels>
    gold_label_map = {}

    for index, row in df.iterrows():

        gold_label_map[row['id']] = [0] * len(label_map)

        for l in row['labels'].strip().split(','):
            gold_label_map[row['id']][label_map[l]] = 1

    return gold_label_map


def get_gold_preds_per_setting(predictions_file, data_file, label_file):
    """ Given a predictions file and the raw data file, 
    return the gold labels and predictions for each setting
    (all entries, seen and unseen subreddits)
    
    args:
        predictions_file: tab separated file with on each line: id \t predictions
        data_file: tab separated file with information about the dataset (incl seen/unseen)
        label_file: tab separated file with id, text, labels

    Returns the gold labels and the predictions for each breakdown of the results
    """

    # first read in the label map
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()
    
    # read in the predictions from the predictions file
    predictions_map = classification_util.read_predictions_file(predictions_file)

    # collect information from the data file
    df = contextual_abuse_dataset.get_df(data_file)

    # setup lists to store the result for each settings
    # predictions and gold labels for:
    # full set
    preds_full = []
    gold_full = []

    # unseen subreddits
    preds_unseen_subreddits = []
    gold_unseen_subreddits = []

    # seen subreddits
    preds_seen_subreddits = []
    gold_seen_subreddits = []

    gold_label_map = get_gold_label_map(label_file)

    # Now read in the predictions
    for post_id, preds in predictions_map.items():

        gold_post = gold_label_map[post_id]

        preds_full.append(preds)
        gold_full.append(gold_post)

        assert len(preds) == len(gold_post)

        #if subreddit is seen:
        if df.loc[df['info_id'] == post_id]['subreddit_seen'].values[0]:
            preds_seen_subreddits.append(preds)
            gold_seen_subreddits.append(gold_post)
        else:
            preds_unseen_subreddits.append(preds)
            gold_unseen_subreddits.append(gold_post)
     

    results_map = {
        'full': (gold_full, preds_full),
        'unseen': (gold_unseen_subreddits,
                  preds_unseen_subreddits),
        'seen': (gold_seen_subreddits, 
                preds_seen_subreddits),
    }

    return results_map


################## Analysis methods ##################

def compute_label_stats(labels):
    """
    Compute statistics about the labels
    (such as label cardinality)

    args:
        labels: a list with label strings (e.g., 
                ["Neutral", "IdentityDirectedAbuse,AffiliationDirectedAbuse"])
    """
    # first get the labels
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()

    # store the counts for each pair of label
    pairwise_label_counts = np.zeros((len(contextual_abuse_dataset.CATEGORY_NAMES), 
                                      len(contextual_abuse_dataset.CATEGORY_NAMES)))

    # keep a count of each label combination
    label_set_count = collections.Counter()

    # number of instances with > 1 label
    num_entries_multilabel = 0

    # number of Neutral instances
    num_neutral = 0

    # label_cardinality: avg number of labels per instance 
    # (Multi-Label Classification: An Overview, Tsoumakas and Katakis)
    label_cardinality = 0

    # now iterate over the training set
    for i, tmp in enumerate(labels):
        
        # Make sure each label combination is represented
        # with a unique string (based on sort)
        label_set = tmp.split(',')
        assert len(label_set) == len(set(label_set))
        label_set.sort()
        
        # Update stats
        label_set_count.update({' '.join(label_set): 1})
        
        if len(label_set) > 1:
            num_entries_multilabel += 1
            
        label_cardinality += len(label_set)
            
        if 'Neutral' in label_set:
            num_neutral += 1
            
        # update pairwise label counts
        for idx1 in range(len(label_set)):
            for idx2 in range(idx1 + 1, len(label_set)):
                label1 = label_map[label_set[idx1]]
                label2 = label_map[label_set[idx2]]
                
                pairwise_label_counts[label1][label2] += 1
                pairwise_label_counts[label2][label1] += 1
                    
                
    print("Number of instances: %s" % len(labels))
    print("Number of non-Neutral instances: %s" % (len(labels) - num_neutral))
    print("Number of instances with >1 label: %s (%.2f%%)" % (num_entries_multilabel,
                                                            100 * float(num_entries_multilabel)/len(labels)))
    print("Percentage of non-Neutral instances with >1 labels: %.1f\n" % (100 * float(num_entries_multilabel)/(len(labels) - num_neutral)))                                                     

    label_cardinality /= len(labels)
    print("Label cardinality: %.3f" % label_cardinality)

    # print the most common label combinations
    for tmp, count in label_set_count.most_common(30):
        print("%s\t%s" % (count, tmp))
        
    print("\nPairwise counts")
    print("\t" + "\t".join(contextual_abuse_dataset.CATEGORY_NAMES))
    for i in range(len(pairwise_label_counts)):
        print(contextual_abuse_dataset.CATEGORY_NAMES[i] + "\t" +  
            "\t".join([str(s) for s in pairwise_label_counts[i]]))

    return label_cardinality, label_set_count, pairwise_label_counts


def label_analysis_helper(predictions_file, data_file, label_file):
    """ Given a predictions file and the full data file,
        do an analysis of the predicted labels
        and print out to file (same dir as predictions file).


    args:
        predictions_file: tab separated file with the predictions 
                          column1: ID; column2: multi hot encoding of prediction
        data_file: path to dataset
        label_file: tab separated file with id, text, labels

    return a map with metrics.
    """

    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()

    # Get gold labels and predictions for each setting
    results_map = get_gold_preds_per_setting(predictions_file, data_file, label_file)

    # in this analysis, we only consider the full results
    gold_labels_full, predictions_full = results_map["full"]

    # a map to store the counts of each combination of
    # [num_labels_predicted] and [num_labels_gold]
    num_predictions_multilabel = defaultdict(int)

    num_too_many_labels = 0  # how many times too many labels were predicted
    num_too_few_labels = 0  # how many times too few labels were predicted
    num_neutral_violated = 0 # how many have predicted neutral and another category?

    
    for i, gold_labels in enumerate(gold_labels_full):

        # how many labels
        num_labels_predicted =  np.array(predictions_full[i]).sum()
        num_labels_gold = np.array(gold_labels).sum()

        # update stats
        if num_labels_predicted > num_labels_gold:
            num_too_many_labels += 1
        elif num_labels_gold > num_labels_predicted:
            num_too_few_labels += 1
        
        # update dict with counts per pair
        num_labels_pairs = str(num_labels_predicted) + "_" + str(num_labels_gold)
        num_predictions_multilabel[num_labels_pairs] += 1

        if num_labels_predicted > 1 and predictions_full[i][label_map["Neutral"]] == 1:
            num_neutral_violated += 1

    
    return {"num_total_labels": len(gold_labels_full),
            "num_too_few_labels": num_too_few_labels,
            "perc_too_few_labels": (float(num_too_few_labels)/len(predictions_full)) * 100,
            "num_too_many_labels": num_too_many_labels,
            "perc_too_many_labels": (float(num_too_many_labels)/len(predictions_full)) * 100,
            "num_neutral_violated":  num_neutral_violated,
            "perc_neutral_violated": (float(num_neutral_violated)/len(predictions_full)) * 100,
    }


def print_label_analysis(predictions_file, data_file, label_file,
                         output_file_suffix="_label_analysis.txt"):
    """ Given a predictions file and the full data file,
        do an analysis of the predicted labels
        and print out to file (same dir as predictions file).


    args:
        predictions_file: tab separated file with the predictions 
                          column1: ID; column2: multi hot encoding of prediction
        data_file: path to dataset
        label_file: tab separated file with id, text, labels
        output_file_suffix: suffix of output file
    """

    output_file = os.path.splitext(predictions_file)[0] + output_file_suffix
    
   
    print("Predictions file: %s" % predictions_file)
    print("Data file: %s" % data_file)
    print("Output file: %s\n" % output_file)
    
    if os.path.isfile(output_file):
        print("Outputfile already exists!")
        return 

    with open(output_file, 'w') as output_file:
        output_file.write("Predictions file: %s\n" % predictions_file)
        output_file.write("Data file: %s\n" % data_file)
        output_file.write("Label file: %s\n\n" % label_file)
       
        results = label_analysis_helper(predictions_file, data_file, label_file)

        for metric, val in results.items():
            output_file.write("%s\t%.3f\n" % (metric, val))

                
def get_sec_categories_results(predictions_file, data_file):
    """"Computes and prints the recall for 
    combinations of primary and secondary categories
    
    args:
        predictions_file: tab separated file with the predictions 
                          column1: ID; column2: multi hot encoding of prediction
        data_file: path to dataset
    """

    # first get the label maps
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()
    
     # read in the predictions from the predictions file
    predictions_map = classification_util.read_predictions_file(predictions_file)

    # collect information from the data file
    df = contextual_abuse_dataset.get_df(data_file)
    
    # store all second categories
    sec_cats = set()

    # get secondary categories for each post
    # <info_id, secondary categories>
    sec_categories_map = defaultdict(set)

    for index, row in df.iterrows():

        # we ignore Neutral and Slur
        if (row['annotation_Primary'] == "Neutral" or
                row['annotation_Primary'] == "Slur"):
                continue
            
        sec_cats.add(row['annotation_Secondary'])
        sec_categories_map[row['info_id']].add(row['annotation_Secondary'])

    # keep counts for each secondary label
    counts_map = {}
    for sec_cat in sec_cats:
        counts_map[sec_cat] = {
            'num_gold':0, # the number of items with this sec category
            'num_pred': 0, # out of the num_gold items, for how many was the 
                           # primary category correctly predicted?
        }

    # Update stats for each post
    for post_id, predictions in predictions_map.items():    

        if post_id in sec_categories_map:
            for sec_cat in sec_categories_map[post_id]:
                counts_map[sec_cat]['num_gold'] += 1
                prim_cat = sec_cat.split(" ")[0]
                if predictions[label_map[prim_cat]]:
                    counts_map[sec_cat]['num_pred'] += 1

        
    # now print the data
    results = {}

    for label in sorted(counts_map.keys()):
        data = counts_map[label]
        print("%s\t%.3f\t%s\t%s" % (label, 
                               float(data['num_pred'])/data['num_gold'], 
                               data['num_pred'],
                               data['num_gold']
                               ))
        results[label + "_recall"] =  float(data['num_pred'])/data['num_gold']
        results[label + "_num_pred"] =  data['num_pred']
        results[label + "_num_gold"] =  data['num_gold']
    return results


def get_context_results(predictions_file, data_file):
    """"Returns the recall for 
    all instances, when labels are based on current context, and 
    when labels are based on previous context """

    # first get the label maps
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()
    
     # read in the predictions from the predictions file
    predictions_map = classification_util.read_predictions_file(predictions_file)

    # collect information from the data file
    df = contextual_abuse_dataset.get_df(data_file)
    
    # For each label (except neutral) keep track of the total number of instances
    # that were annotated with a label, as well as the ones that were predicted
    counts_map = {}

    for label_id, label in inv_label_map.items():
        if label == 'Neutral':
            continue

        counts_map[label] = {
            'full_num_gold': 0,
            'full_num_pred': 0,
            'current_num_gold': 0, # how many instances were there annotated with
                                   # this category and based on current context?
            'current_num_pred': 0, # and out of these, how many were identified?
            'prev_num_pred': 0,
            'prev_num_gold':0,
        }

    
    # for each post, keep track of primary categories and whether it's based on current content
    
    # if an instance has different entries for one primary category,
    # e.g. because of multiple secondary categories,
    # then if one of them is labeled current content,
    # assign it current content
    # <info_id, <prim_cat, context>>
    prim_context_map = defaultdict(dict)
    for index, row in df.iterrows():

        prim_cat = row['annotation_Primary']

        if prim_cat == "Neutral" or prim_cat == "Slur":
            continue

         # default is false (=prev content)
        if prim_cat not in prim_context_map[row['info_id']]:
            prim_context_map[row['info_id']][prim_cat] = False
        
        if row['annotation_Context'] == 'CurrentContent': 
            prim_context_map[row['info_id']][prim_cat] = True


    # Now calculate the statistics
    # For each post
    for post_id, predictions in predictions_map.items():

        if post_id not in prim_context_map:
            continue

        for prim_cat, current_context in prim_context_map[post_id].items():

            gold_label_idx = label_map[prim_cat]
            correct_pred = predictions[gold_label_idx]

            # stats for all instances
            counts_map[prim_cat]['full_num_gold'] += 1
            if correct_pred:
                counts_map[prim_cat]['full_num_pred'] += 1

            # current context
            if current_context:
                counts_map[prim_cat]['current_num_gold'] += 1
                if correct_pred:
                    counts_map[prim_cat]['current_num_pred'] += 1
            # prev context
            else:
                counts_map[prim_cat]['prev_num_gold'] += 1
                if correct_pred:
                    counts_map[prim_cat]['prev_num_pred'] += 1


    # now store the data
    results = {}

    for label, data in counts_map.items():
        # if we're just predicting a smaller set, there may be zero instances
        # for some categories
        if data['full_num_gold'] == 0:
            continue
        
        full_recall = float(data['full_num_pred'])/data['full_num_gold']
        results['fullc_recall_' + label] = full_recall
        results['fullc_support_' + label] =  data['full_num_gold']    

        if data['current_num_gold'] > 0:
            current_recall = float(data['current_num_pred'])/data['current_num_gold']
            results['current_content_recall_' + label] = current_recall
            results['current_content_support_' + label] =  data['current_num_gold']    
        
        if data['prev_num_gold'] > 0:
            prev_recall = float(data['prev_num_pred'])/data['prev_num_gold']
            results['prev_content_recall_' + label] = prev_recall
            results['prev_content_support_' + label] =  data['prev_num_gold']
    

    return results


def print_analysis_predictions_file(predictions_file, data_file, label_file):
    """ Given a predictions file and the full data file,
        evaluate the predictions and write the different metrics to files
        in the same directory as the predictions file.

    args:
        predictions_file: tab separated file with the predictions 
                          column1: ID; column2: multi hot encoding of prediction
        data_file: path to dataset
        label_file: tab separated file with id, text, labels
                    (should correspond to the set the predictions were made on)
    """

    print("\nPredictions file: %s" % predictions_file)
    print("Data file: %s" % data_file)
    print("Label file: %s" % label_file)

    output_file = os.path.splitext(predictions_file)[0] + "_analysis.txt"
    output_file_metrics = os.path.splitext(predictions_file)[0] + "_metrics.txt"
    
    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()

    print("Output file analysis: %s" % output_file)
    print("Output file metrics: %s" % output_file_metrics)

    if os.path.isfile(output_file):
        print("Outputfile already exists!")
        return 

    if os.path.isfile(output_file_metrics):
        print("Outputfile metrics already exists!")
        return 

    with open(output_file, 'w') as output_file:
        with open(output_file_metrics, 'w') as output_file_metrics:
            # first write some file paths info
            output_file.write("Predictions file: %s\n" % predictions_file)
            output_file.write("Data file: %s\n" % data_file)
            output_file.write("Label file: %s\n\n" % label_file)

            output_file_metrics.write("Predictions file: %s\n" % predictions_file)
            output_file_metrics.write("Data file: %s\n" % data_file)
            output_file_metrics.write("Label file: %s\n\n" % label_file)


            # Get gold labels and predictions for each setting
            results_map = get_gold_preds_per_setting(predictions_file, data_file, label_file)

            # For each setting, evaluate and write to file
            for result_type, (golds, preds) in results_map.items():
                output_file.write("\n**%s***\n" % result_type)
                eval_results, eval_results_str = classification_util.get_multilabel_results(
                                                                golds, 
                                                                preds,
                                                                inv_label_map)

                output_file.write(eval_results_str + "\n")
               
                # also write metrics
                for metric, val in eval_results.items():
                    output_file_metrics.write("%s\t%.3f\n" % (result_type + "_" + metric, val))
                
                
            # print results looking at context
            for metric, val in get_context_results(predictions_file, data_file).items():
                output_file_metrics.write("%s\t%.3f\n" % (metric, val))
            
           
            # print results looking at secondary categories
            for metric, val in get_sec_categories_results(predictions_file, data_file).items():
                output_file_metrics.write("%s\t%.3f\n" % (metric, val))


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='Analyzes the classification results')

    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: write_splits')

    my_parser.add_argument('--predictions_file',
                        metavar='predictions file',
                        type=str,
                        help='path to predictions file (tsv)',
                        required=False)
 
    args = my_parser.parse_args()
    args = vars(args)
    
    print("Analyze the classification output: %s" % args["predictions_file"])
 
    if args["action"] == "general_analysis":
        print_analysis_predictions_file(args["predictions_file"],
                                    contextual_abuse_dataset.DATASET_FULL_PATH,
                                    contextual_abuse_dataset.DATASET_TEST_PATH)

    elif args["action"] == "label_analysis":
        print_label_analysis(args["predictions_file"],
                                    contextual_abuse_dataset.DATASET_FULL_PATH,
                                    contextual_abuse_dataset.DATASET_TEST_PATH)

   