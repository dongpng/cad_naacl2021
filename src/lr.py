import argparse
import logging
import warnings

from urllib.parse import urlparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from nlp import load_dataset

import numpy as np
import mlflow
import mlflow.sklearn

import classification_util
import contextual_abuse_dataset

# set logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
    

def run_lr(C=1, run="dev", max_df=0.9, min_df=10):
    """ Run logistic regression.
    The results are stored in a 'mlruns' folder
    in the current directory
    
    C: used in logistic regression (regularization)
    run: dev or test
    max_df: maximum document freq for inclusion in vocab
    min_df: minimum document freq for inclusion in vocab
     """
    # set a random seed
    np.random.seed(892)

    print("Load dataset")
    # training data is shuffled
    train_dataset = load_dataset('contextual_abuse_dataset.py', split="train")
    train_dataset = train_dataset.shuffle(seed=3791)
    test_dataset =  (load_dataset('contextual_abuse_dataset.py', split="validation") if run == "dev" else
                    load_dataset('contextual_abuse_dataset.py', split="test"))
    

    label_map, inv_label_map = contextual_abuse_dataset.get_label_map()

    # get the texts and IDs
    # labels are a list of lists with labels, e.g. [[0], [0], [1, 2], ...]
    train_texts = train_dataset["text"]
    train_ids = train_dataset["id"]
    train_labels = [s['label'] for s in train_dataset["labels_info"]]

    test_texts = test_dataset["text"]
    test_ids = test_dataset["id"]
    test_labels =[s['label'] for s in test_dataset["labels_info"]]
    
    print("Finish loading dataset")

    # Print the first 10 items of the training data
    print(train_texts[:10])
    print(train_labels[:10])
    print(train_ids[:10])


    # fit the vectorizer
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # for one vs rest
    # transform labels into list with multi hot encoding
    mlb = MultiLabelBinarizer()
    train_labels_multi = mlb.fit_transform(train_labels)
    test_labels_multi = mlb.transform(test_labels)
    
    # reverse the vocab
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}

    class_weight = "balanced"  # inv prop to class frequencies
    max_iter = 500
    random_state = 31
    penalty = "l2"

    with mlflow.start_run():

        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("min_df", min_df)        

        # train the classifier
        clf = OneVsRestClassifier(LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=random_state,
                penalty=penalty,
                C=C)).fit(X_train, train_labels_multi)

        # apply predictions
        y_pred = clf.predict(X_test)

        # write output to file 
        classification_util.save_output(test_ids, y_pred, "model_outputs.txt")
        mlflow.log_artifact("model_outputs.txt")
        
       
        # Calculate and write evaluation metrics
        evaluation_score, evaluation_str = classification_util.get_multilabel_results(
                                        test_labels_multi, y_pred, inv_label_map)

        print(evaluation_str)
        for metric, score in evaluation_score.items():
            mlflow.log_metric(metric, score)

        # Print the features
        with open('features.txt', 'w') as features_file:

            for idx, category in enumerate(contextual_abuse_dataset.CATEGORY_NAMES):
                features_file.write("\n***** %s *******\n\n" % category)
                scores = {}
                for idx2, weight in enumerate(clf.coef_[idx]):
                    scores[inv_vocab[idx2]] = weight

                for w in sorted(scores, key=scores.get, reverse=True)[:25]:
                    features_file.write("%s\t%s\n" % (w, scores[w]))

        mlflow.log_artifact("features.txt")

        # Save model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            mlflow.sklearn.log_model(clf, "model", registered_model_name="LR_CAD")
        else:
            mlflow.sklearn.log_model(clf, "model")


        print("\n******\nMajority classifier")
        maj_label_pred = [1] + [0] * (contextual_abuse_dataset.NUM_LABELS - 1)
        y_pred_majority = np.array([maj_label_pred]*len(y_pred))

        evaluation_score_majority, evaluation_str = classification_util.get_multilabel_results(
                                test_labels_multi, y_pred_majority, inv_label_map)

        
        print(evaluation_str)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Logistic regression')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: tune, test')


    args = my_parser.parse_args()
    args = vars(args)
    
    print("Logistic regression")
    if args["action"] == "tune":
        print("Tune")

        # smaller values, stronger regularization
        for C in [0.05, 0.1, 0.25, 0.5, 1]:
            for max_df in [0.9, 0.95, 0.99]:
                for min_df in [5, 10, 20]:
                    print("\n#################")
                    print("%s; %s; %s" % (C, max_df, min_df))
                    run_lr(C, "dev", max_df=max_df, min_df=min_df)
                   
        # best c: 0.25/0.1/0.5  (F1 macro)
        # best c: 1/0.5/0.25 (F1 micro)
        # max df doesn't seem to matter (so 0.95?)
        # best min_df = 5

    elif args["action"] == "test":
        print("Test")
        run_lr(0.25, "test", max_df=0.95, min_df=5)
