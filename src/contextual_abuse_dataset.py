# coding=utf-8
# Copyright 2020 The HuggingFace NLP Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset loader for the contextual abuse dataset."""

from __future__ import absolute_import, division, print_function

import csv
import re
import nlp

import pandas as pd
import datasets

DATASET_FULL_PATH = "../data/cad_v1.tsv"
DATASET_TRAIN_PATH = "../data/cad_v1_train.tsv"
DATASET_DEV_PATH = "../data/cad_v1_dev.tsv"
DATASET_TEST_PATH = "../data/cad_v1_test.tsv"

_DESCRIPTION = """The Contextual Abuse Dataset, NAACL 2021"""

# The primary annotation categories
CATEGORY_NAMES = ['Neutral',
                  'IdentityDirectedAbuse',
                  'AffiliationDirectedAbuse',
                  'PersonDirectedAbuse',
                  'CounterSpeech']
                    
NUM_LABELS = len(CATEGORY_NAMES)

ONLY_DEV_SUBREDDITS = ['ImGoingToHellForThis', 'Negareddit', 'WatchRedditDie']
ONLY_TEST_SUBREDDITS = ['conspiracy', 'CCJ2', 'smuggies']

def get_label_map():
    """ Return a dictionary with
    <label, ID>

    and an inverse mapping
    """
    label_map = {}
    for label in CATEGORY_NAMES:
        label_map[label] = len(label_map)

    inv_label_map =  {v: k for k, v in label_map.items()}
    return label_map, inv_label_map
    

def replace_subreddits_usernames(text):
    """ replaces usernames and subreddit names with a placeholder """
    text_orig = text
    text = re.sub(r'\/r\/\w+', '[subreddit]', text)
    text = re.sub(r'\/u\/\w+', '[user]', text)
    return text
    

def replace_urls(text):
    """ replaces urls. Standalone URLs are replaced with [LINK].
        Descriptions are kept for cases such as [description](link)

        return text
    """
    text = re.sub(r"\[([^\[\]]+)\]\((https:\/\/(.*?))\)", r"\1", text)

    # tricky, if there are multiple urls, we want to make sure 
    # these are not captures all into one group.
    # therefore negation to get the smallest [] group
    text = re.sub(r"\[([^\[\]]+)\]\((\/message\/compose(.*?))\)", r"\1", text)
    text = re.sub(r"\[([^\[\]]+)\]\((\/r\/(.*?))\)", r"\1", text)
    text = re.sub(r'http(s?):\/\/\S+', '[LINK]', text)
    text = re.sub(r'www\.\S+', '[LINK]', text)
    return text


def get_df(data_file):
    """
    Read in the data frame
    
    args:
        data_file: path to data
    """
    return pd.read_csv(data_file, delimiter="\t", quoting=csv.QUOTE_NONE,
                        keep_default_na=False)


class ContextualAbuseRedditDataset(nlp.GeneratorBasedBuilder):
    """
    The contextual abuse dataset with Reddit posts.
    Each post can have multiple labels
    """

    VERSION = nlp.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "labels_info": datasets.features.Sequence(
                        {
                            "label": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # common input tuple for supervised learning
            supervised_keys=("text", "labels_info")
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        print("Dataset train path: %s" % DATASET_TRAIN_PATH)
        print("Dataset dev path: %s" % DATASET_DEV_PATH)
        print("Dataset test path: %s" % DATASET_TEST_PATH)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": DATASET_TRAIN_PATH},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": DATASET_DEV_PATH},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": DATASET_TEST_PATH},
            ),
        ]


    def _generate_examples(self, filepath):
        """ Yields examples. """

        df = get_df(filepath)

        for index, row in df.iterrows():

            # remove linebreaks
            text = row['text'].replace('[linebreak]', "\n").replace("\n ", "\n").strip()

            # get labels into a list
            labels = []
            for l in row['labels'].split(','):
                labels.append({
                     "label": CATEGORY_NAMES.index(l)
                })

            # generate example
            yield row['id'], {
                    "id": row['id'],
                    # ordering matters, first replace urls, then usernames
                    "text": replace_subreddits_usernames(
                                replace_urls(text)),
                    "labels_info": labels
        }

