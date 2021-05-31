import argparse
import csv
import time
import numpy as np
import pandas as pd

import contextual_abuse_dataset

def check_body_post(df):
    """ Does every post have a body and title? We have post-title, and comments below"""
    opening_posts = set()
    for index, row in df.iterrows():
        if 'post' in row['info_id']:
            opening_posts.add(row['info_id'].split("-")[0])
    
    # should match the number of conversations = OK
    num_opening_posts = len(opening_posts)
    
    opening_titles = set()
    for index, row in df.iterrows():
        if 'title' in row['info_id']:
            opening_titles.add(row['info_id'].split("-")[0])
            
    num_opening_titles = len(opening_titles)
    num_intersection = len(opening_posts.intersection(opening_titles))

    #print("check_body_post: number of opening posts %s" % num_opening_posts)
    #print("check_body_post: number of opening titles %s" % num_opening_titles)
    
    if (num_opening_posts == num_opening_titles
        and num_intersection == num_opening_posts
        and len(opening_posts.symmetric_difference(opening_titles)) == 0):
        print("check_body_post: OK")
    else:
        print("check_body_post: Error")


def check_multiple_entries(df):
    """Is the text the same across multiple entries with same ID?"""
    entries_text = {} #<entry ID, [text]>

    # collect data
    for index, row in df.iterrows():
        if row['info_id'] not in entries_text:
            entries_text[row['info_id']] = []

        entries_text[row['info_id']].append(row['meta_text'])

    # texts should be identical
    ok = True
    for id, texts in entries_text.items():
        if len(texts) > 1:
            if len(set(texts)) != 1:
                print("\n***")
                print(id)
                print(texts)
                ok = False

    if ok:
        print("check_multiple_entries: OK")
    else:
        print("check_multiple_entries: Error")


def check_labels(df):
    """Are there posts that have a secondary label while it's Neutral?"""
    
    ok = True
    for index, row in df.iterrows():
        if row['annotation_Primary'] == 'Neutral':
            if (str(row['annotation_Secondary']) != "" and
                str(row['annotation_Secondary']) != "NA" and
                str(row['annotation_Secondary']) != '"NA"'):
                print("check_labels: Neutral error: %s" % row['id'])
                ok = False

        elif row['annotation_Primary'] != 'Slur':
            # we need to make sure the primary cat. in the
            # sec. cat label align with the prim cat. annotation
            prim = row['annotation_Secondary'].split(" ")[0]
            if prim != row['annotation_Primary']:
                print("check_labels: Primary categories don't match: %s" % row['id'])
                ok = False

        if (row['annotation_Primary'] not in 
                contextual_abuse_dataset.CATEGORY_NAMES + ['Slur']):
            print("check_labels: Unknown primary category")
            ok = False
    
    if ok:            
        print("check_labels: OK")
    else:
        print("check_labels: Error")


def check_empty_text(df):
    """Which posts have no text?"""
    ok = True
    for index, row in df.iterrows():
        text = str(row['meta_text'])
        # text == "nan" or 
        if (len(text.strip()) == 0 and row['annotation_Primary'] != "Neutral" and 
            'image' not in row['annotation_highlighted']):
            print("check_empty_text: %s %s" % (index, row))
            ok = False

    if ok:            
        print("check_empty_text: OK")
    else:
        print("check_empty_text: Error")


def check_info_order_as_key(df):
    """ Is each info_order linked to a unique id? """

    info_order_to_id = {}
    for index, row in df.iterrows():
        if row['info_order'] not in info_order_to_id:
            info_order_to_id['info_order'] = row['info_id']
        else:
            assert info_order_to_id['info_order'] == row['info_id']

    print("check_info_order_as_key: OK")


def check_thread_structure(df):
    """ Check the thread structure"""
    ok = True

    # For each thread
    for thread in df['info_thread.id'].unique():
        result = {}

        # Get all entries.
        # For each entry (info_order as key), store the current id (id) and its parent
        for index, row in df.loc[df['info_thread.id'] == thread].iterrows():
            result[row['info_order']] = {
                'parent': row['info_id.parent'],
                'id' : row['info_id']
            }

       
        # check
        for info_order, data in result.items():
            if str(data['parent']) != 'nan':
                parent_order = ",".join(info_order.split(",")[:-1])
                if ',' in parent_order:
                    if parent_order not in result:
                        #print(result)
                        print("check_thread_structure: missing: %s" % parent_order)
                        ok = False
                    else:
                        assert result[parent_order]['id'] == data['parent']
            else:
                continue
       
    if ok:            
        print("check_thread_structure: OK")
    else:
        print("check_thread_structure: Error")


def clean_text(s):
    """ Removes [linebreak] annotations and multiple spaces 
    s: string
    """
    s = s.lower().replace("[linebreak]", "")
    return " ".join(s.split())


def check_annotations(df, output_file="check_data_entries.txt"):
    """ Checks whether the highlights appear in the original text.
        Writes the ids of problematic entries to output_file
        
        df: dataframe
        output_file: output file
        """
    with open(output_file, 'w') as output_file:
        count = 0

        to_write = []
        for index, row in df.iterrows():
            if row['annotation_Primary'] != 'Neutral':
                if str(row['annotation_highlighted']) == 'nan':
                    print("check_annotations nan: %s" %row)
                    continue

                orig_text = clean_text(row['meta_text'])
                annotated_text = clean_text(row['annotation_highlighted'])

                if (annotated_text not in orig_text and 'image' not in annotated_text):
                    count += 1
                    output_file.write("\nInfo id: %s\n" % row['info_id'])
                    output_file.write("Id: %s\n" % row['id'])
                    output_file.write("Annotated: %s\n" % annotated_text)
                    output_file.write("Orig: %s\n" % row['meta_text'])
                    df.at[index, 'cleaned_status'] = "to_fix"

                    to_write.append("%s\t%s\t\n" % (row['id'], row['info_id']))

        for w in to_write:
            output_file.write(w)

    print("check_annotations: Number of problems: %s" % count)


def check_seen_unseen(df):
    """ Check the subreddit seen/unseen values
    (each entry should have one, it should match) 
    """
    for index, row in df.iterrows():
        
        # seen column needs to be set for every entry
        if not (row['subreddit_seen'] == 1 or row['subreddit_seen'] == 0):
            print("check_seen_unseen error: %s \t%s" % (row['id'], row['subreddit_seen']))
        else:
            # the value needs to match with the specification
            if row['subreddit_seen'] == '0' and not (
                row['info_subreddit'] in contextual_abuse_dataset.ONLY_DEV_SUBREDDITS or
                row['info_subreddit'] in contextual_abuse_dataset.ONLY_TEST_SUBREDDITS):
                print("check_seen_unseen error match 1: %s \t%s" % (row['id'], row['subreddit_seen']))
                
            elif row['subreddit_seen'] == '1' and (
                row['info_subreddit'] in contextual_abuse_dataset.ONLY_DEV_SUBREDDITS or
                row['info_subreddit'] in contextual_abuse_dataset.ONLY_TEST_SUBREDDITS):
                print("check_seen_unseen error match 2: %s \t%s" % (row['id'], row['subreddit_seen']))
                


def check_slurs(df):
    """ Slur shouldn't occur with identity directed abuse with same target """
    
    # First store slurs and their targets
    slur_entries = {} #info_id, targets
    for index, row in df.iterrows():
        if row['annotation_Primary'] == 'Slur':
            if row['info_id'] not in slur_entries:
                slur_entries[row['info_id']] = set()
            slur_entries[row['info_id']].add(row['annotation_Target'])
    
    # Now check: a slur shouldn't have an identity directed abuse
    # annotation with the same target
    for index, row in df.iterrows():
        if row['annotation_Primary'] == 'IdentityDirectedAbuse' and row['info_id'] in slur_entries:
            print("Slurs to Double check: %s \t %s \t %s" % 
                        (row['info_id'], 
                        row['annotation_Target'], 
                        slur_entries[row['info_id']]))


def check_splits(df):
    """ All threads need to be in the same split.
    Some subreddits can only occur in dev or test """

    thread_map = {}
    splits = ["train", "dev", "test"]

    ok = True

    for index, row in df.iterrows():
        # because some entries are 'exclude...'
        if row['split'] not in splits:
            continue

        if row['info_thread.id'] in thread_map:
            if thread_map[row['info_thread.id']] != row['split']:
                print("Error check_splits threads: %s" % row['info_thread.id'])
                ok = False

        else:
            thread_map[row['info_thread.id']] = row['split']

        # now check the subreddits
        if (row['info_subreddit'] in contextual_abuse_dataset.ONLY_DEV_SUBREDDITS and
            row['split'] != 'dev'):
                print("Error check_splits subreddits: %s" % row['info_thread.id'])
                ok = False

        elif (row['info_subreddit'] in contextual_abuse_dataset.ONLY_TEST_SUBREDDITS and
            row['split'] != 'test'):
                print("Error check_splits subreddits: %s" % row['info_thread.id'])
                ok = False

    if ok:
        print("check_splits: OK")


def check_neutral_annotations(df):
    """ Neutral posts shouldn't have an annotation """
    ok = True

    for index, row in df.iterrows():
        if (row['annotation_Primary'] == 'Neutral' and 
            row['annotation_highlighted'] != '"NA"'):
            print("Neutral annotation error: %s" % row['id'])
            ok = False

    if ok:
        print("check_neutral_annotations: OK")


def check_context(df):
    """ Check the values of annotation_Context """
    ok = True
    for index, row in df.iterrows():
        if (row['annotation_Primary'] != 'Neutral' and
            (row['annotation_Context'] != 'CurrentContent' and
            row['annotation_Context'] != 'PreviousContent')):

            print("Check context error: %s" % row['id'])
            ok = False

        if (row['annotation_Primary'] == 'Neutral' and 
            row['annotation_Context'] != '"NA"'):
            print("Check context error: %s" % row['id'])
            ok = False
    if ok:
        print("check_context: OK")


def check_targets(df):
    """ Check if posts have targets (where applicable)"""
    ok = True
    empty = set(['"NA"', 'na', 'NA', ''])
   
    selected_prim_cats = set(["Slur", 
                              "IdentityDirectedAbuse",
                              "AffiliationDirectedAbuse"])

    for index, row in df.iterrows():
         # these categories should have a target
        if (row['annotation_Primary'] in selected_prim_cats):
            if (row['annotation_Target'] in empty or
                row['annotation_Target_top.level.category'] in empty):

                print("Check targets error 1: %s" % row['id'])
                ok = False

        # these categories shouldn't have a target
        else:
            if (row['annotation_Target'] != '"NA"' or
                row['annotation_Target_top.level.category'] != '"NA"'):
                print("Check targets error 2: %s" % row['id'])
                ok = False

    if ok:
        print("check_targets: OK")
    

def check_counterspeech(df):
    """ Counterspeech annotations should always be based on 
    previous content """
    ok = True
    for index, row in df.iterrows():
        if (row['annotation_Primary'] == "CounterSpeech" 
            and row['annotation_Context'] != "PreviousContent"):
            print("Counterspeech context error : %s" % row['id'])
            ok = False

    if ok:
        print("check_counterspeech: OK")


def check_nas(df):
    ok = True
    for index, row in df.iterrows():
        for index2, col in enumerate(row):
            if col == "NA": # should be "NA"
               ok = False
               print("Check NA: %s" % row['id'])
    if ok:
        print("check_nas: OK")


if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser(description='Checking the dataset')

    my_parser.add_argument('--input_file',
                        metavar='input file',
                        type=str,
                        help='path to input file (tsv)',
                        required=False)
 
    args = my_parser.parse_args()
    args = vars(args)
    
    print("Check the data: %s" % args["input_file"])
    df = pd.read_csv(args["input_file"], delimiter="\t", 
                    quoting=csv.QUOTE_NONE,  keep_default_na=False)
    
    check_counterspeech(df)
    check_nas(df)
    check_neutral_annotations(df)
    check_context(df)
    check_targets(df)
    check_splits(df)
    check_slurs(df)
    check_seen_unseen(df)
    check_body_post(df)
    check_multiple_entries(df)
    check_labels(df)  
    check_empty_text(df)    
    check_thread_structure(df)
    check_info_order_as_key(df)
    check_annotations(df, output_file="check_annotations_%s.txt" % int(time.time()))
    