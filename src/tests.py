import unittest

import contextual_abuse_dataset
import classification_analysis
import classification_util

class TestCADDataset(unittest.TestCase):

    def test_gold_label_map(self):
        gold_label_map =  classification_analysis.get_gold_label_map("../data/test/test_data_test.tsv")
    
        expected_map  = {
            'doc1': [0,1,0,1,0],
            'doc2': [1,0,0,0,0],
            'doc3': [0,1,0,0,0],
            'doc4': [0,1,1,0,0],
            'doc5': [1,0,0,0,0],
            'doc6': [0,1,1,1,0],
            'doc7': [0,0,1,0,0]
        }
        self.assertDictEqual(gold_label_map, expected_map)


    def test_get_gold_preds_per_setting(self):
        results = classification_analysis.get_gold_preds_per_setting("../data/test/test_predictions.tsv", 
                                                                    "../data/test/test_data.tsv",
                                                                    "../data/test/test_data_test.tsv")
        full_gold, full_preds = results['full']
        unseen_gold, unseen_preds = results['unseen']
        seen_gold, seen_preds = results['seen']

        expected_seen_gold = [[0,1,0,1,0], [1,0,0,0,0], [0,1,0,0,0], [0,1,1,0,0]]
        self.assertEqual(seen_gold, expected_seen_gold)

        expected_unseen_preds = [[1,0,0,0,0], [1,1,0,1,0], [0,0,0,0,0]]
        self.assertEqual(unseen_preds, expected_unseen_preds)


    def test_label_metrics(self):
        results = classification_analysis.label_analysis_helper("../data/test/test_predictions.tsv", 
                                                                "../data/test/test_data.tsv",
                                                                "../data/test/test_data_test.tsv")
        print(results)
        self.assertEqual(results['num_neutral_violated'], 2)
        self.assertEqual(results['num_too_few_labels'], 2)
        self.assertEqual(results['num_too_many_labels'], 1)


    def test_compute_label_stats(self):
        label_cardinality, label_set_count, pairwise_label_counts = classification_analysis.compute_label_stats(
            ["Neutral", "Neutral", "AffiliationDirectedAbuse", "AffiliationDirectedAbuse,IdentityDirectedAbuse"]
        )
        label_map, inv_label_map = contextual_abuse_dataset.get_label_map()
        
        self.assertEqual(label_cardinality, 5/4)
        self.assertEqual(label_set_count['Neutral'], 2)
        self.assertEqual(label_set_count['AffiliationDirectedAbuse IdentityDirectedAbuse'], 1)
        self.assertEqual(pairwise_label_counts[label_map['AffiliationDirectedAbuse']][label_map['IdentityDirectedAbuse']], 1)
        

    def test_sec_label_metrics(self):
        results = classification_analysis.get_sec_categories_results("../data/test/test_predictions.tsv", 
                                        "../data/test/test_data.tsv")

        self.assertEqual(results['AffiliationDirectedAbuse / animosity_recall'], 0)
        self.assertEqual(results['AffiliationDirectedAbuse / animosity_num_gold'], 2)
        self.assertEqual(results['AffiliationDirectedAbuse / animosity_num_pred'], 0)

        self.assertEqual(results['IdentityDirectedAbuse / animosity_recall'], 1)
        self.assertEqual(results['IdentityDirectedAbuse / animosity_num_gold'], 2)
        self.assertEqual(results['IdentityDirectedAbuse / animosity_num_pred'], 2)


    def test_context_analysis(self):
       
        results = classification_analysis.get_context_results("../data/test/test_predictions.tsv", 
                                        "../data/test/test_data.tsv")
       
        self.assertEqual(results['current_content_recall_PersonDirectedAbuse'], 1)
        self.assertEqual(results['current_content_support_PersonDirectedAbuse'], 1)
        self.assertEqual(results['prev_content_recall_PersonDirectedAbuse'], 1)
        self.assertEqual(results['prev_content_support_PersonDirectedAbuse'], 1)
        self.assertEqual(results['fullc_recall_PersonDirectedAbuse'], 1)
        self.assertEqual(results['fullc_support_PersonDirectedAbuse'], 2)

        self.assertEqual(results['fullc_support_IdentityDirectedAbuse'], 4)
        self.assertEqual(results['current_content_recall_IdentityDirectedAbuse'], 1)
        self.assertEqual(results['current_content_support_IdentityDirectedAbuse'], 1)
        self.assertEqual(results['prev_content_recall_IdentityDirectedAbuse'], 1)
        self.assertEqual(results['prev_content_support_IdentityDirectedAbuse'], 3)   

        self.assertEqual(results['current_content_recall_AffiliationDirectedAbuse'], 0)   


    def test_replace_usernames_subreddits(self):

        t1 = "/r/ChapoTrapHouse is a sub."
        t1_a = "[subreddit] is a sub."
        self.assertEqual(t1_a, 
                        contextual_abuse_dataset.replace_subreddits_usernames(t1))


        t2 = "/r/Canada test /r/Canada"
        t2_a = "[subreddit] test [subreddit]"
        self.assertEqual(t2_a, 
                        contextual_abuse_dataset.replace_subreddits_usernames(t2))

        t3 = "a b /u/test, /r/Canada test /r/Canada"
        t3_a = "a b [user], [subreddit] test [subreddit]"
        self.assertEqual(t3_a, 
                        contextual_abuse_dataset.replace_subreddits_usernames(t3))


    def test_replace_urls(self):

        t1= """*I am just a simple bot, **not** a moderator of this subreddit* | 
                [*bot subreddit*](/r/SnapshillBot) | 
                [*contact the maintainers*](/message/compose?to=\/r\/SnapshillBot)"""

        t1_a = """*I am just a simple bot, **not** a moderator of this subreddit* | 
                *bot subreddit* | 
                *contact the maintainers*"""

        self.assertEqual(t1_a, 
                        contextual_abuse_dataset.replace_urls(t1))

        t2 = "www.test.com hi, a, b, c, http://www.abc.om"
        t2_a = "[LINK] hi, a, b, c, [LINK]"
        self.assertEqual(t2_a, 
                        contextual_abuse_dataset.replace_urls(t2))


        t3 = "[test url](/r/subreddit) hi, a, b, c, www.abc.om"
        t3_a = "test url hi, a, b, c, [LINK]"
        self.assertEqual(t3_a, 
                        contextual_abuse_dataset.replace_urls(t3))             


    def test_calculate_class_weights(self):
        labels = [[0], [1,2,3], [2,3], [0], [0], [0]]
        class_weights = classification_util.calculate_class_weights(labels)
        self.assertEqual(class_weights, [2/4, 5/1, 4/2, 4/2])


    def test_multilabel_accuracy(self):
        true_labels = [[0,1,1], [0,0,1], [1, 0, 1]]
        predicted = [[0,1,1], [0,0,1], [0, 0, 1]]
        self.assertEqual(classification_util.multilabel_accuracy(true_labels, predicted), 2/3)


if __name__ == '__main__':
    unittest.main()
