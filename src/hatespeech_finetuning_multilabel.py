import argparse
import os
import random

import numpy as np
import torch

from nlp import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, DistilBertTokenizer

import bert_multilabel
import distillbert_multilabel
import contextual_abuse_dataset
import classification_util


label_map, inv_label_map  = contextual_abuse_dataset.get_label_map()


def set_seed(seed):
    """ 
    Set the seed to make sure results are reproducable
    Based on https://github.com/huggingface/transformers/issues/1410
    """
    print("Set the seed to %s" % seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_label(batch):
    """ because labels_info could potentially also contain other info
        (like the rationales), get label info and store it in labels """
    batch['labels'] = batch['labels_info']['label']
    return batch


def compute_metrics(pred, threshold=0.5):
    """ Evaluate the output of the model """

    print("Compute metrics")
    # gold labels
    # [[1 0 0 0 0 0]
    #...
    # [0 0 1 0 0 0]]
    test_labels_multi = pred.label_ids
    
    # predictions
    s = torch.sigmoid(torch.from_numpy(pred.predictions))
    pred_labels_multi = (s >= threshold).type(torch.uint8)
    
    results, results_str = classification_util.get_multilabel_results(
                test_labels_multi, pred_labels_multi, inv_label_map)

    return results


class MyTrainerMultiLabel(Trainer):
    """ A modified trainer to handle multi-label classification
        with skewed class weights """

    def set_class_weights(self, weights):
        """ Set the class weights
        
        :param list[float] weights: The weights for each class
        """
        self.class_weights = torch.FloatTensor(weights).cuda()

    def compute_loss(self, model, inputs): 
        """ Computing the loss.
        We use BCEWithLogitsLoss for multilabel classification
        (loosely based on following https://medium.com/huggingface/multi-label-text
        -classification-using-bert-the-mighty-transformer-69714fa3fb3d),
        and also weight the classes according to their 
        class distribution 
        """

        # get labels, outputs, logits
        # similar to standard Trainers
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0] 

        # Use BCEWithLogitsLoss
        loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights)
        labels = labels.type_as(logits)
        return loss_fct(logits, labels) ## (output, target)


def fine_tune(model, output_dir, freeze_weights=True, run="dev", 
             random_seed=42, learning_rate=0.00002,
             weight_decay=0.01):

    print("Output directory %s" % output_dir)

    if os.path.isdir(output_dir):
        print("Output directory already exists")
        return

    print("Run %s" % run)
    
    # set the seeed
    set_seed(random_seed)

    print("\n** Get the dataset **\n")
    
    train_dataset = load_dataset('contextual_abuse_dataset.py', split="train")
    test_dataset = (load_dataset('contextual_abuse_dataset.py', split="validation") if run == "dev" else
                    load_dataset('contextual_abuse_dataset.py', split="test"))

    # Get the model
    print("\n** Read in the pre-trained model **\n")
    
    if model == "BERT":
        print("BERT")
        model = bert_multilabel.BertForMultiLabelSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=contextual_abuse_dataset.NUM_LABELS)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif model == "DistilBERT":
        print("DistilBERT")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = distillbert_multilabel.DistilBertForMultiLabelSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', num_labels=contextual_abuse_dataset.NUM_LABELS) 
    else:
        print("Unknow model")
        return

    
    # Freeze the parameters of the underlying encoder
    # see https://huggingface.co/transformers/training.html
    if freeze_weights:
        for param in model.base_model.parameters():
            param.requires_grad = False


    print("** Process the data")

    # Each instance is a dictionary
    # Tokenize: adds input_ids (token IDS) and attention_mask to each instance.
    train_dataset = train_dataset.map(lambda batch: tokenizer(batch['text'], 
                                      padding=True, truncation=True), 
                                      batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(lambda batch: tokenizer(batch['text'], 
                                      padding=True, truncation=True), 
                                      batched=True, batch_size=len(test_dataset))

    # Shuffle trainig
    train_dataset = train_dataset.shuffle(seed=random_seed)

    # Get labels and copy to 'labels' field
    train_dataset = train_dataset.map(get_label)
    test_dataset = test_dataset.map(get_label)
    
    # calculate the weights for the classes
    weights = classification_util.calculate_class_weights(train_dataset["labels"])
   
    # Transform the labels
    mlb = MultiLabelBinarizer(classes=np.arange(contextual_abuse_dataset.NUM_LABELS))
    # we still need to fit the binarizer, even though classes are set
    mlb.fit(train_dataset["labels"])
    
    train_dataset = train_dataset.map(lambda batch: {'labels': mlb.transform([batch['labels']])[0]})
    test_dataset = test_dataset.map(lambda batch:  {'labels': mlb.transform([batch['labels']])[0]})
    
    # Get right input format
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # train
    print("** Set the training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16, 
        seed=random_seed,
        warmup_steps=100, 
        weight_decay=weight_decay,
        learning_rate= learning_rate,
        evaluation_strategy="no", 
        save_steps=1000,
        logging_dir= output_dir + "logs/",
    )

    print("**\nTrainer\n*****")
    trainer = MyTrainerMultiLabel(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.set_class_weights(weights)
    trainer.train()
    trainer.evaluate()
    trainer.save_model(output_dir)
    print("Finished saving")


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='Transformers')

    # Add the arguments
    my_parser.add_argument('action',
                        metavar='action',
                        type=str,
                        help='Which action to perform: tune or test')

    my_parser.add_argument('model',
                        metavar='model',
                        type=str,
                        help='Which model to use')

    my_parser.add_argument('output_dir',
                        metavar='output_dir',
                        type=str,
                        help='Output directory')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    args = vars(args)
    
    print("Transformers")
    if args["action"] == "tune":
        print("Tune %s" % args["model"])

        # for tuning, only run with 2 random seeds.
        random_seeds = [78,971242]
        weight_decays = [0, 0.01, 0.03]
        learning_rates = [0.00002, 0.00003, 0.00004, 0.00005]
        freeze_weights = [False]

        for random_seed in random_seeds:
            for weight_decay in weight_decays:
                for learning_rate in learning_rates:
                    for fw in freeze_weights:
                        model_output_dir = "cad_%s_%s_%s_%.5f_%s/" % (random_seed, 
                                                            args["model"],
                                                            weight_decay, 
                                                            learning_rate, fw)
                        
                        fine_tune(model=args["model"], 
                                output_dir=args["output_dir"] + model_output_dir,
                                freeze_weights=fw, 
                                random_seed=random_seed,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                run="dev")


    if args["action"] == "test":
        if  args["model"] == "DistilBERT" or args["model"] == "BERT":
            print("Train and test: %s" % args["model"])
            random_seeds = [78,971242,8923,601,92064]
            for random_seed in random_seeds:

                weight_decay = None 
                learning_rate = None

                if args["model"] == "DistilBERT":
                    weight_decay = 0
                    learning_rate = 0.00005
                elif args["model"] == "BERT":
                    weight_decay = 0.01
                    learning_rate = 0.00003
                else:
                    print("Unknown model")
                    sys.exit(0)

                freeze_weights = False
                model_output_dir = "cad_final_%s_%s_%s_%.5f_%s/" % (random_seed, 
                                                            args["model"],
                                                            weight_decay, 
                                                            learning_rate, freeze_weights)
                        
                
                fine_tune(model=args["model"], 
                        output_dir=args["output_dir"] + model_output_dir,
                        freeze_weights=freeze_weights, 
                        random_seed=random_seed,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        run="test")