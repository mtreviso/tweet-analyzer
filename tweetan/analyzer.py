import numpy as np
import torch
from torch import optim
from torch import nn

from tweetan.dataset.fields import WordsField, LabelField
from tweetan.dataset.corpus import TweetSentBRCorpus, TextCorpus
from tweetan.dataset.dataset import TweetDataset
from tweetan.dataset import iterator
from tweetan.models import SimpleLSTM, KimCNN
from tweetan.trainer import Trainer



class Analyzer:
    
    def __init__(self, model='simple_lstm', output_dir='data/'):
        if model == 'simple_lstm':
            self.model_class = SimpleLSTM
        elif model == 'kim_cnn':
            self.model_class = KimCNN
        else:
            raise Exception('Model {} not available'.format(model))
        self.trainer = None
        self.output_dir = output_dir

        self.words_field = WordsField()
        self.label_field = LabelField()
        self.attr_fields = [('words', self.words_field), ('label', self.label_field)]


    def train(self, dataset_path, epochs=10, patience=0, batch_size=32):

        ttsbr_train = TweetSentBRCorpus(self.attr_fields, sort_attr='words')
        ttsbr_train.read(dataset_path)
        train_dataset = TweetDataset(ttsbr_train)
        train_iterator = iterator.build(train_dataset, batch_size=batch_size)

        self.words_field.build_vocab(train_dataset, max_size=10000, min_freq=1)
        PAD_ID = self.words_field.vocab.stoi['<pad>']
        
        vocab_size = len(self.words_field.vocab)
        emb_size = 100
        hidden_size = 100
        nb_labels = 3

        model = self.model_class(vocab_size, emb_size, hidden_size, nb_labels)
        criterion = nn.NLLLoss(ignore_index=1)
        optimizer = optim.Adam(model.parameters())

        self.trainer = Trainer(model, criterion, optimizer, self.output_dir)
        self.trainer.fit(train_iterator, epochs, patience=patience)

    def test(self, dataset_path, batch_size=128, return_preds=False):
        ttsbr_test = TweetSentBRCorpus(self.attr_fields, sort_attr='words')
        ttsbr_test.read(dataset_path)
        test_dataset = TweetDataset(ttsbr_test)
        test_iterator = iterator.build(test_dataset, batch_size=batch_size, is_train=False)
        return self.trainer.eval(test_iterator, return_preds=return_preds)


    def predict_probas(self, dataset_path, batch_size=128):
        probas = self.test(dataset_path, batch_size=batch_size, return_preds=True)
        return probas


    def predict_labels(self, dataset_path, batch_size=128):
        probas = self.predict_probas(dataset_path, batch_size=batch_size)
        labels = []
        for pred in probas:
            labels.append(np.argmax(pred, axis=-1))
        return labels

    def accuracy(self, dataset_path):
        ttsbr_test = TweetSentBRCorpus(self.attr_fields, sort_attr='words')
        ttsbr_test.read(dataset_path)
        test_dataset = TweetDataset(ttsbr_test)
        test_iterator = iterator.build(test_dataset, is_train=False)
        _, preds = self.trainer.eval(test_iterator, return_preds=True)
        golds = []
        pred_labels = []
        for batch in test_iterator:
            golds.extend(batch.label.numpy().tolist())
        for pred in preds:
            pred_labels.append(np.argmax(pred, axis=-1))
        acc = np.mean(np.array(pred_labels) == np.array(golds))
        return acc


    def analyze(self, texts, batch_size=128, prediction_type='labels'):
        if isinstance(texts, str):
            texts = [texts]
        c = TextCorpus(self.attr_fields[:1])
        c.read(texts)
        c_dataset = TweetDataset(c)
        c_iterator = iterator.build(c_dataset, is_train=False)

        preds = []
        self.trainer.model.eval()  # informs pytorch that we are going to evaluate this model
        with torch.no_grad():  # don't keep track of gradients
            for batch in c_iterator:
                pred = self.trainer.model(batch.words)
                pred = torch.argmax(pred, dim=-1).tolist()
                if isinstance(pred, int):
                    pred = [pred]
                preds.extend(pred)
        return preds


    def save(self, dirname):
        self.trainer.save(dirname)

    def load(self, dirname):
        self.trainer.load(dirname)
