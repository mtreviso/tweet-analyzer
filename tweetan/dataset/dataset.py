from torchtext.data import Dataset

class TweetDataset(Dataset):

    def __init__(self, corpus, filter_pred=None):
        attr_fields = corpus.attr_fields.items()
        list_of_samples = list(corpus.create_torchtext_examples())
        super().__init__(list_of_samples, attr_fields, filter_pred)
        self.sort_key = lambda x: len(getattr(x, corpus.sort_key))


if __name__ == '__main__':

    import os
    from torchtext.data import BucketIterator

    from tweetan.dataset.fields import WordsField, LabelField
    from tweetan.dataset.corpus import TweetSentBRCorpus

    words_field = WordsField()
    label_field = LabelField()

    attr_fields = [('words', words_field), ('label', label_field)]

    ttsbr_test = TweetSentBRCorpus(attr_fields, sort_attr='words')
    ttsbr_test.read('../../data/ttsbr/testTT')


    def filter_len(x):
        return 1 <= len(x.words) <= 50

    text_dataset = TweetDataset(ttsbr_test, filter_pred=filter_len)
    for i, ex in enumerate(text_dataset):
        print(ex.words, ex.label)
        if i == 4:
            break


    words_field.build_vocab(text_dataset, max_size=10000, min_freq=1)

    PAD_ID = words_field.vocab.stoi['<pad>']
    UNK_ID = words_field.vocab.stoi['<unk>']


    def my_sort_key(ex):
        return len(ex.words)

    iterator = BucketIterator(
        dataset=text_dataset,
        batch_size=64,
        repeat=False,

        # sorts the data within each minibatch in decreasing order according
        # set to true if you want use pack_padded_sequences
        sort_key=text_dataset.sort_key,
        sort=True,
        sort_within_batch=True,
        # shuffle batches
        shuffle=True,
        # device=torch.device('cpu'),
        train=True
    )
    
    for batch in iterator:
        print(batch.words)
        x = batch.words.data
        y = batch.label.data

        for sent, label in zip(x, y):
            text = ' '.join([words_field.vocab.itos[w] for w in sent])
            print(text, label.item())
        break
