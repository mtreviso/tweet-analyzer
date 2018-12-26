from torchtext import data
import re


class Corpus:

    def __init__(self, attr_fields, sort_attr=None):
        # list of examples where each instance contain a list of fields
        self.samples = []
        self.attr_fields = dict(attr_fields)

        # a map from attrs name to their indexes
        self.attr_index = dict(zip(self.attr_fields.keys(), range(len(self.attr_fields))))

        # pick the first attr from attr_fields list if sort attribute is not specified
        self.sort_key = attr_fields[0][0] if sort_attr is None else sort_attr

    def read(self, path, **kwargs):
        raise NotImplementedError

    def create_torchtext_examples(self):
        for s in self.samples:
            yield data.Example.fromlist(s, self.attr_fields.items())

    def __iter__(self):
        for s in self.samples:
            yield s

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)



class TweetSentBRCorpus(Corpus):

    def _process(self, text):
        text = re.sub(r'\d', '0', text)
        text = re.sub(r'\b(\w[A-Z0-9]+\b)', r' <upper> \1 ', text)
        text = re.sub(r'\ +', ' ', text)
        text = text.lower()
        # remove stopwords
        # lemmatize
        # transform email <email>
        # transform url <url>
        # etc ...
        return text.strip()

    def _read_file(self, filepath, label):
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                line = self._process(line)
                self.samples.append([line, label])

        if len(self.attr_fields) != 2:
            raise Exception('Number of fields should be equal to 2.')

    def read(self, path):
        self._read_file(path + '.neg', 0)
        self._read_file(path + '.neu', 1)
        self._read_file(path + '.pos', 2)


class TextCorpus(Corpus):
    def read(self, texts):
        for text in texts:
            self.samples.append([text])


if __name__ == '__main__':

    from nltk import TweetTokenizer
    tokenizer = TweetTokenizer() 

    def tweet_tokenize(sentence):
        global tokenizer
        return tokenizer.tokenize(sentence)


    words_field = data.Field(unk_token='<unk>', 
                             pad_token='<pad>', 
                             eos_token='<eos>',
                             init_token='<bos>',
                             batch_first=True,
                             tokenize=tweet_tokenize)

    labels_field = data.Field(sequential=False, is_target=True)

    attr_fields = [('words', words_field), ('labels', labels_field)]

    ttsbr_test = TweetSentBRCorpus(attr_fields, sort_attr='words')
    ttsbr_test.read('../../data/ttsbr/testTT')

    for sample in ttsbr_test:
        print(sample)

