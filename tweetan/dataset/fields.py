from torchtext.data import Field


class WordsField(Field):

    def __init__(self, **kwargs):

        from nltk import TweetTokenizer
        tokenizer = TweetTokenizer() 

        def tweet_tokenize(sentence):
            return tokenizer.tokenize(sentence)

        super().__init__(unk_token='<unk>', 
                         pad_token='<pad>', 
                         eos_token='<eos>',
                         init_token='<bos>',
                         batch_first=True,
                         tokenize=tweet_tokenize, 
                         **kwargs)

class LabelField(Field):
    def __init__(self, **kwargs):
        super().__init__(sequential=False, 
                         is_target=True, 
                         use_vocab=False,
                         **kwargs)
