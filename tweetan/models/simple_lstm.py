import torch
from torch import nn

class SimpleLSTM(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, nb_labels):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.nb_labels = nb_labels

        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        self.lstm_layer = nn.LSTM(emb_size, hidden_size, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, nb_labels)
        self.dropout = nn.Dropout(0.1)
        self.dropout_out = nn.Dropout(0.5)

        self.hidden = None
        self.bidir = False

    def _init_hidden(self, x):
        bs = x.shape[0]
        ndirs = 2 if self.bidir else 1
        self.hidden = (torch.zeros(ndirs, bs, self.hidden_size), 
                        torch.zeros(ndirs, bs, self.hidden_size))

    def forward(self, x):
        # x is a batch of words
        # if self.hidden_size is None:
        self._init_hidden(x)

        # (bs, seqlen) -> (bs, seqlen, emb_size)
        x = self.emb_layer(x)

        # (bs, seqlen, emb_size) -> (bs, seqlen, hidden_size)
        # x, self.hidden = self.lstm_layer(x, self.hidden)
        x, _ = self.lstm_layer(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # (bs, seqlen, hidden_size) -> (bs, hidden_size)
        x = x[:, -1].squeeze()

        # (bs, hidden_size) -> (bs, nb_labels)
        x = self.output_layer(x)
        # x = self.dropout(x)

        x = torch.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':

    from tweetan.dataset.fields import WordsField, LabelField
    from tweetan.dataset.corpus import TweetSentBRCorpus
    from tweetan.dataset.dataset import TweetDataset
    from tweetan.dataset import iterator

    words_field = WordsField()
    label_field = LabelField()
    attr_fields = [('words', words_field), ('label', label_field)]

    ttsbr_train = TweetSentBRCorpus(attr_fields, sort_attr='words')
    ttsbr_train.read('../../data/ttsbr/trainTT')
    train_dataset = TweetDataset(ttsbr_train)

    ttsbr_test = TweetSentBRCorpus(attr_fields, sort_attr='words')
    ttsbr_test.read('../../data/ttsbr/testTT')
    test_dataset = TweetDataset(ttsbr_test)
    
    words_field.build_vocab(train_dataset, max_size=10000, min_freq=1)

    PAD_ID = words_field.vocab.stoi['<pad>']
    UNK_ID = words_field.vocab.stoi['<unk>']

    train_iterator = iterator.build(train_dataset, batch_size=32)
    test_iterator = iterator.build(test_dataset, batch_size=128)


    vocab_size = len(words_field.vocab)
    emb_size = 100
    hidden_size = 100
    nb_labels = 3

    model = SimpleLSTM(vocab_size, emb_size, hidden_size, nb_labels)

    from torch import optim
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss(ignore_index=PAD_ID)


    acum_loss = 0
    c = 0
    for epoch in range(10):
        print('Epoch %d' % (epoch + 1))
        
        corrects = 0
        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()   # zero the gradient buffers
            output = model(batch.words)

            # print(output.shape, batch.label.shape)
            loss = criterion(output, batch.label)
            loss.backward()
            acum_loss += loss.exp().item()
            optimizer.step()    # Does the update
            c += 1

            corrects += torch.sum(torch.argmax(output, dim=-1) == batch.label).item()
            # print(corrects)

            print('Iter %d/%d | Loss: %f | Acc: %f' % 
                (i+1, 
                len(train_iterator),
                acum_loss/c, 
                corrects/len(train_dataset)), end='\r')

        print('')

    with torch.no_grad():
        corrects = 0
        for batch in enumerate(test_iterator): 
            output = model(batch.words)
            corrects += torch.sum(torch.argmax(output, dim=-1) == batch.label).item()
            print('Acc: %f' % (corrects/len(test_dataset)), end='\r')
