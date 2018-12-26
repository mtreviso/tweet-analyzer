# Tweet Analyzer for Brazilian Portuguese

A simple PyTorch tool to analyze tweets based on TweetSentBR.

## Usage

Import the Analyzer:
```python
>>> from tweetan import Analyzer
>>> a = Analyzer(model='simple_lstm', output_dir='saved-models/')
```

Train it using TTsBR:
```python
>>> a.train('data/ttsbr/trainTT', epochs=10, patience=2, batch_size=32)
```

Save and load whenever you want:
```python
>>> a.save('training-case-1')
>>> a.load('training-case-1')
```

Get the labels (negative, neutal, positive) for a collection of tweets:
```python
>>> tweets = ['ahhh uma semana pra ter de novo :(', 
		  'A mãe do serginho ❤ ❤ ❤ ❤'] 
>>> tags = a.analyze(texts)
[0, 2]
```



## TODO

- Add tests
- Add more complex models: `kimcnn, socher-trnn, etc`
- Fix lint errors: `flake8 tweetan`
- Fix Analyzer: detach dataset loading from training/testing and save its vocab together with the model
