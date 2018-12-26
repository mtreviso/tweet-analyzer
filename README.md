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


## Installation

First, clone this repository using `git`:

```sh
git clone https://github.com/mtreviso/tweet-analyzer.git
```

 Then, `cd` to the tweet-analyzer folder:
```sh
cd tweet-analyzer
```

Automatically create a Python virtualenv and install all dependencies 
using `pipenv install`. And then activate the virtualenv with `pipenv shell`:
```sh
pip3 install pipenv
pipenv install --skip-lock
pipenv shell
```

The `--skip-lock` flag informs pipenv to ignore its lock mechanism, so it works just like pip in a virtual env and performs a faster installation. 

Finally, run the install command:
```sh
python3 setup.py install
```

Please note that since Python 3 is required, all the above commands (pip/python) 
have to be bounded to the Python 3 version.



## TODO

- Add tests
- Add more complex models: `kimcnn, socher-trnn, etc`
- Fix lint errors: `flake8 tweetan`
- Fix Analyzer: detach dataset loading from training/testing and save its vocab together with the model
- Add build and coverage sticker
- See here how we can easily publish a project to pypi: https://github.com/pypa/twine/