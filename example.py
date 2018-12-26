from tweetan import Analyzer
a = Analyzer(model='simple_lstm', output_dir='/tmp/')

a.train('data/ttsbr/trainTT', epochs=10, patience=2, batch_size=32)

a.save('myfinalmodel')
a.load('myfinalmodel')

loss_test = a.eval('data/ttsbr/testTT')
print('Loss:', loss_test)

texts = ['ahhh uma semana pra ter de novo :(', 'A mãe do serginho ❤ ❤ ❤ ❤']

tags = a.analyze(texts)
for txt, tag in zip(texts, tags):
	print(tag, ':', txt)
