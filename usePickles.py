import pickle
from LSTM import LSTM
import matplotlib.pyplot as plt

iteration = 400000

with open('pickles/lstm1-' + str(iteration) + '.pickle', 'r') as handle:
	lstm = pickle.load(handle)

with open('pickles/lstm2-'+ str(iteration) +'.pickle', 'r') as handle:
        lstm2 = pickle.load(handle)

with open('pickles/encoderloss-'+ str(iteration) + '.pickle', 'r') as handle:
        elos = pickle.load(handle)

with open('pickles/decoderloss-'+ str(iteration) +'.pickle', 'r') as handle:
        dlos = pickle.load(handle)


maxError = 70
for i in range(len(elos)):
    if elos[i]>maxError:
        elos[i] = maxError

for i in range(len(elos)):
    if dlos[i]>maxError:
        dlos[i] = maxError

plt.plot(range(len(elos)), elos, 'b', label='Encoder loss')
plt.plot(range(len(dlos)), dlos, 'g', label='Decoder loss')
plt.xlabel('Time at every 10 epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

data = open('de-json.txt', 'r').read() # should be simple plain text file
words = list(set(data.replace("\n", " <eos> ").split(" ")))
data = data.replace("\n", " <eos>\n").split("\n")
data_size, vocab_size = len(data), len(words)
print('German data has {} sentences, {} unique words.'.format(data_size, vocab_size))
char_to_ix = { w:i for i,w in enumerate(words) }
#ix_to_char = { i:w for i,w in enumerate(words) }

data2 = open('en-json.txt', 'r').read() # should be simple plain text file
words2 = list(set(data2.replace("\n", " <eos> ").split(" ")))
data2 = data2.replace("\n", " <eos>\n").split("\n")
data_size2, vocab_size2 = len(data2), len(words2)
print('English data has {} sentences, {} unique words.'.format(data_size2, vocab_size2))
char_to_ix2 = { w:i for i,w in enumerate(words2) }
ix_to_char2 = { i:w for i,w in enumerate(words2) }

encoder = lstm
decoder = lstm2
with open('testSentences.txt', 'r') as f:
	print("Translate using LSTM:")
	for sentence in f:
		print('German:   ', sentence.strip())
		testArray = sentence.lower().split()
		x = [char_to_ix[w] for w in testArray[:-1]]
		htest, stest = encoder.getHidden(x)
		decoder.hprev = htest 
		decoder.sprev = stest
		num = char_to_ix2['<eos>'.strip()]
		oTest = decoder.translate(num)
		txt = ' '.join(ix_to_char2[ix] for ix in oTest)
		print('English:  ', txt)
		print('\n')
		encoder = lstm
		decoder = lstm2
		
