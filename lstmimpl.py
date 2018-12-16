import numpy as np
import sys
import pickle
import logging
from datetime import datetime

# create logger with 'spam_application'
logger = logging.getLogger('lstmimpl')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('lstm.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

class lstmimpl(object):

    def __init__(self, inputdim, outputdim, hiddendim, learning_rate, is_decoder, losses):        
        self.inputdim = inputdim
        self.hiddendim = hiddendim
        self.is_decoder = is_decoder
        self.outputdim = outputdim
        self.losses = losses

        # Previous batch hidden state stored
        self.hprev = np.zeros((hiddendim , 1))
        self.sprev = np.zeros((hiddendim , 1))

        # Parameters
        self.Why = np.random.randn(outputdim, hiddendim)*0.01
        # Input to hidden layers
        self.Wf = np.random.randn(hiddendim, hiddendim + inputdim)*0.01 
        self.Wi = np.random.randn(hiddendim, hiddendim + inputdim)*0.01 
        self.Wc = np.random.randn(hiddendim, hiddendim + inputdim)*0.01 
        self.Wo = np.random.randn(hiddendim, hiddendim + inputdim)*0.01 
        # Biases
        self.by = np.zeros((outputdim, 1))
        self.bf = np.zeros((hiddendim, 1)) 
        self.bi = np.zeros((hiddendim, 1)) 
        self.bc = np.zeros((hiddendim, 1)) 
        self.bo = np.zeros((hiddendim, 1)) 

        # Parameters for adagrad update
        self.mWhy =  np.zeros_like(self.Why)
        self.mWf =  np.zeros_like(self.Wf)
        self.mWi =  np.zeros_like(self.Wi)
        self.mWc =  np.zeros_like(self.Wc)
        self.mWo =  np.zeros_like(self.Wo)
        self.mby =  np.zeros_like(self.by)
        self.mbf =  np.zeros_like(self.bf)
        self.mbi =  np.zeros_like(self.bi)
        self.mbc =  np.zeros_like(self.bc)
        self.mbo =  np.zeros_like(self.bo)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def add_loss(self, loss):
        self.losses.append(loss)

    def train(self, inputs, targets):
        xs, hs, xh, ys, ps, f, inp, cc, o, s = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        s[-1] = np.copy(self.sprev)  

        # Initialize deriviatives
        dWhy = np.zeros_like(self.Why)
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)
        dby = np.zeros_like(self.by)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        dhnext = np.zeros_like(self.hprev)
        dsnext = np.zeros_like(self.sprev)

        # Forward Propagation
        loss = 0
        for t in range(len(inputs)):            
            xs[t] = np.zeros((self.inputdim,1)) 
            # one hot encoding of word
            if(self.is_decoder):
                if(t==0):
                    xs[t][inputs[t]] = 0
                else:
                    xs[t][inputs[t]] = 1
            else:
                xs[t][inputs[t]] = 1
                
            # Concatenate x and h
            xh[t] = np.hstack((xs[t].ravel(), hs[t-1].ravel())).reshape(self.inputdim+self.hiddendim,1)
            # Forget gate equation
            f[t]  = self.sigmoid(np.dot(self.Wf, xh[t]) + self.bf)
            # Input gate layer decides which values we’ll update. 
            inp[t] = self.sigmoid(np.dot(self.Wi, xh[t]) + self.bi)
            # C~ is denoted by cc
            # Indicates new candidate values, that could be added to the state
            cc[t] = np.tanh(np.dot(self.Wc, xh[t]) + self.bc)
            # Update new state
            s[t] = f[t] * s[t-1] + inp[t] * cc[t]
            # This layer decides parts of the cell state we’re going to output
            o[t] = self.sigmoid(np.dot(self.Wo, xh[t]) + self.bo)
            # New hidden layer
            hs[t] = o[t] * np.tanh(s[t])
            # calculate cross-entropy loss
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 
            loss += -np.log(ps[t][targets[t],0]) 

        # Backward propagation: Do gradient descent on above parameters from reverse
        for t in reversed(range(len(inputs))):
            # Back propagation for the softmax layer
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
        
            dh = np.dot(self.Why.T, dy) + dhnext
            
            # Back propagation for the output gate
            do = o[t]*(1-o[t]) *dh * np.tanh(s[t])
            dWo += np.dot(do, xh[t].T)
            dbo += do
            
            # Back propagation for cell state
            ds = dh * o[t] * (1-np.tanh(s[t])**2) + dsnext
            
            # Back propagation for input gate
            dinp = inp[t]*(1-inp[t]) * cc[t] * ds
            dWi += np.dot(dinp, xh[t].T)
            dbi += dinp
            
            # Back propagation for new candidate values
            dcc = (1-cc[t]**2) * inp[t] * ds
            dWc += np.dot(dcc, xh[t].T)
            dbc += dcc 
            
            # Back propagation for forget gate
            df = f[t]*(1-f[t]) * s[t-1] * ds
            dWf += np.dot(df, xh[t].T)
            dbf += df       
                  
            # Combining all to find hnext
            dxh = np.zeros_like(xh[t])
            dxo = np.dot(self.Wo.T, do)
            dxi = np.dot(self.Wi.T, dinp)
            dxcc = np.dot(self.Wc.T, dcc)
            dxf = np.dot(self.Wf.T, df)    
            dxh = dxo + dxi + dxcc + dxf
            
            # Update values for future state 
            dsnext = ds * f[t]
            
            # Update values for future hidden value 
            dhnext = dxh[(xh[t].shape[0]-self.hiddendim):,:]

        # Use clipping to avoid exploding gradients
        for dparam in [dWf, dWi, dWc,dWo,dWhy, dbf,dbi,dbc,dbo, dby]:
            np.clip(dparam, -5, 5, out=dparam) 

        # Apply Adagrad for updating parameters
        for param, dparam, mem in zip([self.Wf, self.Wi, self.Wc, self.Wo, self.Why, self.bf, self.bi, self.bc, self.bo, self.by], 
                                [dWf, dWi, dWc,dWo,dWhy, dbf,dbi,dbc,dbo, dby], 
                                [self.mWf, self.mWi,self.mWc,self.mWo, self.mWhy,self.mbf,self.mbi,self.mbc,self.mbo, self.mby]):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) 
        
        self.hprev = hs[len(inputs)-1]
        self.sprev = s[len(inputs)-1]

        return loss
    
    # Returns the h vector and current state
    def get_hidden(self, xin):
        h = np.zeros_like(self.hprev)
        sp = np.zeros_like(self.sprev)
        for t in range(len(xin)): 
            x = np.zeros((self.inputdim, 1))
            x[xin[t]] = 1
            xh = np.hstack((x.ravel(), h.ravel())).reshape(self.inputdim+self.hiddendim,1)
            f  = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            inp = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            cc = np.tanh(np.dot(self.Wc, xh) + self.bc)
            sp = f * sp + inp * cc
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            h = o * np.tanh(sp)             
        return h, sp
        
    # Returns the indices of translated words in target vocabulary along with candidate probabilities
    def translate(self, eos_index):
        h = self.hprev
        sp = self.sprev
        x = np.zeros((self.inputdim,1))
        y = np.zeros((self.inputdim,1))
        top_five_probabilites = []
        top_five_indices = []
        indices = []
        for ii in range(20):
            xh = np.hstack((x.ravel(), h.ravel())).reshape(self.inputdim+self.hiddendim,1)
            f  = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            inp = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            cc = np.tanh(np.dot(self.Wc, xh) + self.bc)
            sp = f * sp + inp * cc
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            h = o * np.tanh(sp)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            if ii == 3:
                p_flat = np.reshape(p, len(p))
                top_five_indices = (-p_flat).argsort()[:5].tolist()
                for index in top_five_indices:
                    top_five_probabilites.append(p_flat[index])
            i = p.argmax()
            x = np.zeros((self.inputdim, 1))
            x[i] = 1
            indices.append(i)
            if(eos_index == i):
                break

        return indices, top_five_indices, top_five_probabilites

# Save models
def persist_models(n, encoder, decoder):
    name1 = 'models/encoder_' + str(n) + '.model'
    with open(name1, 'wb') as handle:
        pickle.dump(encoder, handle)
    name2 = 'models/decoder_' + str(n) + '.model'
    with open(name2, 'wb') as handle:
        pickle.dump(decoder, handle)
    logger.info("Models saved successfully!")

def load_persisted_models(encoder_model_file_name, decoder_model_file_name):
    encoder = None
    decoder = None
    with open(encoder_model_file_name, 'rb') as handle:
	    encoder = pickle.load(handle)
    with open(decoder_model_file_name, 'rb') as handle:
        decoder = pickle.load(handle)
    logger.info("Models loaded successfully!")
    return encoder, decoder

# Test a translation while training
def test_translation(word_to_index, word_to_index2, index_to_word2, model1, model2):
    logger.info('Testing Translate: German to English')
    test = "ich habe ein buch dexp <eos>"
    logger.info('German: '+ test)
    testArray = test.split()
    x = [word_to_index[w] for w in testArray[:-1]]
    htest, stest = model1.get_hidden(x)
    model2.hprev = htest 
    model2.sprev = stest
    eos_index = word_to_index2['<eos>'.strip()]
    oTest, top_five_indices, top_five_probabilites = model2.translate(eos_index)
    new_keys = [index_to_word2[i] for i in top_five_indices]
    new_dict = dict(zip(new_keys, top_five_probabilites))
    txt = ' '.join(index_to_word2[i] for i in oTest)
    logger.info('Probabilites for word {} : {}'.format(testArray[3], new_dict))
    logger.info('English: {} \n'.format(txt))

def test_translations(word_to_index, word_to_index2, index_to_word2, model1, model2, n):
    sentences = open('test_sentences.txt', 'r').read().split('\n')
    logger.info('Testing Translate: German to English')
    output = open("TestTranslationsOutput/myfile_{}.txt".format(n), "w")
    for test in sentences:
        logger.info('German: '+ test)
        testArray = test.split()
        x = [word_to_index[w] for w in testArray[:-1]]
        htest, stest = model1.get_hidden(x)
        model2.hprev = htest 
        model2.sprev = stest
        eos_index = word_to_index2['<eos>'.strip()]
        oTest, top_five_indices, top_five_probabilites = model2.translate(eos_index)
        new_keys = [index_to_word2[i] for i in top_five_indices]
        new_dict = dict(zip(new_keys, top_five_probabilites))
        txt = ' '.join(index_to_word2[i] for i in oTest)
        logger.info('Probabilites for word {} : {}'.format(testArray[3], new_dict))
        output.write(txt)
        output.write('\n')
        logger.info('English: {} \n'.format(txt))
    output.close()
            
def start(epochs, load_models, encoder_model_file_name, decoder_model_file_name):
    logger.info("=========================Execution Starts===========================")
    learning_rate = 0.1
    model1 = None
    model2 = None

    #read german text file
    data = open('de-json.txt', 'r').read()
    vocab = list(set(data.replace("\n", " <eos> ").split(" ")))
    data = data.replace("\n", " <eos>\n").split("\n")
    data_size, vocab_size = len(data), len(vocab)
    logger.info('data has {} sentences, {} unique words.'.format(data_size, vocab_size))

    #dictionary for encoding and decoding from 1-of-k
    word_to_index = { w:i for i,w in enumerate(vocab) }

    #read english text file
    data2 = open('en-json.txt', 'r').read()
    vocab2 = list(set(data2.replace("\n", " <eos> ").split(" ")))
    data2 = data2.replace("\n", " <eos>\n").split("\n")
    data_size2, vocab_size2 = len(data2), len(vocab2)
    logger.info('data has {} sentences, {} unique words.'.format(data_size2, vocab_size2))

    #dictionary for encoding and decoding from 1-of-k
    word_to_index2 = { w:i for i,w in enumerate(vocab2) }
    index_to_word2 = { i:w for i,w in enumerate(vocab2) }

    if load_models == True:
        model1, model2 = load_persisted_models(encoder_model_file_name, decoder_model_file_name)
        test_translation(word_to_index, word_to_index2, index_to_word2, model1, model2)
    else:
        model1 = lstmimpl(len(vocab), len(vocab), 100, learning_rate, False, [])
        model2 = lstmimpl(len(vocab2), len(vocab2), 100, learning_rate, True, [])

    n = 0
    
    for epoch in range(epochs):
        for i in range(len(data)): 
            
            sentence = data[i]
            words_list = sentence.split()
            x = [word_to_index[w] for w in words_list[:-1]]        
            y = [word_to_index[w] for w in words_list[1:]]
            loss = model1.train(x, y)
            
            model2.hprev = model1.hprev
            model2.sprev = model1.sprev
            
            sentence = data2[i]
            words_list = ["is"] + sentence.split()
            x = [word_to_index2[w] for w in words_list[:-1]] 
            words_list = sentence.split()
            y = [word_to_index2[w] for w in words_list]
            loss2 = model2.train(x, y)

            if n%500==0:
                logger.info('Epoch: {}, Iteration: {}, Encoder Loss: {}, Decoder Loss: {}, Learning Rate: {}'.format(epoch, n, loss, loss2, learning_rate))
                model1.add_loss(loss)
                model2.add_loss(loss2)
                test_translations(word_to_index, word_to_index2, index_to_word2, model1, model2, n)
                
            model1.hprev = np.zeros((100,1))
            model1.sprev = np.zeros((100,1))

            n += 1 

        persist_models(epoch, model1, model2)

if __name__ == "__main__":
    try:
        epochs = int(sys.argv[1])
        load_models = False
        if sys.argv[2] == 'Y':
            load_models = True
            start(epochs, load_models, sys.argv[3], sys.argv[4])
        else:
            start(epochs, load_models, None, None)
    except ValueError:
        logger.error("Invalid command! \nUse command like 'python lstmimpl.py 10(epochs) Y(Load Persisted Models) models/encoder_0.model models/decoder_0.model'")
        sys.exit()