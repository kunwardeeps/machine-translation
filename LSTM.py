import numpy as np
import matplotlib.pyplot as plt
import pickle
  

class LSTM(object):

    def __init__(self, insize, outsize, hidsize, learning_rate, decoder):        
        self.insize = insize
        self.hidsize = hidsize
        self.decoder = decoder
        self.outsize = outsize

        self.hprev = np.zeros((hidsize , 1))#a [h x 1] hidden state stored from last batch of inputs
        self.sprev = np.zeros((hidsize , 1))#a [h x 1] hidden state stored from last batch of inputs


        #parameters
        self.Why = np.random.randn(outsize, hidsize)*0.01#[y x h]
        self.Wf = np.random.randn(hidsize, hidsize + insize)*0.01 # input to hidden
        self.Wi = np.random.randn(hidsize, hidsize + insize)*0.01 # input to hidden
        self.Wc = np.random.randn(hidsize, hidsize + insize)*0.01 # input to hidden
        self.Wo = np.random.randn(hidsize, hidsize + insize)*0.01 # input to hidden
        self.by = np.zeros((outsize, 1))
        self.bf = np.zeros((hidsize, 1)) # output bias
        self.bi = np.zeros((hidsize, 1)) # output bias
        self.bc = np.zeros((hidsize, 1)) # output bias
        self.bo = np.zeros((hidsize, 1)) # output bias

        #the Adagrad gradient update relies upon having a memory of the sum of squares of dparams
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

    def train(self, inputs, targets):
        #=====initialize=====
        xs, hs, xh, ys, ps, f, inp, cc, o, s = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(self.hprev)
        s[-1] = np.copy(self.sprev)  

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

        #=====forward pass=====
        loss = 0
        for t in range(len(inputs)):            
            xs[t] = np.zeros((self.insize,1)) # encode in 1-of-k representation
            if(self.decoder):
                if(t==0):
                    xs[t][inputs[t]] = 0
                else:
                    xs[t][inputs[t]] = 1
            else:
                xs[t][inputs[t]] = 1
            xh[t] = np.hstack((xs[t].ravel(), hs[t-1].ravel())).reshape(self.insize+self.hidsize,1)
            f[t]  = self.sigmoid(np.dot(self.Wf, xh[t]) + self.bf)
            inp[t] = self.sigmoid(np.dot(self.Wi, xh[t]) + self.bi)
            cc[t] = np.tanh(np.dot(self.Wc, xh[t]) + self.bc)
            s[t] = f[t] * s[t-1] + inp[t] * cc[t]
            o[t] = self.sigmoid(np.dot(self.Wo, xh[t]) + self.bo)
            hs[t] = o[t] * np.tanh(s[t])
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next words
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

        #=====backward pass: compute gradients going backwards=====
        for t in reversed(range(len(inputs))):
            #backprop into the softmax layer
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
        
            dh = np.dot(self.Why.T, dy) + dhnext
            
            #backprop into output perceptron
            do = o[t]*(1-o[t]) *dh * np.tanh(s[t])
            dWo += np.dot(do, xh[t].T)
            dbo += do
            
            #backprop into internal state perceptron
            ds = dh * o[t] * (1-np.tanh(s[t])**2) + dsnext
            
            #backprop into input perceptron
            dinp = inp[t]*(1-inp[t]) * cc[t] * ds
            dWi += np.dot(dinp, xh[t].T)
            dbi += dinp
            
            #backprop into cc perceptron
            dcc = (1-cc[t]**2) * inp[t] * ds
            dWc += np.dot(dcc, xh[t].T)
            dbc += dcc 
            
            #backprop into forget perceptron
            df = f[t]*(1-f[t]) * s[t-1] * ds
            dWf += np.dot(df, xh[t].T)
            dbf += df       
                  
            #find hnext by combining contributions from all perceptrons
            dxh = np.zeros_like(xh[t])
            dxo = np.dot(self.Wo.T, do)
            dxi = np.dot(self.Wi.T, dinp)
            dxcc = np.dot(self.Wc.T, dcc)
            dxf = np.dot(self.Wf.T, df)    
            dxh = dxo + dxi + dxcc + dxf
            
            #update the values of future internal value 
            dsnext = ds * f[t]
            #if(self.decoder==True):
                #print sum(dsnext)
            
            #update the values of future hidden value 
            dhnext = dxh[(xh[t].shape[0]-self.hidsize):,:]
            #if(self.decoder==True):
                #print sum(dhnext)
            
        for dparam in [dWf, dWi, dWc,dWo,dWhy, dbf,dbi,dbc,dbo, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  

        #update RNN parameters according to Adagrad
        for param, dparam, mem in zip([self.Wf, self.Wi, self.Wc, self.Wo, self.Why, self.bf, self.bi, self.bc, self.bo, self.by], 
                                [dWf, dWi, dWc,dWo,dWhy, dbf,dbi,dbc,dbo, dby], 
                                [self.mWf, self.mWi,self.mWc,self.mWo, self.mWhy,self.mbf,self.mbi,self.mbc,self.mbo, self.mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        
        self.hprev = hs[len(inputs)-1]
        self.sprev = s[len(inputs)-1]

        return loss
    
    def sample(self, seed, n):
        h = self.hprev
        sp = self.sprev        
        x = np.zeros((self.insize, 1))
        x[seed] = 1
        ixes = []

        for t in range(n):            
            xh = np.hstack((x.ravel(), h.ravel())).reshape(self.insize+self.hidsize,1)
            f  = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            inp = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            cc = np.tanh(np.dot(self.Wc, xh) + self.bc)
            sp = f * sp + inp * cc
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            h = o * np.tanh(sp)  
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.insize), p=p.ravel())
            x = np.zeros((self.insize, 1))
            x[ix] = 1
            ixes.append(ix)

        return ixes
    
    def getHidden(self, xin):
        h = np.zeros_like(self.hprev)
        sp = np.zeros_like(self.sprev)
        for t in range(len(xin)): 
            x = np.zeros((self.insize, 1))
            x[xin[t]] = 1
            xh = np.hstack((x.ravel(), h.ravel())).reshape(self.insize+self.hidsize,1)
            f  = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            inp = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            cc = np.tanh(np.dot(self.Wc, xh) + self.bc)
            sp = f * sp + inp * cc
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            h = o * np.tanh(sp)             
        return h, sp
        
    def translate(self, num):
        h = self.hprev
        sp = self.sprev
        x = np.zeros((self.insize,1))
        y = np.zeros((self.insize,1))
        ixes = []
        count = 0
        while(True):
            xh = np.hstack((x.ravel(), h.ravel())).reshape(self.insize+self.hidsize,1)
            f  = self.sigmoid(np.dot(self.Wf, xh) + self.bf)
            inp = self.sigmoid(np.dot(self.Wi, xh) + self.bi)
            cc = np.tanh(np.dot(self.Wc, xh) + self.bc)
            sp = f * sp + inp * cc
            o = self.sigmoid(np.dot(self.Wo, xh) + self.bo)
            h = o * np.tanh(sp)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            #ix = np.random.choice(range(self.insize), p=p.ravel())
            ix = p.argmax()
            x = np.zeros((self.insize, 1))
            x[ix] = 1
            ixes.append(ix)
            if(num==ix):
                break
            count = count + 1
            if(count==20):
                break

        return ixes
            
            
def test():
    #read german text file
    data = open('de-json.txt', 'r').read()
    words = list(set(data.replace("\n", " <eos> ").split(" ")))
    data = data.replace("\n", " <eos>\n").split("\n")
    data_size, vocab_size = len(data), len(words)
    print('data has {} sentences, {} unique words.'.format(data_size, vocab_size))

    #dictionary for encoding and decoding from 1-of-k
    char_to_ix = { w:i for i,w in enumerate(words) }
    
    #Encoder
    lstm = LSTM(len(words), len(words), 100, 0.1, False)
    
    #read english text file
    data2 = open('en-json.txt', 'r').read()
    words2 = list(set(data2.replace("\n", " <eos> ").split(" ")))
    data2 = data2.replace("\n", " <eos>\n").split("\n")
    data_size2, vocab_size2 = len(data2), len(words2)
    print('data has {} sentences, {} unique words.'.format(data_size2, vocab_size2))

    #dictionary for encoding and decoding from 1-of-k
    char_to_ix2 = { w:i for i,w in enumerate(words2) }
    ix_to_char2 = { i:w for i,w in enumerate(words2) }

    #Decoder
    lstm2 = LSTM(len(words2), len(words2), 100, 0.1, True)

    losses = []
    losses2 = []
    
    n=0
    p=0
    
    while(True):
        if p >= len(data): 
            p = 0 # reset after every epoch
            
        sentence = data[p]
        wordsArray = sentence.split()
        x = [char_to_ix[w] for w in wordsArray[:-1]]        
        y = [char_to_ix[w] for w in wordsArray[1:]]
        loss = lstm.train(x, y)
        
        lstm2.hprev = lstm.hprev
        lstm2.sprev = lstm.sprev
        
        sentence = data2[p]
        wordsArray = ["is"] + sentence.split()
        x = [char_to_ix2[w] for w in wordsArray[:-1]] 
        wordsArray = sentence.split()
        y = [char_to_ix2[w] for w in wordsArray]
        loss2 = lstm2.train(x, y)
        
        if n%50==0:
            print('\n')
            print('Iteration: ', n)
            print('Loss Encoder: ', loss)
            print('Loss Decoder: ', loss2)
            losses.append(loss)
            losses2.append(loss2)

        if n%100==0:
            print('Translate: German to English')
            test = "ich habe ein buch dexp <eos>"
            print('German----> ', test)
            testArray = test.split()
            x = [char_to_ix[w] for w in testArray[:-1]]
            y = [char_to_ix[w] for w in testArray[1:]]
            htest, stest = lstm.getHidden(x)
            lstm2.hprev = htest 
            lstm2.sprev = stest
            num = char_to_ix2['<eos>'.strip()]
            oTest = lstm2.translate(num)
            txt = ' '.join(ix_to_char2[ix] for ix in oTest)
            print('English---> {} \n'.format(txt))
            
        lstm.hprev = np.zeros((100,1))
        lstm.sprev = np.zeros((100,1))

        if(n%100000==0):
            name1 = 'pickles/lstm1-' + str(n) + '.pickle'
            with open(name1, 'wb') as handle:
                pickle.dump(lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)
            name2 = 'pickles/lstm2-' + str(n) + '.pickle'
            with open(name2, 'wb') as handle:
                pickle.dump(lstm2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            name3 = 'pickles/encoderloss-' + str(n) + '.pickle'
            with open(name3, 'wb') as handle:
                pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
            name4 = 'pickles/decoderloss-' + str(n) + '.pickle'
            with open(name4, 'wb') as handle:
                pickle.dump(losses2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
        p += 1
        n += 1 

    plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
    plt.xlabel('time in thousands of iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
    

  
