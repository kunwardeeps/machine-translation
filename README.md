# LSTM-NMT
Vanilla Neural Machine Translation using LSTM

***FROM SCRATCH***

Features to add-
1. Using Bi-RNN/LSTM instead of LSTM
2. Attention Mechanism


Improvements: 

Bi directional RNN1: 
a.	After observing the above problem, I studied this iconic paper (Cho et al. [2014]) which discusses the implementation of bidirectional RNN with attention model.
b.	Bi-RNN will boost the translation accuracy as they can capture the context from both sides of the sentence. 

Attention Mechanism1: 
a.	To utilize the advantages of bi-directional RNN, we also need to implement the attention mechanism as explained by Cho et al in the paper. Attention allows the decoder to access all the hidden states of encoder at every timestep. This ensures for the decoder to use the context present in the various parts of a sentence.  
b.	This principle has a significant impact on neural computation as we can select the most pertinent piece of information, rather than using all available information, a large part of it being irrelevant to compute the neural response.

Word Embedding using word2vec2: 
a.	The current model is using one-to-k hot vector encoding as the input to our model. In this type of encoding, all the words are equidistant. Word2vec encoding incorporates the semantic significance of words. All the words are mapped to another space where all the similar words fall under same cluster.
b.	The length of one-hot vector is equal to the size of vocabulary. In this case the size was approx. 2500 words. Multiplying big vectors take lot of time to process the output, hence increasing the training time. Using word2vec can decrease the input vector size to as low as 50. Hence, decreasing training time.



