'''We will convert sentence into input Embedding'''
# -> Input embeddings allows you to convert original sentence into a vector of 512 dimensions
"""
            ex- 
            * original sentence(tokens) -             YOUR      CAT       IS          LOVELY      CAT
            * Input IDs
                (It will convert tokens into          105       6587      5475        6854        8574
                token IDs accoding to
                the position of the token
                  in the vocabulary.)
            * Embeddings(Each of these numbers          
                    converted into embeddings of
                    512 vector)   
"""
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

        # We are poviding dimensions of the vector model and vocabulary size
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        # This module used to transform a set of integer indices into dense vector representation
        self.embedding=nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        # We will map embedding layer 
        return self.embedding(x) * math.sqrt(self.d_model)  # --> We will scale the embeddings by dimensions of model.
        # In paper the authors multiplied embeddings by square root of model dimensions  --> This prevent unstable gradient training.
        # Increase in dimensions will also increase the variance. For small or big values, it will face vanishing or exploding gradient problem.
        #               ( for softmax if High variance- it will give higher probability)
        #               ( for softmax if Low  variance- it will give lower  probability)
        # It control the variance initialization.

''' We will add Positional Encoding in embeddings.'''
# Due to parallel operation in self attentaion - We had to use the Positional Encoding-

class PositionalEncoding(nn.Module):
    
    def __init__(self,d_model:int ,seq_len:int, dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # By using the Trignometric function(Formula), we will add positional encoding in the Embeddings
        # we will  use PE(positional of the model, 2i)=sin(POS/10000^2i/d_model) --> for even position
        # we will  use PE(positional of the model, 2i)=cos(POS/10000^2i/d_model) --> for odd  position

        # create matrix of shape (seq_len,d_model)
        pe=torch.zeros(seq_len,d_model)  # This matrix will store positonal encodings for each positions in seqence.
        # Create a vector of shape(seq_len,1)
        position= torch.arange(0, seq_len,dtype=torch.float).unsqueeze(1) # Generates a column vector of integers, Adds an extra dimension to make it shape of (seq_len,1)
        # The divisor ensures that the sinusoidal values have different frequencies,allowing them to repreesnt unique positions.
        div_term=torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) # This generates the scaling factors for the sin and cos values.
        # Apply sin to even position
        pe[:,0::2]=torch.sin(position * div_term)
        pe[:,1::2]=torch.cos(position * div_term)

        pe =pe.unsqueeze(0) # (1,seq_len,d_model)
        self.register_buffer('pe',pe) # Register this tensor as buffer [non-trainable parameter]

    
    def forward(self,x):
        x= x + (self.pe[:,x.shape[1],:]).requires_grad_(False) # We specifically telling our model- It is not Trainable parameter . DO not train it
        return self.dropout(x) # Helps to regularize the overall model training.
    

#  We are applying Normalization . To ensures stable training by standardizing activatioins.
#  We also introduce the  gamma and beta values to introduce fluctuations in the data. 
#  It allow the model to adaptively re-scale and shift the normalized values for optimal performance
class LayerNormalization(nn.Module):
    def __init__(self,eps:float =10**-6)-> None:
        super().__init__()
        self.eps=eps  # A small value added to the denominator to prevent division by zero.
        self.alpha=nn.Parameter(torch.ones(1)) # Multiplies --> Learnable parameter . Multiplies the normalized values,giving the model flexibility to re-scale the normalized output.
        self.bias=nn.Parameter(torch.zeros(1)) # Added --> Allows model to shift the normalized values.
     
    def forward(self,x):
        mean =x.mean(dim=-1,keepdim=True) # compute the mean of the input tensor x along with last dimension. keepdim=True --> Ensures the dimension is retrained
        std=x.std(dim=-1,keepdim=True)    # compute the standard deviation along the last dimension. 
        return self.alpha*(x-mean)/(std + self.eps)+ self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->float:
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model) # W2 and B2
    
    def forward(self,x):
        # (Batch, Seq_len, d_model)  --> (Batch, Seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
