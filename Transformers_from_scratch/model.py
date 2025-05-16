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
        x= x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # We specifically telling our model- It is not Trainable parameter . DO not train it
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
    

class ResidualConnection(nn.Module):
    """" Residual connections help stabilize trainin in deep neural networds by allowing information to bypass certain layers"""
    def __init__(self,features:int,dropout:float)->None: # Features- dimensions of the input vector
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization(features)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model # Dimension of the input embedding vector
        self.h=h # Number of attention heads
        assert d_model % h ==0, "d_model is not divisible by h" # Ensures that d_model is divisible by h to avoid dimension mismatch.
        self.d_k=d_model // h # Dimension of each head(d_k)

        self.w_q=nn.Linear(d_model,d_model,bias=False) # Weight matrix for Query(Wq)
        self.w_k=nn.Linear(d_model,d_model,bias=False) # Weight matrix for Key (Wk)
        self.w_v=nn.Linear(d_model,d_model,bias=False) # Weight matrix for Value (Wv)
        self.w_o=nn.Linear(d_model,d_model,bias=False) # Output weight (Wo) This concat the output of all heads back to the original d_model size.
        self.dropout=nn.Dropout(dropout)               # To prevent overfitting
    
    @staticmethod 
    def attention(query,key,value,mask,dropout:nn.Dropout):
        # Just apply the formula from the paper
        d_k=query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores=(query @ key.transpose(-2,-1)/math.sqrt(d_k)) # computes the dot product between Q and K^T [transpose]
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            # Masks are used to prevent attention on certain positons like padding tokens.
            attention_scores.masked_fill(mask==0,-1e9) # The -1e9 ensures that the softmax value for masked position is almost zero.
        attention_scores =attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax converts attention score into probabilities.
        if dropout is not None:
            attention_scores = dropout(attention_scores)   # Regularize teh attention matrix
            # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
            # return attention scores which can be used for visualization
        return (attention_scores @ value),attention_scores  # Returns weighted value matrix.
    
    def forward(self,q,k,v,mask):
        query=self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key  =self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value=self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Splitting for multi-Head Attention
         # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key  =  key.view(key.shape[0]  ,  key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        # Calculate Attention
        x,self.attention_score=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # Combining the Attention heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x) # concatenated output of all heads back to the original d_model size.
    

class EncoderBlock(nn.Module):
    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(features,dropout)for _ in range(2)])

    def forward(self,x,src_mask):
        x=self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask)) # Self attention + residual connection.
        x=self.residual_connection[1](x,self.feed_forward_block) # Feedforward network + residual connection.
        return x  

class Encoder(nn.Module):
    ''' The class is a stack of multiple EncoderBlock layers'''
    def __init__(self,features:int,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):
    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(features,dropout)for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self,features:int,layers:nn.ModuleList)-> None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """ This allows the model to produce predictions in the form of token probabilites"""
    def __init__(self,d_model,vocab_size)->None:
        super().__init__()
        self.proj =nn.Linear(d_model,vocab_size)
    
    def forward(self,x)->None:
         # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
         return self.proj(x)


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding,projection_layer:ProjectionLayer)-> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
    
    def encode(self,src,src_mask):
         # (batch, seq_len, d_model)
         src=self.src_embed(src)
         src=self.src_pos(src)
         return self.encoder(src,src_mask)
    
    def decode(self,encoder_output:torch.Tensor,src_mask:torch.Tensor,tgt:torch.Tensor,tgt_mask:torch.Tensor):
          # (batch, seq_len, d_model)
          tgt=self.tgt_embed(tgt)
          tgt=self.tgt_pos(tgt)
          return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
          # (batch, seq_len, d_model)
          return self.projection_layer(x)
     


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters using Xavier for better training.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer