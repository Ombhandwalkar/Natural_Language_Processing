* So we take an LLM. 
    Now we have the data, so first of all, we convert the data into int8, then perform the matmul with weight (which is already converted into int8),
*  then we convert it into int32, and then we perform addition with bias (which is in int32 format). 
* Then we need to perform the activation function operation; we can not perform this in int. 
* This will affect the accuracy and minimise the precision score of the model. So we dequantise our int32 and convert it into float32, 
* And now we perform the activation function . Again, do we quantise it for the next hidden state ?