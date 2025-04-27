from fastapi import FastAPI,Request
import uvicorn
import numpy as np
import torch 
from transformers import BertTokenizer,BertForSequenceClassification

app=FastAPI()


@app.get('/')
def read_root():
    return ('Hello')

@app.get("/hello")
def read_root():
    return {"Hello": "Hello"}

# Fetching our model from HuggingFace Repository.
def get_model():
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    model=BertForSequenceClassification.from_pretrained('OmBhandwalkar/YTFinetuneBert')
    return tokenizer,model

d={
    1:'Toxic',
    0:'Non Toxic'
}

# Getting our model from function
tokenizer,model=get_model()

# Tokenization Precess takes place orver here.
# Converting normal text to numerical format by using BertTokenizer.
@app.post('/predict')
async def read_root(request:Request):
    data=await request.json()
    print(data)
    if 'text' in data:
        user_input=data['text']
        test_sample=tokenizer([user_input],padding=True,truncation=True,max_lenght=512,return_tensors='pt')
        output=model(**test_sample)
        y_pred=np.argmax(output.logits.detach().numpy(),axis=1)
        response={'Received Text':user_input,'Predicton':d[y_pred[0]]}
    else:
        response={'Received Text':'No text Found'}
    return response

if __name__=='__main__':
    uvicorn.run('main:app',host='0.0.0.0',porst=8080,reload=True,debug=True)