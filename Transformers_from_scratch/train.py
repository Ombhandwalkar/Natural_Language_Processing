import torch
import torch.nn as nn
from torch.utils.data import  Dataset,DataLoader,random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

# Dataset and Language
def get_all_sentences(ds,lang):  
    for item in ds:
        yield  item['translation'][lang]

def get_of_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token=['UNK']))  # If unknow word found in the query just say UNKNOWN
        tokenizer.pre_tokenizer=Whitespace()  # Create the tokenizer according to spaces.
        trainer= WordLevelTrainer(special_tokens=['[UNK]','[PAD]','SOS','EOS'],min_fequency=2) # We are adding special tokens to the vocabulary.
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw=load_dataset('opus-books',f"{config['lang_src']}-{config['lang-tgt']}",split='train')

    # Build Tokenizer
    tokenizer_src=get_of_build_tokenizer(ds_raw,config,config['lang_src'])
    tokenizer_tgt=get_of_build_tokenizer(ds_raw,config,config['lang-tgt'])

    # Keep 90% of the data for training and 10% for validation
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])