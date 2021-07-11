import numpy as np
import nltk
import re
import pickle
from pathlib import Path
import os
from bert import BertBase

class ReduceDim(object):

    def reduce_dim(self, embedding):
        return self.pca.transform(embedding.reshape(1,-1))[0]


class CleanAndEmb(object):

    def __init__(self, pca_path, model_path):
        self.model = BertBase(model_path)
        self.tokenizer = self.load_tokenizer()
        self.pca = self._load_pca_pickle(pca_path)
        
    def _load_pca_pickle(self, pca_path):
        with open(pca_path, "rb") as f:
            return pickle.load(f)

    def load_tokenizer(self):
        try:
            return nltk.data.load("tokenizers/punkt/danish.pickle")
        except LookupError:
            nltk.download('punkt')
            return nltk.data.load("tokenizers/punkt/danish.pickle")
        except Exception as E:
            raise E

    def clean_and_embed(self, search_string: str):
        tokens = self.tokenize_raw_text_data(search_string)
        clean_sentences = self.clean_sents(tokens)
        return self.embed_text(clean_sentences)

    def tokenize_raw_text_data(self, search_string: str):
        if not hasattr(self, "tokenizer"): 
            self.tokenizer = self.load_tokenizer()
        try:
            return self.tokenizer.tokenize(search_string)
        except Exception as E:
            raise E

    def clean_sents(self, tokenized_str: list):
        # For now we only remove \r and \n, as we might remove context useable by the BERT model
        pp_special_chars = lambda sent: re.sub("\s+", " ", re.sub("\r|\n|\t", "", sent))
        #for now, there is no preprocessing on the numbers
        pp_numbers = lambda sent: sent
        preprocess_sent = lambda sent: pp_special_chars(pp_numbers(sent))
        preprocess_full_txt = lambda full_txt: [preprocess_sent(sent) for sent in full_txt]
        #Do something with the paragraph signs and the numbers etc.
        try:
            return preprocess_full_txt(tokenized_str)
        except Exception as E:
            raise E

    def embed_text(self, clean_sentences: list):
        try:
            mean = lambda l: np.array(l).mean(axis=0)
            return mean([self.embed_sent(sent) for sent in clean_sentences])
        except Exception as E:
            raise E

    def embed_sent(self, sent):
        # the DAnlp bert base model cant handle more than X tokens
        # the following try/except solves the symptom but not the root cause.
        # OBS! when root cause is solved the folowing code may be used instead:              
        # embed_sent = lambda sent: model.embed_text(sent)
        try:
            return self.model.embed_text(sent)[1].numpy()
        except Exception as E:
            if "The size of tensor a" and "must match the size of tensor b" in str(E):
                # (n is an arbitrary low number. OBS! This will still fail, if each char becomes a token)
                # TODO: this partially be solved with a better word tokenizer  
                n= 512
                # Here the long sentance is split and embedded, whereafter an average is returned
                split_sent_on_idx = lambda i: self.model.embed_text(sent[i:i+n])[1].numpy()
                split_points = range(0, len(sent), n)
                split_sent_obj = map(split_sent_on_idx, split_points)
                split_sent_np = np.array(list(split_sent_obj)).mean(axis=0)
                # A PyTorch tensor is returned to type match the rest of the sents
                return split_sent_np
            else: 
                raise E
    
class EmbAndReduce(CleanAndEmb, ReduceDim):

    def embed_and_reduce(self, search_string):
        embedding = self.clean_and_embed(search_string)
        return self.reduce_dim(embedding)
    
if __name__ == '__main__':
    search_string = """ De beløb, der refunderes af udligningsordningen, omfatter momsbetalinger fra 
                    (amts)kommunerne og Hovedstadens Sygehusfællesskab i forbindelse med køb af varer 
                    og tjenesteydelser, der bogføres på hovedkontiene 0-6, bortset fra momsbetalinger, 
                    der kan fradrages som indgående moms i et momsregnskab. Stk. 2. Beløbene efter stk. 
                    1 bestemmes som summen af følgende udgifter til moms (1+2): """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    instance = EmbAndReduce(Path(ROOT_DIR).parent / "pca.pkl", model_path=Path.home() / "bert")
    embedding_dim100 = instance.embed_and_reduce(search_string)
    print(embedding_dim100)