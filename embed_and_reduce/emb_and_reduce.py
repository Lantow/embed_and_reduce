from danlp.models import load_bert_base_model
import numpy as np
import nltk
import re


class ReduceDim(object):
    
    def __init__(self, embedding):
        self.embedding = embedding
    
    def reduce_dim(self):
        with open("pca.pkl", "rb") as f:
            pca = pickle.load(f)
        self.emb100 = pca.transform(self.embedding.reshape(1,-1))[0]


class CleanAndEmb(object):
    
    def __init__(self, search_string):
        self.search_string = search_string

    def load_tokenizer(self):
        try:
            self.tokenizer = nltk.data.load("tokenizers/punkt/danish.pickle")
        except LookupError:
            nltk.download('punkt')
            self.tokenizer = nltk.data.load("tokenizers/punkt/danish.pickle")
        except Exception as E:
            raise E

    def tokenize_raw_text_data(self):
        if not hasattr(self, "tokenizer"): self.load_tokenizer()
        try:
            self.tokenized_str = self.tokenizer.tokenize(self.search_string)
        except Exception as E:
            raise E

    def clean_sents(self):
        # For now we only remove \r and \n, as we might remove context useable by the BERT model
        pp_special_chars = lambda sent: re.sub("\s+", " ", re.sub("\r|\n|\t", "", sent))
        #for now, there is no preprocessing on the numbers
        pp_numbers = lambda sent: sent
        preprocess_sent = lambda sent: pp_special_chars(pp_numbers(sent))
        preprocess_full_txt = lambda full_txt: [preprocess_sent(sent) for sent in full_txt]
        #Do something with the paragraph signs and the numbers etc.
        try:
            self.cleaned_tokenized_str = preprocess_full_txt(self.tokenized_str)
        except Exception as E:
            raise E

    def embed_text(self):
        model = load_bert_base_model()
        def embed_sent(sent):
            # the DAnlp bert base model cant handle more than X tokens
            # the following try/except solves the symptom but not the root cause.
            # OBS! when root cause is solved the folowing code may be used instead:              
            # embed_sent = lambda sent: model.embed_text(sent)
            try:
                return model.embed_text(sent)[1].numpy()
            except Exception as E:
                if "The size of tensor a" and "must match the size of tensor b" in str(E):
                    # (n is an arbitrary low number. OBS! This will still fail, if each char becomes a token)
                    # TODO: this partially be solved with a better word tokenizer  
                    n= 512
                    # Here the long sentance is split and embedded, whereafter an average is returned
                    split_sent_on_idx = lambda i: model.embed_text(sent[i:i+n])[1].numpy()
                    split_points = range(0, len(sent), n)
                    split_sent_obj = map(split_sent_on_idx, split_points)
                    split_sent_np = np.array(list(split_sent_obj)).mean(axis=0)
                    #A PyTorch tensor is returned to type match the rest of the sents
                    return split_sent_np
                else: 
                    raise E

        try:
            mean = lambda l: np.array(l).mean(axis=0)
            self.embedding = mean([embed_sent(sent) for sent in self.cleaned_tokenized_str])
        except Exception as E:
            raise E
    
    def clean_and_embed(self):
        self.tokenize_raw_text_data()
        self.clean_sents()
        self.embed_text()
                

class EmbAndReduce(CleanAndEmb, ReduceDim):
    pass
    
    
    
    
    
    

# CAE = CleanAndEmb(""" De beløb, der refunderes af udligningsordningen, omfatter momsbetalinger fra 
#                   (amts)kommunerne og Hovedstadens Sygehusfællesskab i forbindelse med køb af varer 
#                   og tjenesteydelser, der bogføres på hovedkontiene 0-6, bortset fra momsbetalinger, 
#                   der kan fradrages som indgående moms i et momsregnskab. Stk. 2. Beløbene efter stk. 
#                   1 bestemmes som summen af følgende udgifter til moms (1+2): """)
# CAE.clean_and_embed()
# CAE.embeded_sents
# CAE.cleaned_tokenized_str