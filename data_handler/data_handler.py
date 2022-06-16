from data_handler.connection_handler import PostgresConnection
import numpy as np
import nltk
import re
from tqdm import tqdm 


class DataHandler(PostgresConnection):
    
    def load_data(self):
        print("loading data")
        exec_str = "SELECT id, document_text FROM public.scrapers_retsinfodocument WHERE document_emb_full IS NULL LIMIT 10;"
        self.curr.itersize = 5
        self.curr.execute(exec_str)
    
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
        
        self.doc_id = []
        self.sent_text = []
        def tknz_and_keep_id(curr_item):
            sent_text_in_doc = self.tokenizer.tokenize(curr_item[1])
            self.sent_text.append(sent_text_in_doc)
            self.doc_id.append(curr_item[0])
            return sent_text_in_doc
        
        try:
            self.tokenized_map = tqdm(map(tknz_and_keep_id, self.curr), 
                                       "Tokenizing tekst text into sentances")
        except Exception as E:
            raise E
        
    def clean_sents(self):
        # For now we only remove \r and \n, as we might remove context useable by the BERT model
        pp_special_chars = lambda sent: re.sub("\s+", " ", re.sub("\r|\n|\t", "", sent))
        #for now, there is no preprocessing on the numbers
        pp_numbers = lambda sent: sent
        preprocess_sent = lambda sent: pp_special_chars(pp_numbers(sent))
        preprocess_full_txt = lambda full_txt: map(preprocess_sent, full_txt)
        #Do something with the paragraph signs and the numbers etc.
        try:
            self.cleaned_tokenized_map= tqdm(map(preprocess_full_txt, self.tokenized_map),
                                               "cleaning sentances")
        except Exception as E:
            raise E

