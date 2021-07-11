from transformers import BertTokenizer, BertModel
import torch
import os
from pathlib import Path
import urllib.request
from zipfile import ZipFile

class BertBase:

    DANLP_STORAGE_URL = 'http://danlp-downloads.alexandra.dk'
    REMOTE_MODEL_PATH = 'models/bert.botxo.pytorch.zip'
    ABSOLUT_URI = DANLP_STORAGE_URL + '/' + REMOTE_MODEL_PATH
    DEFAULT_DOWNLOAD_PATH = Path.home() / "bert" / "bert.zip"

    def __init__(self, model_path):

        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(
            self.model_path,
            output_hidden_states = True,
            ) 

        self.model.eval()

    @classmethod
    def download_model(cls, path: Path = DEFAULT_DOWNLOAD_PATH):
        if not path.parent.exists():
            os.mkdir(path.parent)
        urllib.request.urlretrieve(cls.ABSOLUT_URI, path.as_posix())
        return cls.unzip_file(path)

    @staticmethod
    def unzip_file(file_path):
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path.parent)
        return os.remove(file_path)

    def embed_text(self, text):
        """
        Calculate the embeddings for each token in a sentence ant the embedding for the sentence based on a BERT language model.
        The embedding for a token is chosen to be the concatenated last four layers, and the sentence embeddings to be the mean of the second to last layer of all tokens in the sentence
        The BERT tokenizer splits in subword for UNK word. The tokenized sentence is therefore returned as well. The embeddings for the special tokens are not returned.
        :param str sentence: raw text
        :return: three lists: token_embeddings (dim: tokens x 3072), sentence_embedding (1x738), tokenized_text
        :rtype: list, list, list
        """
        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize sentence with the BERT tokenizer
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # Map the token strings to their vocabulary indeces
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Mark each of the tokens as belonging to sentence "1"
        segments_ids = [1] * len(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1. to tokens x layers x embedding
        token_embeddings = token_embeddings.permute(1,0,2)
        # choose to concatenate last four layers, dim 4x 768 = 3072
        token_vecs_cat= [torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) for token in token_embeddings]
        # drop the CLS and the SEP tokens and embedding
        token_vecs_cat=token_vecs_cat[1:-1]
        tokenized_text =tokenized_text[1:-1]
        # chose to summarize the last four layers
        #token_vecs_sum=[torch.sum(token[-4:], dim=0) for token in token_embeddings]
        # sentence embedding
        # Calculate the average of all token vectors for the second last layers
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)
        return token_vecs_cat, sentence_embedding, tokenized_text