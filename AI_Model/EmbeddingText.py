from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os

class MiniSentenceTransformer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2'):
        """
        Initialize the MiniSentenceTransformer with a pre-trained model.

        Args:
            model_name (str): The name of the pre-trained model from HuggingFace Hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling on the token embeddings, considering the attention mask.

        Args:
            model_output: Output from the transformer model.
            attention_mask: Attention mask for the input tokens.

        Returns:
            Tensor: Pooled sentence embeddings.
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        """
        Compute sentence embeddings for the given sentences.

        Args:
            sentences (list of str): Sentences to compute embeddings for.

        Returns:
            Tensor: Normalized sentence embeddings.
        """
        # Tokenize the sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    
