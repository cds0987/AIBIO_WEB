import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import os


class GenerateMOLT5(nn.Module):
    def __init__(self):
        super(GenerateMOLT5, self).__init__()
        self.smiles_model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-small-caption2smiles')
        self.smiles_tokenizer=T5Tokenizer.from_pretrained("laituan245/molt5-small-caption2smiles", model_max_length=512)
        self.device=torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.smiles_model.to(self.device)
    def smile_proccess(self,caption):
        input_ids = self.smiles_tokenizer(caption, return_tensors="pt").input_ids
        input_ids=input_ids.to(self.device)
        outputs = self.smiles_model.generate(input_ids, num_beams=4, max_length=64)
        return self.smiles_tokenizer.decode(outputs[0], skip_special_tokens=True)
    def forward(self,inputs):
        return self.smile_proccess(inputs)
    
