import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.SmilesSearch import search_model
import torch
class search_engine:
    def __init__(self):
        self.model=search_model()
    def search(self,smiles,k):
        result=self.model.search(smiles,k)
        return result
    

