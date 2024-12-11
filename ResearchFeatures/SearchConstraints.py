import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.SearchProperty import SmilesSearching
import torch
class SearchConstraintsEngine:
    def __init__(self):
        self.model=SmilesSearching()
    def search(self,property,k):
        result=self.model.find_smiles(property,k)
        return result
   
