import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.CaptionSmiles import CaptionMOLT5
import torch
class CaptionSmiles:
    def __init__(self):
        self.model = CaptionMOLT5()

    def caption(self, smiles):
        caption = self.model(smiles)
        return caption
    
