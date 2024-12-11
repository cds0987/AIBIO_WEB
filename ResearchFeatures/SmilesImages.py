import sys
from PIL import Image
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.SmilesImage import Smiles2Image
import matplotlib.pyplot as plt
class Images_Smiles:
    def __init__(self):
        self.smiles2Image = Smiles2Image()
    def Image2Smiles(self,Image):
        return self.smiles2Image.capImg(Image)


