import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.GenerateSmiles import GenerateMOLT5
class Generate:
    def __init__(self):
        self.model = GenerateMOLT5()
    def generate(self,captions):
        return self.model(captions)
    