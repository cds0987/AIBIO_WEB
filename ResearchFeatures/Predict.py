import sys
import os

# Add the parent directory of 'Free' to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.ChemMTR import CHMTR 
from AI_Model.ChemMLM import CHMLM
from transformers import logging
import warnings
import torch
# Set logging to error only
logging.set_verbosity_error()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import Chem

class prediction:
    def __init__(self):
        self.model1=CHMLM()
        self.model2=CHMTR()
        self.model1.eval()
        self.model2.eval()

    def predict_m1(self,smiles):
        bbbp=self.model1(smiles,'bbbp').detach().numpy()[0][0]
        bace=self.model1(smiles,'bace').detach().numpy()[0][0]
        ctox=self.model1(smiles,'ctox').detach().numpy()[0][0]
        fda=self.model1(smiles,'fda').detach().numpy()[0][0]
        esol=self.model1(smiles,'esol').detach().numpy()[0][0]
        freesolv=self.model1(smiles,'freesolv').detach().numpy()[0][0]
        lipophilicity=self.model1(smiles,'lipophilicity').detach().numpy()[0][0]
        qm7=self.model1(smiles,'qm7').detach().numpy()[0][0]
        return bbbp,bace,ctox,fda,esol,freesolv,lipophilicity,qm7
    def predict_m2(self,smiles,mode_preimium=False):
        bbbp=self.model2(smiles,'bbbp').detach().numpy()[0][0]
        bace=self.model2(smiles,'bace').detach().numpy()[0][0]
        ctox=self.model2(smiles,'ctox').detach().numpy()[0][0]
        fda=self.model2(smiles,'fda').detach().numpy()[0][0]
        esol=self.model2(smiles,'esol').detach().numpy()[0][0]
        freesolv=self.model2(smiles,'freesolv').detach().numpy()[0][0]
        lipophilicity=self.model2(smiles,'lipophilicity').detach().numpy()[0][0]
        qm7=self.model2(smiles,'qm7').detach().numpy()[0][0]
        return bbbp,bace,ctox,fda,esol,freesolv,lipophilicity,qm7
