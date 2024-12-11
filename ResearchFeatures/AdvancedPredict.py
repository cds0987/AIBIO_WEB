import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_Model.PropertyPredictionGraphModel import Graph_models1
import torch
from rdkit import Chem
class Adprediction():
    def __init__(self):
        self.prediction=Graph_models1()
        self.device=torch.device('cpu')
 
    def predict(self,smiles):
        bbbp=self.prediction(smiles,'bbbp')
        bace=self.prediction(smiles,'bace')
        ctox=self.prediction(smiles,'ctox')
        fda=self.prediction(smiles,'fda')
        hiv=self.prediction(smiles,'hiv')
        esol=self.prediction(smiles,'esol')
        freesolv=self.prediction(smiles,'freeSolv')
        lipophilicity=self.prediction(smiles,'lipophilicity')
        qm7=self.prediction(smiles,'qm7')
        E1CC2=self.prediction(smiles,'E1CC2')
        E2CC2=self.prediction(smiles,'E2CC2')
        E1PBE0=self.prediction(smiles,'E1PBE0')
        E2PBE0=self.prediction(smiles,'E2PBE0')
        E1CAM=self.prediction(smiles,'E1CAM')
        E2CAM=self.prediction(smiles,'E2CAM')
        return bbbp,bace,ctox,fda,hiv,esol,freesolv,lipophilicity,qm7,E1CC2,E2CC2,E1PBE0,E2PBE0,E1CAM,E2CAM