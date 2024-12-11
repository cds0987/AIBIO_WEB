import pandas as pd
import torch
class SmilesSearching():
  def __init__(self):
    csv='AI_model_cache/Property Searching/bbbp.csv'
    df=pd.read_csv(csv)
    smiles=df['smiles']
    labels=df['target']
    BBBPindices1=[i for i in range(len(labels)) if labels[i]==1]
    BBBPindices0=[i for i in range(len(labels)) if labels[i]==0]
    BBBPsmiles1=[smiles[i] for i in BBBPindices1]
    BBBPsmiles0=[smiles[i] for i in BBBPindices0]
    self.BBBPsmiles1=BBBPsmiles1
    self.BBBPsmiles0=BBBPsmiles0
    #bbbp_ebd1=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Property Searching/bbbp_embeddings1.pt')
    self.bbbp_ebd1=torch.load('AI_model_cache/Property Searching/bbbp_embeddings1.pt',weights_only=True)
    self.mean_bbbp_ebd1=torch.mean(self.bbbp_ebd1,dim=0)
    self.bbbp_ebd0=torch.load('AI_model_cache/Property Searching/bbbp_embeddings0.pt',weights_only=True)
    self.mean_bbbp_ebd0=torch.mean(self.bbbp_ebd0,dim=0)
    csv='AI_model_cache/Property Searching/bace.csv'
    df=pd.read_csv(csv)
    smiles=df['smiles']
    labels=df['target']
    Baceindices1=[i for i in range(len(labels)) if labels[i]==1]
    Baceindices0=[i for i in range(len(labels)) if labels[i]==0]
    Bacesmiles1=[smiles[i] for i in Baceindices1]
    Bacesmiles0=[smiles[i] for i in Baceindices0]
    self.Bacesmiles1=Bacesmiles1
    self.Bacesmiles0=Bacesmiles0
    self.bace_ebd1=torch.load('AI_model_cache/Property Searching/bace_embeddings1.pt',weights_only=True)
    self.mean_bace_ebd1=torch.mean(self.bace_ebd1,dim=0)
    self.bace_ebd0=torch.load('AI_model_cache/Property Searching/bace_embeddings0.pt',weights_only=True)
    self.mean_bace_ebd0=torch.mean(self.bace_ebd0,dim=0)
    csv='AI_model_cache/Property Searching/fda.csv'
    df=pd.read_csv(csv)
    smiles=df['smiles']
    labels=df['target']
    FDAindices1=[i for i in range(len(labels)) if labels[i]==1]
    FDAindices0=[i for i in range(len(labels)) if labels[i]==0]
    FDAsmiles1=[smiles[i] for i in FDAindices1]
    FDAsmiles0=[smiles[i] for i in FDAindices0]
    self.FDAsmiles1=FDAsmiles1
    self.FDAsmiles0=FDAsmiles0
    self.fda_ebd1=torch.load('AI_model_cache/Property Searching/fda_embeddings1.pt',weights_only=True)
    self.mean_fda_ebd1=torch.mean(self.fda_ebd1,dim=0)
    self.fda_ebd0=torch.load('AI_model_cache/Property Searching/fda_embeddings0.pt',weights_only=True)
    self.mean_fda_ebd0=torch.mean(self.fda_ebd0,dim=0)
    csv='AI_model_cache/Property Searching/ctox.csv'
    df=pd.read_csv(csv)
    smiles=df['smiles']
    labels=df['target']
    CTOXindices1=[i for i in range(len(labels)) if labels[i]==1]
    CTOXindices0=[i for i in range(len(labels)) if labels[i]==0]
    CTOXsmiles1=[smiles[i] for i in CTOXindices1]
    CTOXsmiles0=[smiles[i] for i in CTOXindices0]
    self.CTOXsmiles1=CTOXsmiles1
    self.CTOXsmiles0=CTOXsmiles0
    self.ctox_ebd1=torch.load('AI_model_cache/Property Searching/ctox_embeddings1.pt',weights_only=True)
    self.mean_ctox_ebd1=torch.mean(self.ctox_ebd1,dim=0)
    self.ctox_ebd0=torch.load('AI_model_cache/Property Searching/ctox_embeddings0.pt',weights_only=True)
    self.mean_ctox_ebd0=torch.mean(self.ctox_ebd0,dim=0)
    csv='AI_model_cache/Property Searching/hiv.csv'
    df=pd.read_csv(csv)
    smiles=df['smiles']
    labels=df['target']
    HIVindices1=[i for i in range(len(labels)) if labels[i]==1]
    HIVindices0=[i for i in range(len(labels)) if labels[i]==0]
    HIVsmiles1=[smiles[i] for i in HIVindices1]
    HIVsmiles0=[smiles[i] for i in HIVindices0]
    self.HIVsmiles1=HIVsmiles1
    self.HIVsmiles0=HIVsmiles0[:600]
    self.hiv_ebd1=torch.load('AI_model_cache/Property Searching/hiv_embeddings1.pt',weights_only=True)
    self.hiv_ebd0=torch.load('AI_model_cache/Property Searching/hiv_embeddings0.pt',weights_only=True)
    self.mean_hiv_ebd1=torch.mean(self.hiv_ebd1,dim=0)
    self.mean_hiv_ebd0=torch.mean(self.hiv_ebd0,dim=0)
  def find_smiles(self, property, k):
    # Select the initial embedding based on 'bbbp'
    bbbp_using = self.bbbp_ebd1 if property['bbbp'] == 1 else self.bbbp_ebd0

    # Define a helper function to compute similarity and perform top-k
    def compute_topk(embeddings, mean_embedding, top_k):
        similarity = torch.cosine_similarity(embeddings, mean_embedding.unsqueeze(0))
        topk_result = torch.topk(similarity, top_k)
        return embeddings[topk_result.indices], topk_result.indices.tolist()

    # Compute top-k for 'bace'
    mean_bace = self.mean_bace_ebd1 if property['bace'] == 1 else self.mean_bace_ebd0
    bbbp_using, indices = compute_topk(bbbp_using, mean_bace, 300)
    smiles_bbbp = [self.BBBPsmiles1[i] for i in indices]

    # Compute top-k for 'fda'
    mean_fda = self.mean_fda_ebd1 if property['fda'] == 1 else self.mean_fda_ebd0
    bbbp_using, indices = compute_topk(bbbp_using, mean_fda, 200)
    smiles_bbbp = [smiles_bbbp[i] for i in indices]

    # Compute top-k for 'ctox'
    mean_ctox = self.mean_ctox_ebd1 if property['ctox'] == 1 else self.mean_ctox_ebd0
    bbbp_using, indices = compute_topk(bbbp_using, mean_ctox, 150)
    smiles_bbbp = [smiles_bbbp[i] for i in indices]

    # Compute top-k for 'hiv'
    mean_hiv = self.mean_hiv_ebd1 if property['hiv'] == 1 else self.mean_hiv_ebd0
    bbbp_using, indices = compute_topk(bbbp_using, mean_hiv, 100)
    smiles_bbbp = [smiles_bbbp[i] for i in indices]

    # Compute final top-k for 'bace' again (confirm corrected logic here)
    mean_bace_final = self.mean_bace_ebd1 if property['bace'] == 1 else self.mean_bace_ebd0
    bbbp_using, indices = compute_topk(bbbp_using, mean_bace_final, k)
    smiles_bbbp = [smiles_bbbp[i] for i in indices]

    return smiles_bbbp

