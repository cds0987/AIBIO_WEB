from transformers import AutoTokenizer, AutoModel, AdamW, AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2LMHeadModel
import torch
import pandas as pd
class search_model():
  def __init__(self):
    self.list_data=['bbbp','bace','ctox','fda','ctox','freesolv','lipophilicity']

    self.base_model=AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True,cache_dir="AI_model_cache/HuggingFace")
    self.tokenizer=AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True,cache_dir="AI_model_cache/HuggingFace")
    for param in self.base_model.parameters():
        param.requires_grad = False
    self.bbbp=torch.load('AI_model_cache/Smiles Searching/bbbp.pt',weights_only=True)
    self.bace=torch.load('AI_model_cache/Smiles Searching/bace.pt',weights_only=True)
    self.ctox=torch.load('AI_model_cache/Smiles Searching/ctox.pt',weights_only=True)
    self.fda=torch.load('AI_model_cache/Smiles Searching/fda.pt',weights_only=True)
    self.freesolv=torch.load('AI_model_cache/Smiles Searching/freesolv.pt',weights_only=True)
    self.lipophilicity=torch.load('AI_model_cache/Smiles Searching/lipophilicity.pt',weights_only=True)


    #smiles
    self.smiles_bbbp=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/bbbp.csv')
    self.smiles_bbbp=self.smiles_bbbp['smiles'].tolist()
    self.smiles_bace=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/bace.csv')
    self.smiles_bace=self.smiles_bace['smiles'].tolist()
    self.smiles_ctox=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/ctox.csv')
    self.smiles_ctox=self.smiles_ctox['smiles'].tolist()
    self.smiles_fda=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/fda.csv')
    self.smiles_fda=self.smiles_fda['smiles'].tolist()
    self.smiles_freesolv=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/freesolv.csv')
    self.smiles_freesolv=self.smiles_freesolv['smiles'].tolist()
    self.smiles_lipophilicity=pd.read_csv(f'AI_model_cache/Smiles Searching/Csv/lipophilicity.csv')
    self.smiles_lipophilicity=self.smiles_lipophilicity['smiles'].tolist()
  def searching(self,smile,type,k):
    inputs = self.tokenizer(smile, return_tensors="pt", padding=True, truncation=True)
    ebd=self.base_model(**inputs)
    ebd=ebd.pooler_output
    ebd=ebd.squeeze(0)
    if type=='bbbp':
      similarity=torch.cosine_similarity(ebd,self.bbbp)
      topk_result  = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_bbbp[i] for i in indices]
    elif type=='bace':
      similarity=torch.cosine_similarity(ebd,self.bace)
      topk_result = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_bace[i] for i in indices]
    elif type=='ctox':
      similarity=torch.cosine_similarity(ebd,self.ctox)
      topk_result = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_ctox[i] for i in indices]
    elif type=='fda':
      similarity=torch.cosine_similarity(ebd,self.fda)
      topk_result = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_fda[i] for i in indices]
    elif type=='freesolv':
      similarity=torch.cosine_similarity(ebd,self.freesolv)
      topk_result = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_freesolv[i] for i in indices]
    elif type=='lipophilicity':
      similarity=torch.cosine_similarity(ebd,self.lipophilicity)
      topk_result = torch.topk(similarity, k)
      indices = topk_result .indices.tolist()
      smiles = [self.smiles_lipophilicity[i] for i in indices]
    return smiles
  def search(self,smile,k):
    result={}
    for data in self.list_data:
       result[data]=self.searching(smile,data,k)
    return result

