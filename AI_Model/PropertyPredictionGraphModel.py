import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2LMHeadModel,AutoModelForMaskedLM
from torch_geometric.nn import DeepGCNLayer, GENConv,GCN2Conv,SAGEConv,GraphConv,AGNNConv,GATv2Conv,TransformerConv
device=torch.device('cpu')
import os

class GATV2_Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=48, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2):
        super(GATV2_Transformer, self).__init__()
        self.node_encoder=nn.Sequential(
                          nn.Linear(in_channels, 512),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(512, hidden_channels),
                          )
        self.pe_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleList([
                GATv2Conv(hidden_channels, hidden_channels, heads=heads,edge_dim =1, dropout=dropout,concat=True),
                nn.ReLU()
            ])
            self.pe_blocks.append(block.to(device))
        self.hiddens_transformer=hidden_channels
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hiddens_transformer, nhead=heads), num_layers=num_layers_transformers)
        self.graph_linear=nn.Linear(hidden_channels*heads, hidden_channels)
        self.classifier=nn.Sequential(
                         nn.Linear(self.hiddens_transformer, 2048),
                         nn.ReLU(),
                         nn.Dropout(0.6),
                         nn.Linear(2048, out_channels)
                         )
        self.to(device)
    def forward(self, data):
        data=data.to(device)
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        x=x.to(device)
        edge_index=edge_index.to(device)
        edge_attr=edge_attr.to(device)
        encoder=self.node_encoder(x)
        graph_encoder=encoder
        for i,block in enumerate(self.pe_blocks):
            gat,activation=block
            graph_encoder = gat(graph_encoder, edge_index,edge_attr)
        tfm_encoder   = self.transformer_encoder(encoder)
        graph_encoder = self.graph_linear(graph_encoder)
        tfm_encoder   = torch.sigmoid((tfm_encoder))
        ebd=tfm_encoder*graph_encoder
        return self.classifier(ebd)
    
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")





class Graph_models1(nn.Module):
    def __init__(self):
        super(Graph_models1,self).__init__()
        #Base model
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR",cache_dir="AI_model_cache/HuggingFace")
        self.base_model=AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR",cache_dir="AI_model_cache/HuggingFace")
        self.base_model.lm_head=nn.Identity()
        for param in self.base_model.parameters():
            param.requires_grad = False
        #Graph
        self.bbbp_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/bbbp.pt',map_location=torch.device('cpu'),weights_only=False)
        self.bace_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/bace.pt',map_location=torch.device('cpu'),weights_only=False)
        self.ctox_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/ctox.pt',map_location=torch.device('cpu'),weights_only=False)
        self.fda_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/fda.pt',map_location=torch.device('cpu'),weights_only=False)
        self.hiv_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/hiv.pt',map_location=torch.device('cpu'),weights_only=False)
        self.esol_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/esol.pt',map_location=torch.device('cpu'),weights_only=False)
        self.lipophilicity_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/lipophilicity.pt',map_location=torch.device('cpu'),weights_only=False)
        self.freeSolv_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/freeSolv.pt',map_location=torch.device('cpu'),weights_only=False)
        self.qm7_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/qm7.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E1CC2_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E1-CC2.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E2CC2_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E2-CC2.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E1PBE0_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E1-PBE0.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E2PBE0_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E2-PBE0.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E1CAM_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E1-CAM.pt',map_location=torch.device('cpu'),weights_only=False)
        self.E2CAM_graph=torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graphs/E2-CAM.pt',map_location=torch.device('cpu'),weights_only=False)
        #initialize model
        self.bbbp_model=GATV2_Transformer(in_channels=self.bbbp_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2)
        self.bace_model=GATV2_Transformer(in_channels=self.bace_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2)
        self.ctox_model=GATV2_Transformer(in_channels=self.ctox_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2)
        self.fda_model=GATV2_Transformer(in_channels=self.fda_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2)
        self.hiv_model=GATV2_Transformer(in_channels=self.hiv_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=2)
        self.esol_model=GATV2_Transformer(in_channels=self.esol_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.lipophilicity_model=GATV2_Transformer(in_channels=self.lipophilicity_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.freeSolv_model=GATV2_Transformer(in_channels=self.freeSolv_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.qm7_model=GATV2_Transformer(in_channels=self.qm7_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E1CC2_model=GATV2_Transformer(in_channels=self.E1CC2_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E2CC2_model=GATV2_Transformer(in_channels=self.E2CC2_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E1PBE0_model=GATV2_Transformer(in_channels=self.E1PBE0_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E2PBE0_model=GATV2_Transformer(in_channels=self.E2PBE0_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E1CAM_model=GATV2_Transformer(in_channels=self.E1CAM_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        self.E2CAM_model=GATV2_Transformer(in_channels=self.E2CAM_graph.x.size(1), hidden_channels=64, heads=8, num_layers=1, dropout=0.7,num_layers_transformers=1,out_channels=1)
        #Load weight
        
        self.bbbp_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/bbbp.pt',map_location=torch.device('cpu'),weights_only=True))
        self.bace_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/bace.pt',map_location=torch.device('cpu'),weights_only=True))
        self.ctox_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/ctox.pt',map_location=torch.device('cpu'),weights_only=True))
        self.fda_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/fda.pt',map_location=torch.device('cpu'),weights_only=True))
        self.hiv_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/hiv.pt',map_location=torch.device('cpu'),weights_only=True))
        self.esol_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/esol.pt',map_location=torch.device('cpu'),weights_only=True))
        self.freeSolv_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/freeSolv.pt',map_location=torch.device('cpu'),weights_only=True))
        self.qm7_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/qm7.pt',map_location=torch.device('cpu'),weights_only=True))
        self.lipophilicity_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/lipophilicity.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E1CC2_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E1-CC2.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E2CC2_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E2-CC2.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E1PBE0_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E1-PBE0.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E2PBE0_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E2-PBE0.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E1CAM_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E1-CAM.pt',map_location=torch.device('cpu'),weights_only=True))
        self.E2CAM_model.load_state_dict(torch.load('/content/drive/MyDrive/AI_store/Chemical_App/Source Code/AI_Model/AI_model_cache/Graph-Model-Weight/E2-CAM.pt',map_location=torch.device('cpu'),weights_only=True))

        self.to(device)
    def proccess_smile(self,smiles):
        return self.tokenizer(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
    def add_node_to_graph(self,new_node,graph, top_k=11):
       if new_node.dim()==1:
          new_node = new_node.unsqueeze(0)
       edge_index=graph.edge_index.to(device)
       x=graph.x.to(device)
       new_node = new_node.to(device)
       similarity_scores = F.cosine_similarity(x, new_node, dim=1)
       top_k_indices = similarity_scores.topk(top_k, largest=True).indices
       x_new = torch.cat([x, new_node], dim=0)
       new_node_index = x.shape[0]  
       new_edges = torch.stack([torch.tensor([new_node_index] * top_k).to(device), top_k_indices], dim=0)
       edge_index_new = torch.cat([edge_index, new_edges], dim=1)
       new_edge_attr = similarity_scores[top_k_indices].unsqueeze(1).to(device) 
       graph.x=x_new
       graph.edge_index=edge_index_new
       graph=self.create_edge_attributes(graph)
       return graph.to(device),new_node_index
    def create_edge_attributes(self,data):
        x = data.x
        edge_index = data.edge_index
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        source_features = x[source_nodes]  
        target_features = x[target_nodes]  
        combined_metric = F.cosine_similarity(source_features, target_features, dim=-1)
        combined_metric = combined_metric.unsqueeze(dim=-1)
        data.edge_attr = combined_metric
        return data
    def reg_inference(self,Model,new_node_features,data_graph,k):
      if new_node_features.dim()==1:
         new_node_features = new_node_features.unsqueeze(0)
      new_graph,new_node_index=self.add_node_to_graph(new_node_features,data_graph,k)
      Model.eval()
      with torch.no_grad():
         out=Model(new_graph).squeeze(dim=1)
      pred=out[new_node_index]
      return pred.squeeze(0)
    def cls_inference(self,Model,new_node_features,data_graph,k):
     if new_node_features.dim()==1:
        new_node_features = new_node_features.unsqueeze(0)
     new_graph,new_node_index=self.add_node_to_graph(new_node_features,data_graph,k)
     Model.eval()
     with torch.no_grad():
        out=Model(new_graph).squeeze(dim=1)
     pred=out[new_node_index]
     pred = torch.argmax(pred)
     return pred.squeeze(0)
    def forward(self,smile,task):
        inputs = self.proccess_smile(smile)
        outputs = self.base_model(**inputs, output_hidden_states=False)
        logits = outputs.logits
        ebd = logits.mean(dim=1)
        if task=='bbbp':
            out=self.cls_inference(self.bbbp_model,ebd,self.bbbp_graph,5)
        elif task=='bace':
            out=self.cls_inference(self.bace_model,ebd,self.bace_graph,6)
        elif task=='ctox':
            out=self.cls_inference(self.ctox_model,ebd,self.ctox_graph,12)
        elif task=='fda':
            out=self.cls_inference(self.fda_model,ebd,self.fda_graph,10)
        elif task=='hiv':
            out=self.cls_inference(self.hiv_model,ebd,self.hiv_graph,24)
        elif task=='esol':
            out=self.reg_inference(self.esol_model,ebd,self.esol_graph,5)
        elif task=='lipophilicity':
            out=self.reg_inference(self.lipophilicity_model,ebd,self.lipophilicity_graph,16)
        elif task=='freeSolv':
            out=self.reg_inference(self.freeSolv_model,ebd,self.freeSolv_graph,5)
        elif task=='qm7':
            out=self.reg_inference(self.qm7_model,ebd,self.qm7_graph,16)
        elif task=='E1CC2':
            out=self.reg_inference(self.E1CC2_model,ebd,self.E1CC2_graph,12)
        elif task=='E2CC2':
            out=self.reg_inference(self.E2CC2_model,ebd,self.E2CC2_graph,12)
        elif task=='E1PBE0':
            out=self.reg_inference(self.E1PBE0_model,ebd,self.E1PBE0_graph,12)
        elif task=='E2PBE0':
            out=self.reg_inference(self.E2PBE0_model,ebd,self.E2PBE0_graph,12)
        elif task=='E1CAM':
            out=self.reg_inference(self.E1CAM_model,ebd,self.E1CAM_graph,12)
        elif task=='E2CAM':
            out=self.reg_inference(self.E2CAM_model,ebd,self.E2CAM_graph,12)
        return out.item()


