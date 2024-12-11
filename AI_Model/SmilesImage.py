import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from transformers import AutoTokenizer
from torchvision import transforms
import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Set cache directories programmatically
#Backend/Store/Hugging Face cache
os.environ["TRANSFORMERS_CACHE"] = "AI_model_cache/Image2Molecule"
os.environ["HF_DATASETS_CACHE"] = "AI_model_cache/Image2Molecule"
os.environ["HF_METRICS_CACHE"] = "AI_model_cache/Image2Molecule"
def load_mobilenet(pretrained=True, cache_dir='AI_model_cache/Image2Molecule'):
    if cache_dir:
        os.environ['TORCH_HOME'] = cache_dir
    return mobilenet_v3_large(pretrained=pretrained)

class CNNLSTMModelMobileNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(CNNLSTMModelMobileNet, self).__init__()
        
        # MobileNetV3 Encoder
        mobilenet = load_mobilenet(pretrained=True)
        self.cnn = nn.Sequential(*list(mobilenet.children())[:-2])  # Remove classifier
        self.cnn_fc = nn.Linear(960, embedding_dim)  # Map to desired dimension
        # LSTM Decoder
        self.lstm = nn.LSTM(embedding_dim , hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Project to vocabulary size
        
    def forward(self, images):
        # CNN forward pass
        cnn_features = self.cnn(images)  # Shape: [batch_size, cnn_output_dim, 1, 1]
        cnn_features = cnn_features.permute(0, 2, 3, 1)
        cnn_features = cnn_features.view(cnn_features.size(0), -1, cnn_features.size(-1))
        cnn_features = self.cnn_fc(cnn_features)  # Map to [batch_size, cnn_output_dim]
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_features)  # Shape: [batch_size, seq_len, hidden_dim]
        logits = self.fc(lstm_out)  # Shape: [batch_size, seq_len, vocab_size]
        
        return logits
    
class Smiles2Image():
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True,model_max_length=49, cache_dir="AI_model_cache/Image2Molecule")
        self.model=CNNLSTMModelMobileNet(embedding_dim=300, hidden_dim=512, vocab_size=self.tokenizer.vocab_size+1)
        self.model.load_state_dict(torch.load('AI_model_cache/Image2Molecule/Model (best).pt',weights_only=False))
        self.model.eval()
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match your model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])
    def capImg(self,Image):
        input=self.processed_image(Image)
        input=input.unsqueeze(0)
        out=self.model(input)
        
        predicted_token_ids = torch.argmax(out, dim=-1)
        smiles = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        return smiles
    def processed_image(self,Image):
        out=self.transform(Image)
        return out
    
   
