import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
class CrossAttention(nn.Module):
    def __init__(self, in_chans=18, embed_dim=256,num_heads=8, attn_drop=0):
        super().__init__()
        self.embedding=nn.Linear(in_chans, embed_dim)
        self.attention=nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop,batch_first=True)
        self.key_gen=nn.Linear(embed_dim, embed_dim)
        self.value_gen=nn.Linear(embed_dim, embed_dim)
    def forward(self, x_input,x_gasbase):
        #x_input的shape为(B,18)，x_gasbase的shape为(24,18)
        x1=self.embedding(x_input)
        x2=self.embedding(x_gasbase)
        q=x1.unsqueeze(1)#q的shape为(B,1,256)
        k=self.key_gen(x2).unsqueeze(0).repeat(x1.size(0),1,1)#k的shape为(B,24,256)       
        v=self.value_gen(x2).unsqueeze(0).repeat(x1.size(0),1,1)#v的shape为(B,24,256)
        q = nn.functional.normalize(q, p=2, dim=-1)
        k = nn.functional.normalize(k, p=2, dim=-1)
        attn_output,attn_weights=self.attention(q,k,v)#attn_output的shape为(B,1,256)，attn_weights的shape为(B,1,24)
        context_vector=attn_output.squeeze(1)
        #context_vector的shape为(B,256)，attn_weights的shape为(B,24)
        return context_vector,attn_weights
    
class MLP(nn.Module):
    def __init__(self, in_features=256, hidden_features=128, out_features=256, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Classifer(nn.Module):
    def __init__(self, in_features=256, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop=nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Gas_class_module(nn.Module):
    def __init__(self, in_chans=18, embed_dim=256, hidden_features=128, num_heads=8, attn_drop=0, drop=0.):
        super().__init__()
        self.cross_attention=CrossAttention(in_chans, embed_dim, num_heads, attn_drop)
        self.classifer=nn.Linear(embed_dim, 24)
        self.norm=nn.LayerNorm(embed_dim)
        self.mlp=MLP(embed_dim, hidden_features, embed_dim, drop)
    def forward(self, x_input, x_gasbase):
        context_vector,attn_weights=self.cross_attention(x_input, x_gasbase)
        x=context_vector
        x1=self.mlp(self.norm(x))+x
        output=self.classifer(self.norm(x1))
        return output,attn_weights

class Gas_Data(Dataset):
    def __init__(self, x_input, labels):

        self.x_input = x_input
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
         
        return self.x_input[idx], self.labels[idx]
    
class GasDatasplit():
    def __init__(self,data_csv_path,test_size=0.2,val_size=0.1):
        
        # 2. 加载主要的训练/测试数据 (这部分逻辑保持不变)
        data_df = pd.read_csv(data_csv_path)
        #data_df = data_df.sample(frac=0.7, random_state=42)
        self.x_input = data_df.iloc[:, :18].values.astype('float32')
        self.x_input=torch.tensor(self.x_input, dtype=torch.float32)
        smoothing_filter = torch.ones(1, 1, 3, dtype=torch.float32) / 3.0
        self.x_input = F.conv1d(self.x_input.unsqueeze(1), smoothing_filter, padding=1).squeeze(1)
        self.x_input = torch.diff(self.x_input, dim=1)
        self.x_input =F.normalize(self.x_input, p=2, dim=1)
        raw_labels = data_df.iloc[:, 18].values.astype(int)
        self.labels = torch.tensor(raw_labels - 1, dtype=torch.long)
        self.test_size=test_size
        self.val_size=val_size
        
    def __call__(self):
        data_train, data_test, label_train, label_test = train_test_split(self.x_input, self.labels, test_size=self.test_size, random_state=42)
        data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=self.val_size, random_state=42)
        train_dataset=Gas_Data(data_train, label_train)
        val_dataset=Gas_Data(data_val, label_val)
        test_dataset=Gas_Data(data_test, label_test)
        return train_dataset, val_dataset, test_dataset

        
def get_gasbase(gasbase_csv_path):
    # 1. 加载 Gasbase "知识库" (用于生成 K 和 V)
    # pandas 会自动将第一行识别为表头
    gasbase_df = pd.read_csv(gasbase_csv_path)
        # 按 'gas_type' 列排序，以确保气体顺序正确
    gasbase_df = gasbase_df.sort_values(by='gas_type')
    # 提取特征部分 (通过列名丢弃 'gas_type' 列)
    gasbase_features = gasbase_df.drop(columns=['gas_type']).values.astype('float32')
    x_gasbase = torch.tensor(gasbase_features, dtype=torch.float32)
    smoothing_filter = torch.ones(1, 1, 3, dtype=torch.float32) / 3.0
    x_gasbase = F.conv1d(x_gasbase.unsqueeze(1), smoothing_filter, padding=1).squeeze(1)
    x_gasbase = torch.diff(x_gasbase, dim=1)
    x_gasbase = F.normalize(x_gasbase, p=2, dim=1)
    return x_gasbase

def calculate_accuracy(outputs, labels):
    """
    计算一个批次的正确预测数和总样本数。
    
    参数:
    outputs (torch.Tensor): 模型的原始输出 (logits)。
    labels (torch.Tensor): 真实的标签。
    
    返回:
    tuple: (正确预测的数量, 当前批次的总样本数)
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total


        

