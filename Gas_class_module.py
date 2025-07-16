import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
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
        data_df = data_df.sample(frac=0.3, random_state=42)
        self.x_input = data_df.iloc[:, :18].values.astype('float32')
        self.x_input=torch.tensor(self.x_input, dtype=torch.float32)
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

        
    


        

