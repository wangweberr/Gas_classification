import os
import torch
import torch.nn as nn
import logging
import time
import shutil
from tqdm import tqdm
import pandas as pd
from Gas_class_module import Gas_class_module, GasDatasplit,Gas_Data, Classifer,CrossAttention
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts



batch_size=16
epochs=20
patience=10
data_csv_path='./data/simulation.csv'
gasbase_csv_path='./data/Gasbase.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_gasbase(gasbase_csv_path):
    # 1. 加载 Gasbase "知识库" (用于生成 K 和 V)
    # pandas 会自动将第一行识别为表头
    gasbase_df = pd.read_csv(gasbase_csv_path)
        # 按 'gas_type' 列排序，以确保气体顺序正确
    gasbase_df = gasbase_df.sort_values(by='gas_type')
    # 提取特征部分 (通过列名丢弃 'gas_type' 列)
    gasbase_features = gasbase_df.drop(columns=['gas_type']).values.astype('float32')
    x_gasbase = torch.tensor(gasbase_features, dtype=torch.float32)
    return x_gasbase
    # self.x_gasbase 的形状仍然是 (24, 18)


def train_epoch(model,train_loader,optimizer,criterion,device,epoch,x_gasbase):
    model.train()
    train_loss=0.0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(
    total=len(train_loader), 
    desc=f"训练轮次 {epoch+1}", 
    unit="batch",
    ncols=100,
    leave=False
    )
    for data,label in train_loader:
        data=data.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output,_=model(data,x_gasbase)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        current_loss=loss.detach().item()
        train_loss+=current_loss
        correct, total = calculate_accuracy(output, label)
        total_correct += correct
        total_samples += total
        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
        })
        progress_bar.update()
    progress_bar.close()
    
    return train_loss/len(train_loader) ,total_correct / total_samples

def evaluate(model,val_loader,criterion,device,x_gasbase):
    total_loss=0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(
    total=len(val_loader), 
    desc="验证", 
    unit="batch",
    ncols=100,
    leave=False
    
)
    with torch.no_grad():
        for data,label in val_loader:
            data=data.to(device)
            label=label.to(device)
            output,__annotations__=model(data,x_gasbase)
            loss=criterion(output,label)
            current_loss=loss.detach().item()
            total_loss+=current_loss
            correct, total = calculate_accuracy(output, label)
            total_correct += correct
            total_samples += total
            progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}",
                    })
            progress_bar.update()
        progress_bar.close()
        return total_loss/len(val_loader) ,total_correct / total_samples




def main():
    #基础设置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./output', exist_ok=True)
    log_file_path = os.path.join('./output', 'train.log')
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')#文件写入器
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))#设置格式
    logging.getLogger().addHandler(file_handler)
    #数据集
    train_dataset,val_dataset,test_dataset=GasDatasplit(data_csv_path,test_size=0.2,val_size=0.1)()
    logging.info(f"数据集加载完成: 训练集{len(train_dataset)}样本, 验证集{len(val_dataset)}样本, 测试集{len(test_dataset)}样本")
    train_loader=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    x_gasbase=get_gasbase(gasbase_csv_path)
    x_gasbase=x_gasbase.to(device)
    #模型
    model=Gas_class_module(in_chans=18, embed_dim=256, hidden_features=128, num_heads=8, attn_drop=0, drop=0.)
    model = model.to(device)
    logging.info(f"模型已加载到设备: {device}")
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine  = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
    criterion =nn.CrossEntropyLoss()
    #看总参数
    total_params=sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    #训练
    history={
    'train_loss':[],
    'val_loss':[],
    'train_acc':[],
    'val_acc':[]}
    best_val_loss=float('inf')#初始化最佳验证损失
    best_epoch=0              #记录最佳模型对应轮次
    patience_counter=0        #计数器，记录连续未改进的轮次

    logging.info("========== 开始新的训练任务 ==========")
    for epoch in range(epochs):
        logging.info(f"轮次{epoch+1}/{epochs}开始训练")
        train_loss,train_acc=train_epoch(model,train_loader,optimizer,criterion,device,epoch,x_gasbase)
        val_loss,val_acc=evaluate(model,val_loader,criterion,device,x_gasbase)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        #打印训练信息
        logging.info(f"轮次{epoch+1}/{epochs} 训练loss={train_loss:.4f}; 训练acc={train_acc:.4f}; 验证loss={val_loss:.4f}; 验证acc={val_acc:.4f}")
        #保存模型
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            best_epoch=epoch
            patience_counter=0
            current_model_path=os.path.join('./out_model',f'best_model_{epoch+1}.pth')
            best_model_path=os.path.join('./out_model','best_model.pth')
            model_to_save=model
            torch.save(model_to_save.state_dict(),current_model_path)
            torch.save(model_to_save.state_dict(),best_model_path)
            logging.info(f"轮次{epoch+1} 新最佳模型已保存  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        else:
            patience_counter+=1
            logging.info(f"轮次{epoch+1} 验证损失未改进  patience={patience_counter}/{patience}")
            if patience_counter>=patience:
                logging.info(f"早停: 验证损失连续{patience}轮没有下降")
                break
    logging.info("训练完成，开始测试集评估")
    test_loss,test_acc=evaluate(model,test_loader,criterion,device,x_gasbase)
    loss_str = f"{best_val_loss:.3f}".replace('.', 'p')
    final_model_path=os.path.join('./out_model',f'final_model--loss{loss_str}.pth')
    shutil.copy2(best_model_path, final_model_path)
    logging.info(f"最佳模型已另存为: {final_model_path}")
    logging.info(f"最佳验证损失 {best_val_loss:.4f}, 最佳轮次 {best_epoch+1}")
    logging.info(f"测试集: loss {test_loss:.4f}, acc {test_acc:.4f}")
if __name__=='__main__':
    main()