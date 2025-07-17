import torch
import torch.nn.functional as F
import pandas as pd
from Gas_class_module import Gas_class_module

def load_model_and_gasbase():
    """加载模型和气体知识库"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载气体知识库
    gasbase_df = pd.read_csv('./data/Gasbase.csv')
    gasbase_df = gasbase_df.sort_values(by='gas_type')
    gasbase_features = gasbase_df.drop(columns=['gas_type']).values.astype('float32')
    x_gasbase = torch.tensor(gasbase_features, dtype=torch.float32).to(device)
    smoothing_filter = torch.ones(1, 1, 3, dtype=torch.float32).to(device) / 3.0
    x_gasbase = F.conv1d(x_gasbase.unsqueeze(1), smoothing_filter, padding=1).squeeze(1)
    x_gasbase = torch.diff(x_gasbase, dim=1)
    x_gasbase = F.normalize(x_gasbase, p=2, dim=1)
    # 加载模型
    model = Gas_class_module(in_chans=17, embed_dim=256, hidden_features=128, num_heads=8, attn_drop=0, drop=0.)
    model.load_state_dict(torch.load('./out_model/best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("模型加载成功")

    return model, x_gasbase, device

def predict_csv(csv_path):
    """推理CSV文件中的数据"""
    model, x_gasbase, device = load_model_and_gasbase()

    # 读取数据
    df = pd.read_csv(csv_path)
    #df = df.iloc[:40000:500]
    input_data = df.iloc[:, :18].values.astype('float32')
    true_labels = df['gas_type'].values.astype(int)
    print(f"开始推理 {len(input_data)} 个样本...")

    results = []
    for i, data in enumerate(input_data):
        # 转换为tensor
        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        smoothing_filter = torch.ones(1, 1, 3, dtype=torch.float32).to(device) / 3.0
        input_tensor = F.conv1d(input_tensor.unsqueeze(1), smoothing_filter, padding=1).squeeze(1)
        input_tensor = torch.diff(input_tensor, dim=1)
        input_tensor = F.normalize(input_tensor, p=2, dim=1)
        # 推理
        with torch.no_grad():
            output, attn_weights = model(input_tensor, x_gasbase)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

        results.append({
            'sample_id': i,
            'predicted_class': predicted_class + 1,  # +1因为类别从1开始
            'true_class': true_labels[i],
            'attn_weights': attn_weights.squeeze(0).tolist(),
            'confidence': confidence,

        })

        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{len(input_data)} 个样本")

    # 保存结果
    result_df = pd.DataFrame(results)
    output_path = csv_path.replace('.csv', '_predictions.csv')
    result_df.to_csv(output_path, index=False)

    print(f"推理完成，结果已保存到: {output_path}")
    print(f"预测类别分布:")
    print(result_df['predicted_class'].value_counts().sort_index())
    print(f"平均置信度: {result_df['confidence'].mean():.4f}")

def main():
    """主函数"""
    #csv_path = './data/simulation_less.csv'
    csv_path = './data/data_standard_1118_V3.csv'
    predict_csv(csv_path)

if __name__ == '__main__':
    main()