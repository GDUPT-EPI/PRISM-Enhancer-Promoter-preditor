import torch
import numpy as np

def convert_pytorch_to_dnabert1_matrix(pytorch_model_path, output_path):
    """
    将pytorch模型中的嵌入矩阵提取并保存为dnabert1_matrix.npy格式
    
    参数:
        pytorch_model_path: pytorch模型文件路径
        output_path: 输出的npy文件路径
    """
    # 加载pytorch模型
    print(f"加载模型: {pytorch_model_path}")
    state_dict = torch.load(pytorch_model_path, map_location='cpu')
    
    # 提取词嵌入矩阵
    word_embeddings_key = 'bert.embeddings.word_embeddings.weight'
    if word_embeddings_key not in state_dict:
        raise KeyError(f"模型中未找到键: {word_embeddings_key}")
    
    word_embeddings = state_dict[word_embeddings_key]
    print(f"词嵌入矩阵形状: {word_embeddings.shape}")
    
    # 将PyTorch张量转换为NumPy数组
    word_embeddings_np = word_embeddings.cpu().numpy()
    
    # 保存为npy文件
    np.save(output_path, word_embeddings_np)
    print(f"已将嵌入矩阵保存到: {output_path}")
    
    return word_embeddings_np

if __name__ == "__main__":
    # 输入和输出路径
    version = "6"  # 版本号
    input_model_path = "./bert/DNABERT{}.bin".format(version)
    output_matrix_path = "dnabert{}_matrix.npy".format(version)
    
    # 执行转换
    embedding_matrix = convert_pytorch_to_dnabert1_matrix(input_model_path, output_matrix_path)
    
    # 验证输出
    print(f"转换完成，输出矩阵形状: {embedding_matrix.shape}")
    print(f"矩阵数据类型: {embedding_matrix.dtype}")
    
    # 显示矩阵的前几行作为示例
    print("\n矩阵前5行前5列:")
    print(embedding_matrix[:5, :5])