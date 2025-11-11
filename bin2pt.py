import torch

def convert_pytorch_to_custom_format(input_path, output_path):
    """
    将pytorch_model.bin转换为自定义的.pt格式
    
    参数:
        input_path: 输入的pytorch模型文件路径
        output_path: 输出的.pt文件路径
    """
    # 加载原始模型
    print(f"加载模型: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')
    
    # 显示模型信息
    print(f"模型中的键数量: {len(state_dict)}")
    print("\n模型中的所有键:")
    for i, key in enumerate(state_dict.keys()):
        print(f"{i+1}: {key} - {state_dict[key].shape}")
    
    # 保存为.pt格式
    print(f"\n保存模型到: {output_path}")
    torch.save(state_dict, output_path)
    
    # 验证保存的模型
    print("\n验证保存的模型...")
    loaded_state_dict = torch.load(output_path, map_location='cpu')
    print(f"验证成功，加载的模型键数量: {len(loaded_state_dict)}")
    
    return state_dict

if __name__ == "__main__":
    # 输入和输出路径
    input_model_path = "./bert/pytorch_model.bin"
    output_model_path = "premodel_DNABERT1_model_pgd_gene_new_19.pt"
    
    # 执行转换
    state_dict = convert_pytorch_to_custom_format(input_model_path, output_model_path)
    
    print("\n转换完成！")