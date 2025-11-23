#!/usr/bin/env python3
"""
模型评估脚本
用于在测试数据集上评估预训练模型的性能
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, accuracy_score
from tqdm import tqdm
from pathlib import Path

# 导入项目模块
from config import *
from data_loader import load_all_test_data
from models.pleat.embedding import KMerTokenizer, DNAEmbedding
from models.pleat.optimized_pre import sequence_to_tokens_fast, tokens_to_ids_fast, OptimizedSequenceDataset
from models.EPIModel import EPIModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

def simple_collate_fn(batch):
    """
    简化的collate函数，因为数据已在Dataset中预处理
    
    Args:
        batch: 包含多个样本的列表，每个样本是一个元组
              (enhancer_ids, promoter_ids, enhancer_features, promoter_features, label)
              
    Returns:
        tuple: 包含五个元素的元组
            - enhancer_ids: 增强子ID张量
            - promoter_ids: 启动子ID张量
            - enhancer_features: 增强子特征张量
            - promoter_features: 启动子特征张量
            - labels: 标签张量
    """
    # 分离各个组件
    enhancer_ids = [item[0] for item in batch]
    promoter_ids = [item[1] for item in batch]
    enhancer_features = [item[2] for item in batch]
    promoter_features = [item[3] for item in batch]
    labels = [item[4] for item in batch]
    
    # 使用pad_sequence填充序列
    padded_enhancer_ids = torch.nn.utils.rnn.pad_sequence(enhancer_ids, batch_first=True, padding_value=0)
    padded_promoter_ids = torch.nn.utils.rnn.pad_sequence(promoter_ids, batch_first=True, padding_value=0)
    
    # 如果填充后长度仍小于最大长度，继续填充
    if padded_enhancer_ids.size(1) < MAX_ENHANCER_LENGTH:
        padding_size = MAX_ENHANCER_LENGTH - padded_enhancer_ids.size(1)
        padded_enhancer_ids = F.pad(
            padded_enhancer_ids, (0, padding_size), mode='constant', value=0
        )
    
    if padded_promoter_ids.size(1) < MAX_PROMOTER_LENGTH:
        padding_size = MAX_PROMOTER_LENGTH - padded_promoter_ids.size(1)
        padded_promoter_ids = F.pad(
            padded_promoter_ids, (0, padding_size), mode='constant', value=0
        )
    
    # 直接堆叠特征张量和标签
    padded_enhancer_features = torch.stack(enhancer_features)
    padded_promoter_features = torch.stack(promoter_features)
    labels = torch.tensor(labels, dtype=torch.float)
    
    return padded_enhancer_ids, padded_promoter_ids, padded_enhancer_features, padded_promoter_features, labels


def evaluate_model(model, dataloader, device, cell_name):
    """
    评估模型在指定数据集上的性能
    
    Args:
        model: 要评估的模型
        dataloader: 测试数据加载器
        device: 设备(cpu/cuda)
        cell_name: 细胞系名称
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {cell_name}"):
            enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            enhancer_features = enhancer_features.to(device)
            promoter_features = promoter_features.to(device)
            labels = labels.to(device)
            
            outputs, _, _ = model(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # 计算各种指标
    aupr = average_precision_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    # 使用0.5作为阈值计算其他指标
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    
    # 计算正样本的平均概率（所有正样本的预测概率平均值）
    positive_mask = all_labels == 1
    if np.sum(positive_mask) > 0:
        positive_avg_prob = np.mean(all_preds[positive_mask])
    else:
        positive_avg_prob = 0.0
    
    return {
        'cell_line': cell_name,
        'aupr': aupr,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'positive_avg_prob': positive_avg_prob,
        'total_samples': len(all_labels),
        'positive_samples': np.sum(positive_mask),
        'negative_samples': np.sum(all_labels == 0)
    }


def main():
    """
    主函数：加载模型和数据，进行评估，并保存结果
    """
    # 设置设备
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model_path = os.path.join(PROJECT_ROOT, "save_model/CBAT/DNABERT1_pgd_genes_ALL_train_model_lr0.0002_epoch0.pt")
    print(f"加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请检查模型路径是否正确")
        return
    
    # 使用weights_only=False加载模型，因为模型包含自定义类
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    
    # 加载所有测试数据
    test_data = load_all_test_data()
    
    # 排除ALL细胞系
    cell_lines_to_evaluate = [cell for cell in TEST_CELL_LINES if cell != "ALL"]
    print(f"评估细胞系: {', '.join(cell_lines_to_evaluate)}")
    
    # 存储所有结果
    all_results = []
    
    # 对每个细胞系进行评估
    for cell_line in cell_lines_to_evaluate:
        if cell_line not in test_data:
            print(f"警告: {cell_line} 的测试数据不存在，跳过")
            continue
            
        enhancers_test, promoters_test, labels_test = test_data[cell_line]
        
        # 创建优化的测试数据集
        optimized_test_dataset = OptimizedSequenceDataset(
            enhancers=enhancers_test,
            promoters=promoters_test,
            labels=labels_test,
            cache_dir=os.path.join(CACHE_DIR, f"{cell_line}_test_cache"),
            use_cache=True
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            dataset=optimized_test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )
        
        # 评估模型
        results = evaluate_model(model, test_loader, device, cell_line)
        all_results.append(results)
        
        # 打印结果
        print(f"\n{cell_line} 评估结果:")
        print(f"  AUPR: {results['aupr']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  F1分数: {results['f1']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  正样本平均概率: {results['positive_avg_prob']:.4f}")
        print(f"  总样本数: {results['total_samples']}")
        print(f"  正样本数: {results['positive_samples']}")
        print(f"  负样本数: {results['negative_samples']}")
    
    # 将结果转换为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果到CSV文件
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'compete','CBAT')):
        os.makedirs(os.path.join(PROJECT_ROOT, 'compete','CBAT'))
    output_path = os.path.join(PROJECT_ROOT, 'compete','CBAT',"evulate.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n评估结果已保存到: {output_path}")
    
    # 计算并打印平均指标
    avg_aupr = np.mean([r['aupr'] for r in all_results])
    avg_auc = np.mean([r['auc'] for r in all_results])
    avg_accuracy = np.mean([r['accuracy'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_positive_prob = np.mean([r['positive_avg_prob'] for r in all_results])
    
    print("\n平均指标:")
    print(f"  平均AUPR: {avg_aupr:.4f}")
    print(f"  平均AUC: {avg_auc:.4f}")
    print(f"  平均准确率: {avg_accuracy:.4f}")
    print(f"  平均F1分数: {avg_f1:.4f}")
    print(f"  平均召回率: {avg_recall:.4f}")
    print(f"  平均正样本概率: {avg_positive_prob:.4f}")


if __name__ == "__main__":
    main()