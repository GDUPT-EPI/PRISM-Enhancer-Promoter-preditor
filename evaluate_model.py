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
import xml.etree.ElementTree as ET

# 导入项目模块
from config import *
from data_loader import load_all_test_data, load_all_val_data
from models.pleat.embedding import KMerTokenizer, DNAEmbedding
from models.pleat.optimized_pre import sequence_to_tokens_fast, tokens_to_ids_fast, OptimizedSequenceDataset
from models.EPIModel import EPIModel
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_cell_types_from_xml(xml_path):
    """
    从XML文件中加载细胞类型列表
    
    Args:
        xml_path: XML文件路径
        
    Returns:
        list: 细胞类型名称列表
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        cell_types = [cell.get('name') for cell in root.findall('type')]
        return cell_types
    except Exception as e:
        print(f"警告: 无法从 {xml_path} 加载细胞类型: {e}")
        print("使用默认细胞系列表")
        return TEST_CELL_LINES

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


def evaluate_model_on_all_data(model, device, cell_lines_to_eval):
    """
    在ALL数据上评估模型，并按细胞系分组计算指标
    
    Args:
        model: 要评估的模型
        device: 设备(cpu/cuda)
        cell_lines_to_eval: 要评估的细胞系列表
        
    Returns:
        list: 每个细胞系的评估结果列表
    """
    # 加载ALL测试数据
    test_data = load_all_test_data()
    if "ALL" not in test_data:
        print("警告: ALL测试数据不存在")
        return []
    
    enhancers_all, promoters_all, labels_all = test_data["ALL"]
    
    # 创建优化的测试数据集
    optimized_test_dataset = OptimizedSequenceDataset(
        enhancers=enhancers_all,
        promoters=promoters_all,
        labels=labels_all,
        cache_dir=os.path.join(CACHE_DIR, "ALL_test_cache"),
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
    
    # 获取ALL数据中的细胞系信息
    pairs_df = pd.read_csv(os.path.join(DATA_DIR, "test", "ALL", "pairs_hg19.csv"))
    cell_lines_in_data = pairs_df['cell_line'].unique()
    
    # 为每个要评估的细胞系创建结果存储
    results = []
    
    # 对每个细胞系进行评估
    for cell_line in cell_lines_to_eval:
        if cell_line == "ALL":
            continue  # 跳过ALL，因为我们要按具体细胞系评估
            
        # 检查该细胞系是否在数据中
        if cell_line not in cell_lines_in_data:
            print(f"警告: {cell_line} 不在ALL测试数据中，跳过")
            continue
        
        print(f"在ALL数据中评估 {cell_line} 细胞系...")
        
        # 获取该细胞系的样本索引
        cell_indices = pairs_df[pairs_df['cell_line'] == cell_line].index.tolist()
        
        # 如果没有该细胞系的样本，跳过
        if len(cell_indices) == 0:
            print(f"警告: {cell_line} 在ALL测试数据中没有样本，跳过")
            continue
        
        model.eval()
        all_preds = []
        all_labels = []
        
        # 遍历数据加载器，只收集该细胞系的预测结果
        with torch.no_grad():
            batch_start_idx = 0
            for data in tqdm(test_loader, desc=f"Evaluating {cell_line} in ALL"):
                enhancer_ids, promoter_ids, enhancer_features, promoter_features, labels = data
                enhancer_ids = enhancer_ids.to(device)
                promoter_ids = promoter_ids.to(device)
                enhancer_features = enhancer_features.to(device)
                promoter_features = promoter_features.to(device)
                labels = labels.to(device)
                
                outputs, _, _ = model(enhancer_ids, promoter_ids, enhancer_features, promoter_features)
                
                # 获取当前批次中的样本在原始数据中的索引范围
                batch_end_idx = batch_start_idx + len(labels)
                batch_indices = list(range(batch_start_idx, batch_end_idx))
                
                # 找出当前批次中属于目标细胞系的样本
                for i, idx in enumerate(batch_indices):
                    if idx in cell_indices:
                        all_preds.append(outputs[i].cpu().numpy())
                        all_labels.append(labels[i].cpu().numpy())
                
                batch_start_idx = batch_end_idx
        
        if len(all_preds) == 0:
            print(f"警告: {cell_line} 在ALL测试数据中没有有效样本，跳过")
            continue
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)
        
        # 计算各种指标
        try:
            aupr = average_precision_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_preds)
            
            # 使用0.5作为阈值计算其他指标
            binary_preds = (all_preds >= 0.5).astype(int)
            accuracy = accuracy_score(all_labels, binary_preds)
            f1 = f1_score(all_labels, binary_preds)
            recall = recall_score(all_labels, binary_preds)
            
            # 计算正样本的平均概率
            positive_mask = all_labels == 1
            if np.sum(positive_mask) > 0:
                positive_avg_prob = np.mean(all_preds[positive_mask])
            else:
                positive_avg_prob = 0.0
            
            results.append({
                'cell_line': cell_line,
                'aupr': aupr,
                'auc': auc,
                'accuracy': accuracy,
                'f1': f1,
                'recall': recall,
                'positive_avg_prob': positive_avg_prob,
                'total_samples': len(all_labels),
                'positive_samples': np.sum(positive_mask),
                'negative_samples': np.sum(all_labels == 0)
            })
            
            # 打印结果
            print(f"\n{cell_line} 评估结果 (来自ALL数据):")
            print(f"  AUPR: {aupr:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  正样本平均概率: {positive_avg_prob:.4f}")
            print(f"  总样本数: {len(all_labels)}")
            print(f"  正样本数: {np.sum(positive_mask)}")
            print(f"  负样本数: {np.sum(all_labels == 0)}")
        except Exception as e:
            print(f"评估 {cell_line} 时出错: {str(e)}")
            continue
    
    return results


def main():
    """
    主函数：加载模型和数据，进行评估，并保存结果
    """
    # 设置设备
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model_path = os.path.join(PROJECT_ROOT, "save_model/CBAT/epimodel_epoch_2.pth")
    print(f"加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请检查模型路径是否正确")
        return
    
    # 创建模型实例
    model = EPIModel()
    
    # 使用weights_only=False加载模型状态字典，因为模型包含自定义类
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    # 使用strict=False来忽略不匹配的键
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # 从XML文件加载细胞类型列表
    cell_type_xml_path = os.path.join(PROJECT_ROOT, "vocab", "cell_type.xml")
    cell_lines_from_xml = load_cell_types_from_xml(cell_type_xml_path)
    
    # 添加ALL细胞系到评估列表中（如果存在）
    all_cell_lines = list(cell_lines_from_xml)
    if "ALL" not in all_cell_lines:
        all_cell_lines.append("ALL")
    
    print(f"评估细胞系: {', '.join(all_cell_lines)}")
    
    # 存储所有结果
    all_results = []
    
    # 首先在ALL数据上评估各个细胞系
    all_results.extend(evaluate_model_on_all_data(model, device, all_cell_lines))
    
    # 然后尝试加载各个细胞系的独立测试数据（如果存在）
    test_data = load_all_test_data()
    for cell_line in all_cell_lines:
        if cell_line == "ALL":
            continue  # 已经在上面处理过了
            
        # 检查是否有独立的测试数据
        if cell_line in test_data:
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
            results['cell_line'] = f"{cell_line}_independent"  # 标记为独立数据
            all_results.append(results)
            
            # 打印结果
            print(f"\n{cell_line} 独立测试数据评估结果:")
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