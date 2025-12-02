2025-12-02 03:26:11,492 - INFO - PRISM预训练日志系统已初始化
2025-12-02 03:26:11,492 - INFO - 日志文件: /root/autodl-tmp/CBAT/CBAT/log
2025-12-02 03:26:11,492 - INFO - 预处理线程数: 12
2025-12-02 03:26:11,492 - INFO - ================================================================================
2025-12-02 03:26:11,492 - INFO - PRISM预训练开始 (Domain-KL数据)
2025-12-02 03:26:11,492 - INFO - ================================================================================
2025-12-02 03:26:11,492 - INFO - 加载训练数据 (domain-kl)...
2025-12-02 03:26:12,652 - INFO - 训练样本数: 90937
2025-12-02 03:26:12,655 - INFO - 训练细胞系: CD34, HeLa, K562, Liver, Thyroid
2025-12-02 03:26:14,223 - INFO - 创建数据加载器...
2025-12-02 03:26:14,223 - INFO - 使用随机EP导入 (跨细胞系随机批次)
2025-12-02 03:26:14,224 - INFO - 创建PRISM模型...
2025-12-02 03:26:15,826 - INFO - 模型总参数: 68,693,803
2025-12-02 03:26:15,827 - INFO - 可训练参数: 68,693,803
2025-12-02 03:26:15,827 - INFO - GPU可用: True
2025-12-02 03:26:15,827 - INFO - 模型在GPU上: True
2025-12-02 03:26:15,830 - INFO - 批量大小: 128 (纯细胞系批次)
2025-12-02 03:26:15,830 - INFO - 训练轮数: 20
2025-12-02 03:26:15,830 - INFO - 学习率: 0.0002
2025-12-02 03:26:15,830 - INFO - 总训练步数: 14220
2025-12-02 03:26:15,830 - INFO - ================================================================================
2025-12-02 03:26:15,830 - INFO - 开始训练
2025-12-02 03:26:15,830 - INFO - ================================================================================
2025-12-02 03:26:15,834 - INFO - 未发现可恢复检查点，将从头开始训练
2025-12-02 03:26:15,834 - INFO - 模式: 仅训练EP互作主干
2025-12-02 03:34:18,436 - INFO - Epoch 1/20 - Train Loss: 0.6938, Cell Acc: 0.0000, EP Acc: 0.2197
2025-12-02 03:34:18,973 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_1.pth
2025-12-02 03:34:20,056 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_1.pt
2025-12-02 03:34:20,056 - INFO - 模式: 仅训练EP互作主干
2025-12-02 03:42:22,889 - INFO - Epoch 2/20 - Train Loss: 0.5704, Cell Acc: 0.0000, EP Acc: 0.1994
2025-12-02 03:42:23,479 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_2.pth
2025-12-02 03:42:24,715 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_2.pt
2025-12-02 03:42:24,717 - INFO - 模式: 仅训练EP互作主干
2025-12-02 03:50:27,432 - INFO - Epoch 3/20 - Train Loss: 0.5617, Cell Acc: 0.0000, EP Acc: 0.1995
2025-12-02 03:50:27,915 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_3.pth
2025-12-02 03:50:28,998 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_3.pt
2025-12-02 03:50:28,998 - INFO - 模式: 仅训练EP互作主干
2025-12-02 03:58:31,666 - INFO - Epoch 4/20 - Train Loss: 0.5575, Cell Acc: 0.0000, EP Acc: 0.1994
2025-12-02 03:58:32,306 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_4.pth
2025-12-02 03:58:33,478 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_4.pt
2025-12-02 03:58:33,478 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:06:37,903 - INFO - Epoch 5/20 - Train Loss: 0.5541, Cell Acc: 0.0000, EP Acc: 0.1995
2025-12-02 04:06:38,553 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_5.pth
2025-12-02 04:06:39,735 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_5.pt
2025-12-02 04:06:39,736 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:14:43,651 - INFO - Epoch 6/20 - Train Loss: 0.5542, Cell Acc: 0.0000, EP Acc: 0.1996
2025-12-02 04:14:44,268 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_6.pth
2025-12-02 04:14:45,439 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_6.pt
2025-12-02 04:14:45,439 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:22:49,625 - INFO - Epoch 7/20 - Train Loss: 0.5441, Cell Acc: 0.0000, EP Acc: 0.1995
2025-12-02 04:22:50,244 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_7.pth
2025-12-02 04:22:51,454 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_7.pt
2025-12-02 04:22:51,454 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:30:56,357 - INFO - Epoch 8/20 - Train Loss: 0.5454, Cell Acc: 0.0000, EP Acc: 0.1995
2025-12-02 04:30:56,994 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_8.pth
2025-12-02 04:30:58,162 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_8.pt
2025-12-02 04:30:58,163 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:39:02,080 - INFO - Epoch 9/20 - Train Loss: 0.5443, Cell Acc: 0.0000, EP Acc: 0.1994
2025-12-02 04:39:02,705 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_9.pth
2025-12-02 04:39:03,853 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_9.pt
2025-12-02 04:39:03,853 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:47:08,261 - INFO - Epoch 10/20 - Train Loss: 0.5425, Cell Acc: 0.0000, EP Acc: 0.1995
2025-12-02 04:47:08,862 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_10.pth
2025-12-02 04:47:10,057 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_10.pt
2025-12-02 04:47:10,057 - INFO - 模式: 仅训练EP互作主干
2025-12-02 04:55:15,569 - INFO - Epoch 11/20 - Train Loss: 0.5457, Cell Acc: 0.0000, EP Acc: 0.1994
2025-12-02 04:55:16,166 - INFO - 保存检查点: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_epoch_11.pth
2025-12-02 04:55:17,376 - INFO - 保存完整状态: /root/autodl-tmp/CBAT/CBAT/save_model/251202-EP/prism_full_epoch_11.pt
2025-12-02 04:55:17,376 - INFO - 模式: 仅训练EP互作主干
