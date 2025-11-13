from torch.utils.data import Dataset
from torchvision import transforms
import linecache
import os
import torch
import itertools
import numpy as np
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    
    def __init__(self, enhancers, promoters, labels):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        
        self.enhancer_count = len(enhancers)
        self.promoter_count = len(promoters)
        self.label_count = len(labels)
        
    def __getitem__(self, idx):
        #enhancer_sequence = linecache.getline(self.enhancers_file, idx + 1).strip()
        
        #promoter_sequence = linecache.getline(self.promoters_file, idx + 1).strip()
        #label = int(linecache.getline(self.labels_file, idx + 1).strip())
        
        '''
        enhancer_sequence_encoded = self.one_hot_encoding(enhancer_sequence)
        promoter_sequence_encoded = self.one_hot_encoding(promoter_sequence)
        '''
        '''
        enhancer_sequence_encoded = self.get_token_ids(enhancer_sequence)
        promoter_sequence_encoded = self.get_token_ids(promoter_sequence)
        '''
        enhancer_sequence = str(self.enhancers[idx])
        promoter_sequence = str(self.promoters[idx])
        
        
        label = int(self.labels[idx])
        return enhancer_sequence, promoter_sequence, label
    
    def __len__(self):
        return self.label_count
    
    def count_lines_in_txt(self, file_path):
        with open(file_path, "r") as file:
            line_count = len(file.readlines())
        
        return line_count
    
def match_motif(pwm_matrix, sequence, threshold):
        motif_length = len(pwm_matrix)
        matches = []
        for i in range(len(sequence) - motif_length + 1):
            subsequence = sequence[i:i+motif_length]
            score = 1.0
            for j in range(motif_length):
                nucleotide = subsequence[j]
                if score < 0: break
                if nucleotide == 'A':
                    score *= pwm_matrix[j][0]
                elif nucleotide == 'C':
                    score *= pwm_matrix[j][1]
                elif nucleotide == 'G':
                    score *= pwm_matrix[j][2]
                elif nucleotide == 'T':
                    score *= pwm_matrix[j][3]
            if score >= threshold:  # 设置一个阈值来决定匹配与否
                matches.append(i)
        return matches

def replace_motif_with_N(sequence, matches, motif_length):
        modified_sequence = list(sequence)
        for match in matches:
            modified_sequence[match:match+motif_length] = 'N' * motif_length
        return ''.join(modified_sequence)

class MaskDataSet(Dataset):

    def __init__(self, enhancers, promoters, labels, pwm_matrix):
        self.enhancers = enhancers
        self.promoters = promoters
        self.labels = labels
        self.pwm_matrix = pwm_matrix
        
        self.enhancer_count = len(enhancers)
        self.promoter_count = len(promoters)
        self.label_count = len(labels)

        
    def __getitem__(self, idx):
        #enhancer_sequence = linecache.getline(self.enhancers_file, idx + 1).strip()
        
        #promoter_sequence = linecache.getline(self.promoters_file, idx + 1).strip()
        #label = int(linecache.getline(self.labels_file, idx + 1).strip())
        
        '''
        enhancer_sequence_encoded = self.one_hot_encoding(enhancer_sequence)
        promoter_sequence_encoded = self.one_hot_encoding(promoter_sequence)
        '''
        '''
        enhancer_sequence_encoded = self.get_token_ids(enhancer_sequence)
        promoter_sequence_encoded = self.get_token_ids(promoter_sequence)
        '''
        enhancer_sequence = str(self.enhancers[idx])
        promoter_sequence = str(self.promoters[idx])

        matches = match_motif(self.pwm_matrix, enhancer_sequence, 0)
        enhancer_sequence = replace_motif_with_N(enhancer_sequence, matches, len(self.pwm_matrix))

        matches = match_motif(self.pwm_matrix, promoter_sequence, 0)
        promoter_sequence = replace_motif_with_N(promoter_sequence, matches, len(self.pwm_matrix))

        label = int(self.labels[idx])
        return enhancer_sequence, promoter_sequence, label
    
    def __len__(self):
        return self.label_count
    
    def count_lines_in_txt(self, file_path):
        with open(file_path, "r") as file:
            line_count = len(file.readlines())
        
        return line_count
    
    
    
    

    
    

   
class IDSDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        
    def __getitem__(self, idx):
        enhancer_sequence, promoter_sequence, label = self.dataset[idx]
        
        
        enhancer_sequence_encoded = self.get_token_ids(enhancer_sequence)
        promoter_sequence_encoded = self.get_token_ids(promoter_sequence)
        

        '''
        enhancer_sequence_encoded = self.one_hot_encoding(enhancer_sequence)
        promoter_sequence_encoded = self.one_hot_encoding(promoter_sequence)
        '''

        return enhancer_sequence_encoded.squeeze(), promoter_sequence_encoded.squeeze(), label
        
        #return torch.unsqueeze(enhancer_sequence_encoded, 0), torch.unsqueeze(promoter_sequence_encoded, 0), label
        #return enhancer_sequence_encoded, promoter_sequence_encoded, label
        
    
    def __len__(self):
        return len(self.dataset)
    
    
    def get_token_ids(self, sequence):
        token_dict = self.get_tokenizer()
        
        k = 6
        token_6mers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        token_6mers = ["null" if 'N' in mer else mer for mer in token_6mers]
        #token_6mers.append('null')
        token_ids = [token_dict[mer] for mer in token_6mers]
             
        
        #二维卷积需要有通道维度
        #return torch.tensor(token_ids).reshape(-1, len(token_ids))
        
        return torch.tensor(token_ids)
            
    def get_tokenizer(self):
        f= ['A','C','G','T']
        c = itertools.product(f,f,f,f,f,f)
        res=[]
        for i in c:
            temp=i[0] + i[1] + i[2] + i[3] + i[4] + i[5]
            res.append(temp)
            token_dict = {word: index + 1 for index, word in enumerate(res)}
            
            token_dict['null'] = 0
        return token_dict
    
    def one_hot_encoding(self, sequence):
        encoding = torch.zeros(len(sequence), 4)
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, base in enumerate(sequence):
            if base == 'N': continue
            encoding[i][base_to_index[base]] = 1
        return encoding
    
    
class BalancedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.positive_indices = self._get_positive_indices()
        self.negative_indices = self._get_negative_indices()

    def __getitem__(self, index):
        # 根据索引获取样本
        if index < len(self.negative_indices):
            return self.dataset[self.negative_indices[index]]
        elif index >= len(self.negative_indices) and index < len(self.dataset):
            positive_index = self.positive_indices[index - len(self.negative_indices)]
            return self.dataset[positive_index]
        else:
            positive_index = self.positive_indices[(index - len(self.negative_indices)) % len(self.positive_indices)]
            time = (index - len(self.negative_indices)) // len(self.positive_indices)
            k = time * 20
            enhaner, promoter, label = self.dataset[positive_index]
            enhaner = enhaner[k:] + enhaner[:k]
            promoter = promoter[k:] + promoter[:k]
            return enhaner, promoter, label
            

    def __len__(self):
        return 2 * len(self.negative_indices)

    def _get_positive_indices(self):
        # 获取正样本的索引
        positive_indices = []
        for i in range(len(self.dataset)):
            _, _, label = self.dataset[i]
            if label == 1:  # 假设正样本的标签为1
                positive_indices.append(i)
        return positive_indices

    def _get_negative_indices(self):
        # 获取负样本的索引
        negative_indices = []
        for i in range(len(self.dataset)):
            _, _, label = self.dataset[i]
            if label == 0:  # 假设负样本的标签为0
                negative_indices.append(i)
        return negative_indices