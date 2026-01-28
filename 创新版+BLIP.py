import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  
from torchvision import transforms
from transformers import BlipProcessor, BlipModel, logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold  
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import time
from 创新版数据读取 import load_train_valid_test
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 统计模型参数量
def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*50)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数量：{trainable_params:,}")
    print("="*50)
    return total_params, trainable_params

# 自定义包装层
class LayerWithDropout(nn.Module):
    def __init__(self, original_layer, dropout_prob=0.1):
        super().__init__()
        self.original_layer = original_layer  
        self.dropout = nn.Dropout(dropout_prob)  
        for param in self.original_layer.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            outputs = (self.dropout(outputs[0]),) + outputs[1:]
        else:
            outputs = self.dropout(outputs)
        return outputs


# BLIP 的数据集类
class BLIPMultiModalDataset(Dataset):
    def __init__(self, data, blip_processor, image_transform=None):
        self.data = data
        self.blip_processor = blip_processor
        self.label2idx = {'positive':0, 'neutral':1, 'negative':2}
        self.idx2label = {v: k for k, v in self.label2idx.items()}  
        self.text_max_len = 32  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            dummy_image = torch.randn(3, 512, 512)
            dummy_pil = transforms.ToPILImage()(dummy_image)
            dummy_processed = self.blip_processor(
                images=dummy_pil,
                text="",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.text_max_len
            )
            return {
                'image': dummy_processed['pixel_values'].squeeze(0),
                'label': torch.tensor(0, dtype=torch.long),
                'guid': f'guid_{idx}',
                'input_ids': dummy_processed['input_ids'].squeeze(0),
                'attention_mask': dummy_processed['attention_mask'].squeeze(0)
            }
        
        sample = self.data[idx]
        
        # 提取图文数据（测试集/验证集/训练集完全一致）
        text = sample.get('txt_content', '')
        img_pil = sample.get('jpg_content')
        
        # BLIP 集成处理图文
        blip_encoding = self.blip_processor(
            images=img_pil,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_len
        )
        
        # 提取处理后的张量
        pixel_values = blip_encoding['pixel_values'].squeeze(0)
        input_ids = blip_encoding['input_ids'].squeeze(0)
        attention_mask = blip_encoding['attention_mask'].squeeze(0)
        
        # 标签处理
        tag_value = sample.get('tag')
        if tag_value not in self.label2idx:
            label = torch.tensor(0, dtype=torch.long)  # 测试集无标签，返回占位符
        else:
            label = torch.tensor(self.label2idx[tag_value], dtype=torch.long)  # 训练/验证集正常映射
        
        return {
            'image': pixel_values,
            'label': label,  
            'guid': sample.get('guid', f'guid_{idx}'),
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


# BLIP 多模态融合模型
class BLIPFusionModel(nn.Module):
    def __init__(
        self,
        local_blip_path="./local_blip_model", 
        num_classes=3,  
        fusion_mode="add", 
        dropout_prob=0.3,  
        hidden_layer_dim=256,  
        unfreeze_vision_layers=2,  
        unfreeze_text_layers=2,
        unfreeze_layer_dropout=0.1 
    ):
        super().__init__()
        self.blip = BlipModel.from_pretrained(local_blip_path, cache_dir=None)
        self.fusion_mode = fusion_mode
        self.unfreeze_layer_dropout = unfreeze_layer_dropout  
        
        self.blip_feature_dim = self._get_actual_feature_dim()
        if not hasattr(self.__class__, 'dim_printed'):
            print(f"动态获取 BLIP 特征维度：{self.blip_feature_dim}")
            self.__class__.dim_printed = True
        
        self._freeze_blip_layers(unfreeze_vision_layers, unfreeze_text_layers)
        
        # 融合分类头
        if fusion_mode == "add":
            fusion_input_dim = self.blip_feature_dim
        elif fusion_mode == "concat":
            fusion_input_dim = self.blip_feature_dim * 2
        else:
            raise ValueError("融合方式仅支持 'add' 或 'concat'")
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_layer_dim, num_classes)
        )

    def _get_actual_feature_dim(self):
        dummy_pixel_values = torch.randn(1, 3, 224, 224)  
        dummy_input_ids = torch.randint(0, 100, (1, 32)).long()  
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        
        with torch.no_grad():
            outputs = self.blip(
                pixel_values=dummy_pixel_values,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                return_dict=True
            )
            return outputs.image_embeds.shape[-1]
            
    def _freeze_blip_layers(self, unfreeze_vision_layers, unfreeze_text_layers):
        # 先全冻结 BLIP 所有参数
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # 视觉编码器解冻 + Dropout
        vision_model = self.blip.vision_model
        vision_layers = vision_model.encoder.layers
        total_vision_layers = len(vision_layers)
        
        vision_unfreeze_start = max(0, total_vision_layers - unfreeze_vision_layers)
        if not hasattr(self.__class__, 'layer_printed'):
            print(f"视觉层解冻范围：索引 {vision_unfreeze_start} ~ {total_vision_layers-1}（共 {unfreeze_vision_layers} 层），添加Dropout概率 {self.unfreeze_layer_dropout}")
        
        for i, layer in enumerate(vision_layers):
            if i >= vision_unfreeze_start:
                vision_layers[i] = LayerWithDropout(
                    original_layer=layer,
                    dropout_prob=self.unfreeze_layer_dropout
                )
        
        if hasattr(self.blip, 'vision_proj'):
            for param in self.blip.vision_proj.parameters():
                param.requires_grad = True
            self.blip.vision_proj = nn.Sequential(
                self.blip.vision_proj,
                nn.Dropout(self.unfreeze_layer_dropout)
            )
        
        # 文本编码器解冻 + Dropout
        total_text_layers = 0
        text_unfreeze_start = 0
        text_model = self.blip.text_model
        if not hasattr(text_model, 'encoder'):
            raise AttributeError("text_model 无 encoder 属性")
        
        text_encoder = text_model.encoder
        text_layers = text_encoder.layer  
        total_text_layers = len(text_layers)
        
        if total_text_layers == 0:
            raise AttributeError("text_encoder.layer 为空")
        text_unfreeze_start = max(0, total_text_layers - unfreeze_text_layers)
        
        if not hasattr(self.__class__, 'layer_printed'):
            print(f"文本层解冻范围：索引 {text_unfreeze_start} ~ {total_text_layers-1}（共 {unfreeze_text_layers} 层），添加Dropout概率 {self.unfreeze_layer_dropout}")
            print(f"成功：BLIP 文本编码器解冻 {unfreeze_text_layers} 层（总层数 {total_text_layers}，起始索引 {text_unfreeze_start}）")
            print(f"BLIP 层冻结信息：")
            print(f"  视觉编码器：总层数 {total_vision_layers} | 解冻最后 {unfreeze_vision_layers} 层（索引 {vision_unfreeze_start} ~ {total_vision_layers-1}）| 解冻层Dropout：{self.unfreeze_layer_dropout}")
            print(f"  文本编码器：总层数 {total_text_layers} | 解冻最后 {unfreeze_text_layers} 层（索引 {text_unfreeze_start} ~ {total_text_layers-1}）| 解冻层Dropout：{self.unfreeze_layer_dropout}")
            self.__class__.layer_printed = True

        # 解冻指定的最后 N 层文本层
        for i, layer in enumerate(text_layers):
            if i >= text_unfreeze_start:
                text_layers[i] = LayerWithDropout(
                    original_layer=layer,
                    dropout_prob=self.unfreeze_layer_dropout
                )
        
        # 解冻文本投影层
        if hasattr(self.blip, 'text_proj'):
            for param in self.blip.text_proj.parameters():
                param.requires_grad = True
            # 包装文本投影层
            self.blip.text_proj = nn.Sequential(
                self.blip.text_proj,
                nn.Dropout(self.unfreeze_layer_dropout)
            )
        elif hasattr(self.blip, 'text_projection'):
            for param in self.blip.text_projection.parameters():
                param.requires_grad = True
            # 包装文本投影层
            self.blip.text_projection = nn.Sequential(
                self.blip.text_projection,
                nn.Dropout(self.unfreeze_layer_dropout)
            )

    def forward(self, pixel_values, input_ids, attention_mask):
        # BLIP 提取图文特征
        blip_outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 直接从 BlipOutput 提取图文特征
        image_embeds = blip_outputs.image_embeds
        text_embeds = blip_outputs.text_embeds
    
        # add 融合前做L2归一化，对齐图文特征分布
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        
        # 多模态融合
        if self.fusion_mode == "add":
            fused_embeds = image_embeds + text_embeds
        else:
            fused_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 分类预测：直接返回 logits
        logits = self.classifier(fused_embeds)
        return logits

# 单折训练评估函数
def train_single_fold(
    fold_idx, model, train_loader, valid_loader, criterion, optimizer, scheduler, device,
    epochs, patience, exp_name
):
    best_acc = 0.0
    history = {'train_losses': [], 'valid_losses': [], 'valid_accs': []}
    no_improvement_count = 0

    print(f"\n===== 开始训练第 {fold_idx+1} 折 =====")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        pbar_train = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[{exp_name}] Fold {fold_idx+1} Epoch {epoch+1:02d} (Train)",
            leave=False
        )

        for batch_idx, batch in pbar_train:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar_train.set_postfix(batch_loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # 验证阶段
        model.eval()
        total_valid_loss = 0.0
        all_logits = []
        all_labels = []
        pbar_valid = tqdm(
            enumerate(valid_loader),
            total=len(valid_loader),
            desc=f"[{exp_name}] Fold {fold_idx+1} Epoch {epoch+1:02d} (Valid)",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in pbar_valid:
                pixel_values = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(pixel_values, input_ids, attention_mask)
                batch_loss = criterion(logits, labels).item()
                total_valid_loss += batch_loss

                all_logits.append(logits)
                all_labels.append(labels)
                pbar_valid.set_postfix(batch_loss=f"{batch_loss:.4f}")

        # 计算验证指标
        avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        if len(all_logits) > 0 and len(all_labels) > 0:
            all_logits = torch.cat(all_logits, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
            all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
            valid_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
        else:
            valid_acc = 0.0

        # 记录历史
        history['train_losses'].append(avg_train_loss)
        history['valid_losses'].append(avg_valid_loss)
        history['valid_accs'].append(valid_acc)

        # 训练轮结束后，更新学习率
        scheduler.step(avg_valid_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{exp_name}] Fold {fold_idx+1} Epoch {epoch+1}/{epochs} | 学习率：{current_lr:.6f} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_valid_loss:.4f} | 验证准确率：{valid_acc:.4f}")

        # 早停
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'best_blip_{exp_name}_fold_{fold_idx+1}.pth')
            no_improvement_count = 0
            print(f"[{exp_name}] Fold {fold_idx+1} 更新最佳模型，当前最佳准确率：{best_acc:.4f}")
        else:
            no_improvement_count += 1
            print(f"[{exp_name}] Fold {fold_idx+1} 早停计数：{no_improvement_count}/{patience}")

        if no_improvement_count > patience:
            print(f"[{exp_name}] Fold {fold_idx+1} 早停触发，终止训练")
            break

    # 加载当前折最佳模型
    model.load_state_dict(torch.load(f'best_blip_{exp_name}_fold_{fold_idx+1}.pth'))
    print(f"===== 第 {fold_idx+1} 折训练结束，最佳验证准确率：{best_acc:.4f} =====")
    return best_acc, history, model

# 绘制单折训练曲线
def plot_fold_training_curve(history, model_name, fold_idx):
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['valid_losses'], 'r--', label='验证损失')
    ax1.set_title(f'{model_name} - 第 {fold_idx+1} 折 损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0,2)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, history['valid_accs'], 'r-', label='验证准确率')
    ax2.set_title(f'{model_name} - 第 {fold_idx+1} 折 验证准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"fig_{fold_idx}")


# 计算单折F1并提取错误样例
def calculate_fold_class_f1(model, valid_loader, device, idx2label, exp_name, valid_data, fold_idx):
    model.eval()
    all_labels = []
    all_preds = []
    all_sample_indices = []  
    
    with torch.no_grad():
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"计算 {exp_name} 第 {fold_idx+1} 折 最佳模型类别F1")
        for batch_idx, batch in pbar:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播获取logits
            logits = model(pixel_values, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            neutral_threshold = 0.5
            for i in range(len(probs)):
                neutral_prob = probs[i][1]
                if neutral_prob > neutral_threshold:
                    preds[i] = 1  # 1是neutral的索引
            
            # 收集真实标签、预测标签、样本索引（关联原始数据）
            batch_size = labels.shape[0]
            start_idx = batch_idx * valid_loader.batch_size
            end_idx = start_idx + batch_size
            sample_indices = list(range(start_idx, end_idx))[:batch_size]  # 防止超出总长度
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_sample_indices.extend(sample_indices)
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_sample_indices = np.array(all_sample_indices)
    
    # 提取错误样例
    error_mask = (all_labels != all_preds)
    error_indices = all_sample_indices[error_mask]  
    error_true_labels = all_labels[error_mask]
    error_pred_labels = all_preds[error_mask]
    
    # 构建错误样例列表
    error_samples = []
    for idx, true_label_idx, pred_label_idx in zip(error_indices, error_true_labels, error_pred_labels):
        if idx < len(valid_data):
            sample = valid_data[idx]
            img_pil = sample.get('jpg_content')
            text = sample.get('txt_content', '')
            true_label = idx2label[true_label_idx]
            pred_label = idx2label[pred_label_idx]
            error_samples.append({
                'image': img_pil,
                'text': text,
                'true_label': true_label,
                'pred_label': pred_label,
                'index': idx
            })
    
    class_f1 = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*80)
    print(f"最佳模型 {exp_name} 第 {fold_idx+1} 折 验证集详细类别评估报告")
    print("="*80)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[idx2label[i] for i in sorted(idx2label.keys())],
        digits=4  
    ))
    print(f"宏观F1（平等加权，关注少数类别）：{macro_f1:.4f}")
    print(f"加权F1（按样本量加权，贴近准确率）：{weighted_f1:.4f}")
    print(f"各类别F1单独展示：")
    for idx, cls_name in idx2label.items():
        print(f"  {cls_name}：{class_f1[idx]:.4f}")
    print(f"第 {fold_idx+1} 折 验证集错误样例总数：{len(error_samples)}")
    print("="*80 + "\n")
    
    return class_f1, macro_f1, weighted_f1, error_samples

# 模型融合
def load_ensemble_models(n_folds, device, model_params):
    models = []
    for fold_num in range(1, n_folds+1):
        model = BLIPFusionModel(**model_params).to(device)
        model_path = f"best_blip_blip_cross_validation_fold_{fold_num}.pth"
        model.load_state_dict(torch.load(model_path))
        model.eval()  
        models.append(model)
    print(f"\n=====================================")
    print(f"已加载 {n_folds} 折模型，准备进行融合预测")
    print(f"=====================================")
    return models

@torch.no_grad()
def ensemble_predict(models, pixel_values, input_ids, attention_mask, device, neutral_threshold=0.25):
    all_probs = []
    pixel_values = pixel_values.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    
    if isinstance(models, list):
        for model in models:
            logits = model(pixel_values, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
    else:
        logits = models(pixel_values, input_ids, attention_mask)
        avg_probs = torch.softmax(logits, dim=1).cpu()  
    
    # 沿用优化后的后处理逻辑
    preds = torch.argmax(avg_probs, dim=1)
    for i in range(len(avg_probs)):
        positive_prob = avg_probs[i][0]
        negative_prob = avg_probs[i][2]
        if positive_prob < 0.05 and negative_prob < 0.05:
            preds[i] = 1
    
    return preds, avg_probs

def evaluate_on_original_valid(model, is_ensemble, original_valid_data, blip_processor, device, batch_size=16, neutral_threshold=0.25):
    valid_dataset = BLIPMultiModalDataset(original_valid_data, blip_processor)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    all_labels = []
    all_preds = []
    
    if not is_ensemble:
        model.eval()
    
    with torch.no_grad():
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="在原始valid集上评估")
        for batch_idx, batch in pbar:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if not is_ensemble:
                # 单模型预测
                logits = model(pixel_values, input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                
                # neutral后处理
                for i in range(len(probs)):
                    if probs[i][1] > neutral_threshold:
                        preds[i] = 1
            else:
                # 融合模型预测
                preds, _ = ensemble_predict(model, pixel_values, input_ids, attention_mask, device, neutral_threshold)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*100)
    print(f"原始valid集 评估结果汇总")
    print("="*100)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[valid_dataset.idx2label[i] for i in sorted(valid_dataset.idx2label.keys())],
        digits=4
    ))
    print(f"准确率（Accuracy）：{acc:.4f}")
    print(f"宏观F1（Macro F1）：{macro_f1:.4f}")
    print(f"加权F1（Weighted F1）：{weighted_f1:.4f}")
    print(f"各类别F1单独展示：")
    for idx, cls_name in valid_dataset.idx2label.items():
        print(f"  {cls_name}：{class_f1[idx]:.4f}")
    print("="*100 + "\n")
    
    return acc, class_f1, macro_f1, weighted_f1



# 测试集预测并输出指定格式结果
def predict_test_set(ensemble_models, test_data, blip_processor, device, batch_size=16, neutral_threshold=0.25):
    test_dataset = BLIPMultiModalDataset(test_data, blip_processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False  # 保留最后一个不完整批次，避免丢失样本
    )
    
    test_results = [] 
    
    # 融合模型批量预测
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="用融合模型预测测试集")
        for batch_idx, batch in pbar:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            guids = batch['guid']  
            
            # 调用融合预测函数
            preds, _ = ensemble_predict(
                models=ensemble_models,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                device=device,
                neutral_threshold=neutral_threshold
            )
            
            # 转换预测结果（idx -> tag）
            for guid, pred_idx in zip(guids, preds):
                pred_tag = test_dataset.idx2label[pred_idx.item()]
                test_results.append( (guid, pred_tag) )
    
    return test_results

def output_test_results(test_results, save_to_file=True, filename="test_predict_results.csv"):
    result_lines = [f"{guid},{tag}" for guid, tag in test_results]
    
    if save_to_file:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("guid,tag\n")
            f.write("\n".join(result_lines))
        print(f"\n预测结果已保存到文件：{filename}")
    
    return result_lines

def merge_ensemble_models_weights(n_folds, device, model_params):
    # 加载第一个模型作为基础
    model_base = BLIPFusionModel(**model_params).to(device)
    model_path_1 = f"best_blip_blip_cross_validation_fold_1.pth"
    weights_base = torch.load(model_path_1, map_location=device)
    
    # 遍历其他模型，累加权重
    for fold_num in range(2, n_folds+1):
        model_path = f"best_blip_blip_cross_validation_fold_{fold_num}.pth"
        weights_current = torch.load(model_path, map_location=device)
        # 权重累加
        for key in weights_base.keys():
            weights_base[key] += weights_current[key]
    
    # 权重取平均
    for key in weights_base.keys():
        weights_base[key] = weights_base[key] / n_folds
    
    # 加载平均权重到基础模型
    model_base.load_state_dict(weights_base)
    model_base.eval()
    print(f"\n=====================================")
    print(f"已合并 {n_folds} 折模型权重，生成单模型")
    print(f"=====================================")
    return model_base



if __name__ == "__main__":
    # 基础训练参数
    BATCH_SIZE = 16
    EPOCHS = 18   #18
    PATIENCE = 3
    LEARNING_RATE = 1.2e-5   #1.2e-5
    NUM_WORKERS = 2
    LABEL_SMOOTHING = 0.05
    N_FOLDS = 5  # 
    p = 1  #1
    m = 1  #1
    n = 1  #1
    
    
    # 模型结构参数
    model_params = {
        'local_blip_path': "./local_blip_model",
        'num_classes': 3,
        'fusion_mode': "add",
        'dropout_prob': 0.4, 
        'hidden_layer_dim': 256,
        'unfreeze_vision_layers': 2,
        'unfreeze_text_layers': 2,
        'unfreeze_layer_dropout': 0.4  
    }
    
    # BLIP 处理器参数
    USE_FAST_PROCESSOR = False
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备：{device}")
    print(f"当前实验参数（交叉验证 {N_FOLDS} 折）：融合方式={model_params['fusion_mode']} | 解冻视觉层={model_params['unfreeze_vision_layers']} | 解冻文本层={model_params['unfreeze_text_layers']} | 解冻层Dropout={model_params['unfreeze_layer_dropout']} | 学习率={LEARNING_RATE} | 批次大小={BATCH_SIZE} | 标签平滑={LABEL_SMOOTHING}")
    
    # 加载数据并保存原始valid_data
    train_data, original_valid_data, test_data = load_train_valid_test(picture="PIL")
    full_data = train_data + original_valid_data  # 合并为完整数据集用于最终交叉验证
    print(f"数据加载完成：原始训练集{len(train_data)}条 | 原始验证集{len(original_valid_data)}条 | 合并后总数据集{len(full_data)}条 | 测试集{len(test_data)}条")

    # 提取完整数据集的标签（用StratifiedKFold分层抽样）
    label2idx = {'positive':0, 'neutral':1, 'negative':2}
    full_labels = []
    for sample in full_data:
        tag = sample.get('tag', 'positive')
        full_labels.append(label2idx[tag])
    full_labels = np.array(full_labels)
    
    # BLIP处理器
    blip_processor = BlipProcessor.from_pretrained(model_params['local_blip_path'], cache_dir=None, use_fast=USE_FAST_PROCESSOR)

    # 分层交叉验证器
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)  

    # 结果记录列表
    fold_accs = []
    fold_macro_f1s = []
    fold_weighted_f1s = []
    fold_class_f1s = []  # 记录每折的各类别F1

    # 遍历每折进行训练和评估
    start = time.time()
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(full_data, full_labels)):

        # 分割当前折的训练集和验证集
        fold_train_data = [full_data[i] for i in train_idx]
        fold_valid_data = [full_data[i] for i in valid_idx]
        print(f"\n=====================================")
        print(f"第 {fold_idx+1} 折：训练集{len(fold_train_data)}条 | 验证集{len(fold_valid_data)}条")
        print(f"=====================================")

        # 训练集：使用加权采样解决类别不平衡
        fold_train_dataset = BLIPMultiModalDataset(fold_train_data, blip_processor)
        fold_valid_dataset = BLIPMultiModalDataset(fold_valid_data, blip_processor)

        # 构建训练集加权采样器
        fold_train_labels = [label2idx[sample.get('tag', 'positive')] for sample in fold_train_data]
        sample_counts = np.array([1910, 335, 955], dtype=np.float32)  
        class_weights_sampler = 1.0 / sample_counts
        class_weights_sampler = class_weights_sampler / class_weights_sampler.min()
        class_weights_sampler[1] = min(class_weights_sampler[1], 3.0)
        class_weights_sampler[2] = class_weights_sampler[2] * n  
        class_weights_sampler[0] = class_weights_sampler[0] * p
        class_weights_sampler[1] = class_weights_sampler[1] * m
        print(f"权重：{class_weights_sampler}")

        fold_train_sample_weights = [class_weights_sampler[label] for label in fold_train_labels]

        sampler = WeightedRandomSampler(
            weights=fold_train_sample_weights,
            num_samples=len(fold_train_data),
            replacement=False
        )

        # 构建DataLoader
        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )

        fold_valid_loader = DataLoader(
            fold_valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2
        )

        # 初始化当前折的模型（每次折都重新初始化，避免跨折参数污染）
        model = BLIPFusionModel(**model_params).to(device)

        # 统计参数量（仅第一折打印）
        if fold_idx == 0:
            count_model_parameters(model)

        # 定义损失函数、优化器、学习率调度器
        class_sample_counts = torch.tensor([(full_labels==0).sum(), (full_labels==1).sum(), (full_labels==2).sum()], dtype=torch.float32)
        class_weights = (1.0 / class_sample_counts) * (len(full_data) / 3)
        class_weights = class_weights / class_weights.min() * 1.2
        class_weights[1] = min(class_weights[1], 8.0)
        class_weights = torch.clamp(class_weights, max=10.0).to(device)
        class_weights[2] = class_weights[2] * n
        class_weights[0] = class_weights[0] * p
        class_weights_sampler[1] = class_weights_sampler[1] * m

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING
        )

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=3,
            min_lr=1e-7
        )

        # 训练当前折
        exp_name = "blip_cross_validation"
        best_acc, fold_history, best_model = train_single_fold(
            fold_idx, model, fold_train_loader, fold_valid_loader, criterion, optimizer, scheduler,
            device, EPOCHS, PATIENCE, exp_name
        )

        # 绘制当前折训练曲线
        plot_fold_training_curve(fold_history, exp_name, fold_idx)

        # 计算当前折F1分数和错误样例
        class_f1, macro_f1, weighted_f1, error_samples = calculate_fold_class_f1(
            best_model, fold_valid_loader, device, fold_valid_dataset.idx2label,
            exp_name, fold_valid_data, fold_idx
        )

        # 记录当前折结果
        fold_accs.append(best_acc)
        fold_macro_f1s.append(macro_f1)
        fold_weighted_f1s.append(weighted_f1)
        fold_class_f1s.append(class_f1)
    finish = time.time()
    print(f"训练时间{finish-start}s")

    # 汇总交叉验证结果
    print("\n" + "="*100)
    print(f"交叉验证（{N_FOLDS} 折）结果汇总")
    print("="*100)
    print(f"各折最佳准确率：{[f'{acc:.4f}' for acc in fold_accs]}")
    print(f"平均准确率：{np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"\n各折宏观F1：{[f'{f1:.4f}' for f1 in fold_macro_f1s]}")
    print(f"平均宏观F1：{np.mean(fold_macro_f1s):.4f} ± {np.std(fold_macro_f1s):.4f}")
    print(f"\n各折加权F1：{[f'{f1:.4f}' for f1 in fold_weighted_f1s]}")
    print(f"平均加权F1：{np.mean(fold_weighted_f1s):.4f} ± {np.std(fold_weighted_f1s):.4f}")

    # 汇总各类别平均F1
    fold_class_f1s = np.array(fold_class_f1s)
    avg_positive_f1 = np.mean(fold_class_f1s[:, 0])
    avg_neutral_f1 = np.mean(fold_class_f1s[:, 1])
    avg_negative_f1 = np.mean(fold_class_f1s[:, 2])
    print(f"\n各类别平均F1：")
    print(f"  positive：{avg_positive_f1:.4f} ± {np.std(fold_class_f1s[:, 0]):.4f}")
    print(f"  neutral：{avg_neutral_f1:.4f} ± {np.std(fold_class_f1s[:, 1]):.4f}")
    print(f"  negative：{avg_negative_f1:.4f} ± {np.std(fold_class_f1s[:, 2]):.4f}")
    print("="*100 + "\n")

    # 加载所有折模型进行融合
    # final_model = load_ensemble_models(N_FOLDS, device, model_params)
    
    final_model =merge_ensemble_models_weights(5, device, model_params)
    
    # acc, class_f1, macro_f1, weighted_f1 = evaluate_on_original_valid(
    #     model=final_model,
    #     is_ensemble=True,
    #     original_valid_data=original_valid_data,
    #     blip_processor=blip_processor,
    #     device=device,
    #     batch_size=BATCH_SIZE
    # )


    test_pred_results = predict_test_set(
        ensemble_models=final_model,
        test_data=test_data,
        blip_processor=blip_processor,
        device=device,
        batch_size=BATCH_SIZE
    )
    
    # 输出指定格式结果，并保存到文件
    results = output_test_results(test_pred_results)
