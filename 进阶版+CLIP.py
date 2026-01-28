import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics import accuracy_score, classification_report  
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from 数据读取 import load_train_valid_test
import time

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

# 适配 CLIP 的数据集类
class CLIPMultiModalDataset(Dataset):
    def __init__(self, data, clip_processor, clip_tokenizer, image_transform=None):
        self.data = data
        self.clip_processor = clip_processor  # CLIP 图像处理器
        self.clip_tokenizer = clip_tokenizer  # CLIP 文本分词器
        self.label2idx = {'positive':0, 'neutral':1, 'negative':2}
        self.text_max_len = 77  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            return {
                'image': torch.randn(3, 512, 512),
                'label': torch.tensor(0, dtype=torch.long),
                'guid': f'guid_{idx}',
                'input_ids': torch.zeros(77, dtype=torch.long),
                'attention_mask': torch.zeros(77, dtype=torch.long)
            }
        
        sample = self.data[idx]
        
        # 文本处理
        text = sample.get('txt_content', '')
        text_encoding = self.clip_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.text_max_len,
            return_tensors='pt'
        )
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        # 图像处理
        img_pil = sample.get('jpg_content') 
        if self.clip_processor is not None:
            img_processed = self.clip_processor(
                images=img_pil,
                return_tensors="pt"
            )['pixel_values'].squeeze(0)
        
        # 标签处理
        if sample.get('tag') is None:
            label = torch.tensor(0, dtype=torch.long)
        else:
            label = self.label2idx[sample['tag']]
            label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': img_processed,
            'label': label,
            'guid': sample.get('guid', f'guid_{idx}'),
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# CLIP 多模态融合模型
class CLIPFusionModel(nn.Module):
    def __init__(
        self,
        local_clip_path="./local_clip_model",  
        num_classes=3,  
        fusion_mode="add",  
        dropout_prob=0.3,  
        hidden_layer_dim=256,  
        unfreeze_vision_layers=0, 
        unfreeze_text_layers=0     
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(local_clip_path, cache_dir=None)
        self.clip_feature_dim = self.clip.config.projection_dim 
        self.fusion_mode = fusion_mode
        
        self._freeze_clip_layers(unfreeze_vision_layers, unfreeze_text_layers)
        
        # 定义融合分类头
        if fusion_mode == "add":
            fusion_input_dim = self.clip_feature_dim
        elif fusion_mode == "concat":
            fusion_input_dim = self.clip_feature_dim * 2
        else:
            raise ValueError("融合方式仅支持 'add' 或 'concat'")
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_layer_dim, num_classes)
        )

    def _freeze_clip_layers(self, unfreeze_vision_layers, unfreeze_text_layers):
        for param in self.clip.parameters():
            param.requires_grad = False
        
        vision_layers = self.clip.vision_model.encoder.layers  
        total_vision_layers = len(vision_layers)
        
        vision_unfreeze_start = max(0, total_vision_layers - unfreeze_vision_layers)
        for i, layer in enumerate(vision_layers):
            if i >= vision_unfreeze_start:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        
        text_layers = self.clip.text_model.encoder.layers  
        total_text_layers = len(text_layers)
        
        text_unfreeze_start = max(0, total_text_layers - unfreeze_text_layers)
        for i, layer in enumerate(text_layers):
            if i >= text_unfreeze_start:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 文本投影层
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
        
        print(f"CLIP 层冻结信息：")
        print(f"  视觉编码器：总层数 {total_vision_layers} | 解冻最后 {unfreeze_vision_layers} 层（索引 {vision_unfreeze_start} ~ {total_vision_layers-1}）")
        print(f"  文本编码器：总层数 {total_text_layers} | 解冻最后 {unfreeze_text_layers} 层（索引 {text_unfreeze_start} ~ {total_text_layers-1}）")

    def forward(self, pixel_values, input_ids, attention_mask):
        # CLIP 提取图文特征
        clip_outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_embeds = clip_outputs.image_embeds
        text_embeds = clip_outputs.text_embeds
        
        # 多模态融合
        if self.fusion_mode == "add":
            fused_embeds = image_embeds + text_embeds
        else:
            fused_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # 分类预测
        logits = self.classifier(fused_embeds)
        probs = torch.softmax(logits, dim=1)
        
        return probs

# 训练评估函数
def run_clip_exp(
    model, train_loader, valid_loader, criterion, optimizer, device,
    epochs, patience, exp_name
):
    best_acc = 0.0
    history = {'train_losses': [], 'valid_losses': [], 'valid_accs': []}
    no_improvement_count = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        pbar_train = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[{exp_name}] Epoch {epoch+1:02d} (Train)",
            leave=False
        )

        for batch_idx, batch in pbar_train:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            probs = model(pixel_values, input_ids, attention_mask)
            loss = criterion(torch.log(probs), labels)

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
        all_probs = []
        all_labels = []
        pbar_valid = tqdm(
            enumerate(valid_loader),
            total=len(valid_loader),
            desc=f"[{exp_name}] Epoch {epoch+1:02d} (Valid)",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, batch in pbar_valid:
                pixel_values = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                probs = model(pixel_values, input_ids, attention_mask)
                batch_loss = criterion(torch.log(probs), labels).item()
                total_valid_loss += batch_loss

                all_probs.append(probs)
                all_labels.append(labels)
                pbar_valid.set_postfix(batch_loss=f"{batch_loss:.4f}")

        # 计算验证指标
        avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        if len(all_probs) > 0 and len(all_labels) > 0:
            all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
            valid_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
        else:
            valid_acc = 0.0

        history['train_losses'].append(avg_train_loss)
        history['valid_losses'].append(avg_valid_loss)
        history['valid_accs'].append(valid_acc)

        print(f"[{exp_name}] Epoch {epoch+1}/{epochs} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_valid_loss:.4f} | 验证准确率：{valid_acc:.4f}")

        # 早停逻辑
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'best_clip_{exp_name}.pth')
            no_improvement_count = 0
            print(f"[{exp_name}] 更新最佳模型，当前最佳准确率：{best_acc:.4f}")
        else:
            no_improvement_count += 1
            print(f"[{exp_name}] 早停计数：{no_improvement_count}/{patience}")

        if no_improvement_count >= patience:
            print(f"[{exp_name}] 早停触发，终止训练")
            break

    # 绘制训练曲线
    plot_training_curve(history, exp_name)

    model.load_state_dict(torch.load(f'best_clip_{exp_name}.pth'))
    return best_acc, history

# 绘制训练曲线
def plot_training_curve(history, model_name):
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['valid_losses'], 'r--', label='验证损失')
    ax1.set_title(f'{model_name} - 损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 2.0)
    ax1.legend()
    
    ax2.plot(epochs, history['valid_accs'], 'r-', label='验证准确率')
    ax2.set_title(f'{model_name} - 验证准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}")

# 评估函数
def evaluate_test_set(model, test_loader, device, target_names=['positive', 'neutral', 'negative']):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        pbar_test = tqdm(enumerate(test_loader), total=len(test_loader), desc="测试集评估")
        for batch_idx, batch in pbar_test:
            pixel_values = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            probs = model(pixel_values, input_ids, attention_mask)
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            true = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_true.extend(true)
    print("\n" + "="*80)
    print("分类报告（Precision/Recall/F1-Score）：")
    print("="*80)
    print(classification_report(all_true, all_preds, target_names=target_names, digits=4))
    test_acc = accuracy_score(all_true, all_preds)
    print(f"整体准确率：{test_acc:.4f}")
    print("="*80)
    return test_acc, all_true, all_preds

# 主函数
if __name__ == "__main__":
    # 基础训练参数
    BATCH_SIZE = 16  # 
    EPOCHS = 18  # 
    PATIENCE = 4  
    LEARNING_RATE = 5e-5  
    NUM_WORKERS = 2  
    
    # 模型结构参数
    LOCAL_CLIP_PATH = "./local_clip_model"  
    NUM_CLASSES = 3  
    FUSION_MODE = "add"  # 
    DROPOUT_PROB = 0.5  # 
    HIDDEN_LAYER_DIM = 256  # 
    UNFREEZE_VISION_LAYERS = 2  # 解冻视觉编码器最后 N 层：0（全冻）/2/4/12（全解，视觉层共12层）
    UNFREEZE_TEXT_LAYERS = 2    # 解冻文本编码器最后 N 层：0（全冻）/2/4/12（全解，文本层共12层）
    
    # CLIP 处理器参数
    USE_FAST_PROCESSOR = False  # 是否使用快速处理器：True/False
    
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_data, valid_data, test_data = load_train_valid_test(picture="PIL")
    print(f"数据加载完成：训练集{len(train_data)}条 | 验证集{len(valid_data)}条 | 测试集{len(test_data)}条")

    # 初始化 CLIP 处理器和分词器
    clip_processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_PATH, cache_dir=None, use_fast=USE_FAST_PROCESSOR)
    clip_tokenizer = CLIPTokenizer.from_pretrained(LOCAL_CLIP_PATH, cache_dir=None)

    # 构建数据集和数据加载器
    train_dataset = CLIPMultiModalDataset(train_data, clip_processor, clip_tokenizer)
    valid_dataset = CLIPMultiModalDataset(valid_data, clip_processor, clip_tokenizer)
    test_dataset = CLIPMultiModalDataset(test_data, clip_processor, clip_tokenizer)  # 新增测试集数据集
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,  prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,  prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,  prefetch_factor=2)  # 新增测试集加载器

    # 初始化 CLIP 融合模型
    clip_fusion_model = CLIPFusionModel(
        local_clip_path=LOCAL_CLIP_PATH,
        num_classes=NUM_CLASSES,
        fusion_mode=FUSION_MODE,
        dropout_prob=DROPOUT_PROB,
        hidden_layer_dim=HIDDEN_LAYER_DIM,
        unfreeze_vision_layers=UNFREEZE_VISION_LAYERS,
        unfreeze_text_layers=UNFREEZE_TEXT_LAYERS
    ).to(device)

    # 统计参数量
    count_model_parameters(clip_fusion_model)

    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(clip_fusion_model.parameters(), lr=LEARNING_RATE)

    # 运行 CLIP 多模态实验
    exp_name = f"clip"
    print("\n" + "="*70)
    print(f"开始 CLIP 多模态融合实验：{exp_name}")
    print("="*70)
    start = time.time()

    best_acc, _ = run_clip_exp(
        model=clip_fusion_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        exp_name=exp_name
    )
    end = time.time()
    print(f"训练耗时：{end-start:.2f} 秒")

    # 输出最终结果
    print("\n" + "="*70)
    print(f"训练结束！最佳验证准确率：{best_acc:.4f}")
    print("="*70)

    # 评估
    evaluate_test_set(clip_fusion_model, valid_loader, device)