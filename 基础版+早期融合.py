import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# 从数据读取模块加载数据（保持你的原有逻辑）
from 数据读取 import load_train_valid_test

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
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

# 多模态数据集类（修复图像格式处理、兼容224×224）
class MultiModalDataset(Dataset):
    def __init__(self, data, text_tokenizer, image_transform=None, image_size=224):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.image_size = image_size  # 显式指定图像尺寸
        self.label2idx = {'positive':0, 'neutral':1, 'negative':2}
        self.text_max_len = 32
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 兼容空占位数据，严格返回224×224张量
        if len(self.data) == 0:
            return {
                'image': torch.randn(3, self.image_size, self.image_size),
                'label': torch.tensor(0, dtype=torch.long),
                'guid': f'guid_{idx}',
                'input_ids': torch.zeros(self.text_max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.text_max_len, dtype=torch.long)
            }
        
        sample = self.data[idx]
        
        # 文本处理（原有逻辑不变）
        text = sample.get('txt_content', '')
        text_encoding = self.text_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.text_max_len,
            return_tensors='pt'
        )
        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        # 图像处理（核心修复：兼容PIL、强制224×224、处理变换）
        image = sample.get('jpg_content')
        # 基础变换：无论是否传自定义变换，都保证转为224×224张量
        basic_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        if image is not None:
            image = basic_transform(image)  # 先做基础尺寸/格式转换
            # 再应用自定义变换（训练集增强+归一化/验证集仅归一化）
            if self.image_transform is not None:
                image = self.image_transform(image)
        else:
            # 图像为空时，返回随机224×224张量兜底
            image = torch.randn(3, self.image_size, self.image_size)
        
        # 标签处理（原有逻辑不变）
        if sample.get('tag') is None:
            label = torch.tensor(0, dtype=torch.long)
        else:
            label = self.label2idx[sample['tag']]
            label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image,
            'label': label,
            'guid': sample.get('guid', f'guid_{idx}'),
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# 文本特征提取器（原有逻辑不变）
class TextFeatureExtractor(nn.Module):
    def __init__(
        self, 
        local_bert_path='bert-base-uncased',
        freeze_embeddings=True,
        freeze_encoder_layers=6,
        hidden_dim=256,
        dropout_prob=0.3,
        activation='gelu'
    ):  
        super().__init__()
        self.bert = BertModel.from_pretrained(local_bert_path, cache_dir=None)
        self.hidden_dim = hidden_dim
        
        # 冻结embedding层
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        # 冻结指定编码器层
        if freeze_encoder_layers > 0:
            for _, (name, param) in enumerate(self.bert.named_parameters()):
                if 'encoder.layer.' in name:
                    try:
                        layer_num = int(name.split('.')[3])
                    except (ValueError, IndexError):
                        continue
                    if layer_num < freeze_encoder_layers:
                        param.requires_grad = False
        
        # 激活函数
        if activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数：{activation}，可选['relu', 'gelu', 'tanh']")
        
        # 特征投影层
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            self.act,
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        text_feature = self.fc(cls_hidden)
        return text_feature

# 图像特征提取器（适配224×224输入，原有逻辑不变）
class ImageFeatureExtractor(nn.Module):
    def __init__(
        self, 
        local_resnet_path='./local_weights/resnet18-f37072fd.pth',
        freeze_resnet_layers=6,
        hidden_dim=256,
        dropout_prob=0.3
    ):  
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.hidden_dim = hidden_dim 
        
        # 加载本地权重（兼容权重文件不存在的简易兜底）
        try:
            local_weights = torch.load(local_resnet_path, map_location='cpu')
            self.resnet.load_state_dict(local_weights)
        except:
            print(f"本地ResNet权重加载失败，使用随机初始化权重")
        
        # 移除最后一层全连接层（ResNet18适配224×224输入，输出512维特征）
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 冻结指定层
        resnet_params = list(self.resnet.parameters())
        freeze_num = min(freeze_resnet_layers, len(resnet_params))  # 防止索引越界
        for param in resnet_params[:freeze_num]:
            param.requires_grad = False
        
        # 特征投影层（224×224输入对应512维特征，无需修改）
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, image):
        features = self.resnet(image)
        image_feature = self.fc(features)
        return image_feature

# 早期融合模型（支持消融实验，原有逻辑不变）
class EarlyFusionModel(nn.Module):
    def __init__(
        self,
        text_feature_extractor,
        image_feature_extractor,
        fusion_hidden_dim=512,
        num_classes=3,
        dropout_prob=0.3
    ):
        super().__init__()
        self.text_extractor = text_feature_extractor
        self.image_extractor = image_feature_extractor
        self.text_hidden = text_feature_extractor.hidden_dim
        self.image_hidden = image_feature_extractor.hidden_dim
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.text_hidden + self.image_hidden, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, image, only_text=False, only_image=False):
        # 提取双模态特征
        text_feat = self.text_extractor(input_ids, attention_mask)
        image_feat = self.image_extractor(image)
        
        # 消融实验：屏蔽单一模态（全0张量替代）
        if only_text:
            image_feat = torch.zeros_like(image_feat).to(image_feat.device)
        elif only_image:
            text_feat = torch.zeros_like(text_feat).to(text_feat.device)
        
        # 特征拼接+分类
        fused_feat = torch.cat([text_feat, image_feat], dim=1)
        logits = self.fusion_classifier(fused_feat)
        probs = torch.softmax(logits, dim=1)
        
        return probs

# 绘制训练曲线（原有逻辑不变）
def plot_training_curve(history, model_name):
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['valid_losses'], 'r--', label='验证损失')
    ax1.set_title(f'{model_name} - 损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, history['valid_accs'], 'r-', label='验证准确率')
    ax2.set_title(f'{model_name} - 验证准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# 早期融合通用训练评估函数（原有逻辑不变）
def run_early_fusion_exp(
    model, train_loader, valid_loader, criterion, optimizer, device, 
    epochs, patience, exp_name, only_text=False, only_image=False
):
    best_acc = 0.0
    history = {'train_losses': [], 'valid_losses': [], 'valid_accs': []}
    no_improvement_count = 0
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        # 训练进度条
        pbar_train = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"[{exp_name}] Epoch {epoch+1:02d} (Train)",
            leave=False
        )
        # 批次训练
        for batch_idx, batch in pbar_train:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播（传入消融参数）
            probs = model(input_ids, attention_mask, images, only_text, only_image)
            loss = criterion(torch.log(probs), labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar_train.set_postfix(batch_loss=f"{loss.item():.4f}")
        
        # 计算平均训练损失
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                probs = model(input_ids, attention_mask, images, only_text, only_image)
                # 计算验证损失
                batch_loss = criterion(torch.log(probs), labels).item()
                total_valid_loss += batch_loss
                # 收集预测结果
                all_probs.append(probs)
                all_labels.append(labels)
                
                pbar_valid.set_postfix(batch_loss=f"{batch_loss:.4f}")
        
        # 计算验证指标（兼容空数据加载器）
        avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        valid_acc = 0.0
        if all_probs and all_labels:  # 更鲁棒的非空判断
            try:
                all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
                all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
                if len(all_probs) == len(all_labels):
                    valid_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
            except Exception as e:
                print(f"计算准确率异常：{e}")
        
        # 记录历史
        history['train_losses'].append(avg_train_loss)
        history['valid_losses'].append(avg_valid_loss)
        history['valid_accs'].append(valid_acc)
        
        # 打印日志
        print(f"[{exp_name}] Epoch {epoch+1}/{epochs} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_valid_loss:.4f} | 验证准确率：{valid_acc:.4f}")
        
        # 早停逻辑
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'best_early_fusion_{exp_name}.pth')
            no_improvement_count = 0
            print(f"[{exp_name}] 更新最佳模型，当前最佳准确率：{best_acc:.4f}")
        else:
            no_improvement_count += 1
            print(f"[{exp_name}] 早停计数：{no_improvement_count}/{patience}")
        
        # 触发早停
        if no_improvement_count >= patience:
            print(f"[{exp_name}] 早停触发，终止训练")
            break
    
    # 绘制训练曲线
    plot_training_curve(history, exp_name)
    
    # 加载最佳模型返回最终准确率
    model.load_state_dict(torch.load(f'best_early_fusion_{exp_name}.pth', map_location=device))
    
    return best_acc, history

# 主函数：训练+消融实验（完整224×224图像处理配置）
if __name__ == "__main__":
    # 超参数配置（核心：IMAGE_SIZE=224固定）
    BATCH_SIZE = 16
    EPOCHS = 2  # 恢复合理训练轮数
    PATIENCE = 5
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224  # 显式指定224×224图像尺寸
    TEXT_MAX_LEN = 32
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0  # 兼容CPU运行
    
    # 文本分支参数
    TEXT_FREEZE_EMBEDDINGS = True
    TEXT_FREEZE_ENCODER_LAYERS = 6   # 0-12
    TEXT_HIDDEN_DIM = 256
    TEXT_DROPOUT = 0.3
    TEXT_ACTIVATION = 'gelu'
    
    # 图像分支参数
    IMAGE_FREEZE_RESNET_LAYERS = 3   # 0-5（ResNet18共5层主卷积）
    IMAGE_HIDDEN_DIM = 256
    IMAGE_DROPOUT = 0.3
    
    # 融合层参数
    FUSION_HIDDEN_DIM = 1024
    FUSION_DROPOUT = 0.3
    
    # 设备配置（兼容CPU/GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备：{device}")
    print(f"图像输入尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    
    # 数据加载（确保返回PIL格式图像，与数据集类匹配）
    train_data, valid_data, test_data = load_train_valid_test(picture="PIL")
    print(f"数据加载完成：训练集{len(train_data)} | 验证集{len(valid_data)} | 测试集{len(test_data)}")
    
    # 初始化Bert分词器
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=None)
    
    # 图像变换（核心修复：仅做归一化/增强，尺寸/ToTensor在数据集类内完成，避免类型错误）
    train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 仅训练集增强
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    val_test_image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    
    # 构建数据集和数据加载器（传入IMAGE_SIZE，显式指定224×224）
    train_dataset = MultiModalDataset(
        train_data, text_tokenizer, 
        image_transform=train_image_transform, 
        image_size=IMAGE_SIZE
    )
    valid_dataset = MultiModalDataset(
        valid_data, text_tokenizer, 
        image_transform=val_test_image_transform, 
        image_size=IMAGE_SIZE
    )
    # 数据加载器配置（兼容CPU，关闭pin_memory）
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else 1
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else 1
    )
    
    # 初始化特征提取器和融合模型（移至设备）
    text_encoder = TextFeatureExtractor(
        freeze_embeddings=TEXT_FREEZE_EMBEDDINGS,
        freeze_encoder_layers=TEXT_FREEZE_ENCODER_LAYERS,
        hidden_dim=TEXT_HIDDEN_DIM,
        dropout_prob=TEXT_DROPOUT,
        activation=TEXT_ACTIVATION
    ).to(device)
    
    image_encoder = ImageFeatureExtractor(
        freeze_resnet_layers=IMAGE_FREEZE_RESNET_LAYERS,
        hidden_dim=IMAGE_HIDDEN_DIM,
        dropout_prob=IMAGE_DROPOUT
    ).to(device)
    
    early_fusion_model = EarlyFusionModel(
        text_feature_extractor=text_encoder,
        image_feature_extractor=image_encoder,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout_prob=FUSION_DROPOUT
    ).to(device)
    
    # 统计参数量
    count_model_parameters(early_fusion_model)
    
    # 定义损失函数
    criterion = nn.NLLLoss().to(device)
    
    # 运行早期融合消融实验
    print("\n" + "="*70)
    print("开始早期融合消融实验（完整融合/仅文本/仅图像）")
    print("="*70)
    exp_results = {}
    
    # 实验1：完整早期融合（文本+图像）- 基准实验
    print("\n--- 实验1：完整早期融合（文本+图像）---")
    optimizer1 = optim.AdamW(early_fusion_model.parameters(), lr=LEARNING_RATE)
    exp_results['full_fusion'], _ = run_early_fusion_exp(
        model=early_fusion_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer1,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        exp_name="full_fusion"
    )
    
    # 实验2：消融实验 - 仅文本模态（屏蔽图像）
    print("\n--- 实验2：消融实验 - 仅文本模态 ---")
    optimizer2 = optim.AdamW(early_fusion_model.parameters(), lr=LEARNING_RATE)
    exp_results['only_text'], _ = run_early_fusion_exp(
        model=early_fusion_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer2,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        exp_name="only_text",
        only_text=True
    )
    
    # 实验3：消融实验 - 仅图像模态（屏蔽文本）
    print("\n--- 实验3：消融实验 - 仅图像模态 ---")
    optimizer3 = optim.AdamW(early_fusion_model.parameters(), lr=LEARNING_RATE)
    exp_results['only_image'], _ = run_early_fusion_exp(
        model=early_fusion_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer3,
        device=device,
        epochs=EPOCHS,
        patience=PATIENCE,
        exp_name="only_image",
        only_image=True
    )
    
    # 消融实验结果汇总
    print("\n" + "="*70)
    print("早期融合消融实验结果汇总（验证集最高准确率）")
    print("="*70)
    for exp_name, acc in exp_results.items():
        if exp_name == 'full_fusion':
            print(f"完整早期融合（文本+图像）：{acc:.4f}")
        elif exp_name == 'only_text':
            print(f"仅文本模态（消融图像）：{acc:.4f}")
        elif exp_name == 'only_image':
            print(f"仅图像模态（消融文本）：{acc:.4f}")
    print("="*70)