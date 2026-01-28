import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")
import time
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

# 多模态数据集类
class MultiModalDataset(Dataset):
    def __init__(self, data, text_tokenizer, image_transform=None, image_size=224):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.image_size = image_size  
        self.label2idx = {'positive':0, 'neutral':1, 'negative':2}
        self.text_max_len = 32
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if len(self.data) == 0:
            return {
                'image': torch.randn(3, self.image_size, self.image_size),
                'label': torch.tensor(0, dtype=torch.long),
                'guid': f'guid_{idx}',
                'input_ids': torch.zeros(self.text_max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.text_max_len, dtype=torch.long)
            }
        
        sample = self.data[idx]
        
        # 文本处理
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
        
        # 图像处理
        image = sample.get('jpg_content')
        basic_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        if image is not None:
            image = basic_transform(image)  
            if self.image_transform is not None:
                image = self.image_transform(image)
        else:
            image = torch.randn(3, self.image_size, self.image_size)
        
        # 标签处理
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

# 文本特征提取器
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

# 图像特征提取器
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
        
        # 加载本地权重
        try:
            local_weights = torch.load(local_resnet_path, map_location='cpu')
            self.resnet.load_state_dict(local_weights)
        except:
            print(f"本地ResNet权重加载失败，使用随机初始化权重")
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 冻结指定层
        resnet_params = list(self.resnet.parameters())
        freeze_num = min(freeze_resnet_layers, len(resnet_params))  
        for param in resnet_params[:freeze_num]:
            param.requires_grad = False
        
        # 特征投影层
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

# 早期融合模型
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

# 绘制训练曲线
def plot_training_curve(history, model_name):
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['valid_losses'], 'r--', label='验证损失')
    ax1.set_title(f'{model_name} - 损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 2.0)
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
    plt.savefig(f"{model_name}")

# 早期融合通用训练评估函数
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
            
            # 前向传播
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
        
        # 计算验证指标
        avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0.0
        valid_acc = 0.0
        if all_probs and all_labels:  
            try:
                all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
                all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
                if len(all_probs) == len(all_labels):
                    valid_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1))
            except Exception as e:
                print(f"计算准确率异常：{e}")
        
        history['train_losses'].append(avg_train_loss)
        history['valid_losses'].append(avg_valid_loss)
        history['valid_accs'].append(valid_acc)
        
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
    
    model.load_state_dict(torch.load(f'best_early_fusion_{exp_name}.pth', map_location=device))
    
    return best_acc, history

# 分类报告评估函数
def evaluate_exp_classification_report(model, valid_loader, device, exp_scene, target_names=['positive', 'neutral', 'negative']):
    model.eval()
    all_preds = []
    all_true = []
    only_text = (exp_scene == 'only_text')
    only_image = (exp_scene == 'only_image')
    
    with torch.no_grad():
        pbar_eval = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"[{exp_scene}] 分类报告评估")
        for batch_idx, batch in pbar_eval:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播（适配消融实验场景）
            probs = model(input_ids, attention_mask, images, only_text, only_image)
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            true = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_true.extend(true)
    
    
    exp_name_mapping = {
        'full_fusion': '完整早期融合（文本+图像）',
        'only_text': '仅文本模态（消融图像）',
        'only_image': '仅图像模态（消融文本）'
    }
    print("\n" + "="*80)
    print(f"{exp_name_mapping[exp_scene]} - 验证集分类报告（Precision/Recall/F1-Score）：")
    print("="*80)
    print(classification_report(all_true, all_preds, target_names=target_names, digits=4))
    eval_acc = accuracy_score(all_true, all_preds)
    print(f"验证集整体准确率（分类报告复核）：{eval_acc:.4f}")
    print("="*80)
    return eval_acc, all_true, all_preds

# 主函数：训练+消融实验
if __name__ == "__main__":
    # 超参数配置
    BATCH_SIZE = 16
    EPOCHS = 18  #
    PATIENCE = 4
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224  #
    TEXT_MAX_LEN = 32
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0  
    
    # 文本分支参数
    TEXT_FREEZE_EMBEDDINGS = True
    TEXT_FREEZE_ENCODER_LAYERS = 2   # 0-12
    TEXT_HIDDEN_DIM = 256
    TEXT_DROPOUT = 0.3
    TEXT_ACTIVATION = 'gelu'
    
    # 图像分支参数
    IMAGE_FREEZE_RESNET_LAYERS = 2   # 0-5
    IMAGE_HIDDEN_DIM = 256
    IMAGE_DROPOUT = 0.3
    
    # 融合层参数
    FUSION_HIDDEN_DIM = 1024
    FUSION_DROPOUT = 0.3
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备：{device}")
    print(f"图像输入尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    
    # 数据加载
    train_data, valid_data, test_data = load_train_valid_test(picture="PIL")
    print(f"数据加载完成：训练集{len(train_data)} | 验证集{len(valid_data)} | 测试集{len(test_data)}")
    
    # 初始化Bert分词器
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=None)
    
    # 图像变换
    train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 仅训练集增强
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    val_test_image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    
    # 构建数据集和数据加载器
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
    # 数据加载器配置
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
    
    # 初始化特征提取器和融合模型
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
    start = time.time()
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
    end = time.time()
    print(f"耗时{end-start}")
    
    # 实验2：消融实验 - 仅文本模态（屏蔽图像）
    print("\n--- 实验2：消融实验 - 仅文本模态 ---")
    start = time.time()
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
    end = time.time()
    print(f"耗时{end-start}")
    
    # 实验3：消融实验 - 仅图像模态（屏蔽文本）
    print("\n--- 实验3：消融实验 - 仅图像模态 ---")
    optimizer3 = optim.AdamW(early_fusion_model.parameters(), lr=LEARNING_RATE)
    start = time.time()
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
    end = time.time()
    print(f"耗时{end-start}")
    
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
    
    # 输出三个实验的分类报告
    print("\n" + "="*90)
    print("早期融合消融实验 - 分类报告详细汇总")
    print("="*90)
    for exp_scene in ['full_fusion', 'only_text', 'only_image']:
        early_fusion_model.load_state_dict(torch.load(f'best_early_fusion_{exp_scene}.pth', map_location=device))
        evaluate_exp_classification_report(early_fusion_model, valid_loader, device, exp_scene)