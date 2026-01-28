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
import cv2  
from PIL import Image  
from 创新版数据读取 import load_train_valid_test 
import time

# Grad-CAM实现类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None  
        self.gradient = None     
        
        # 注册前向/反向钩子，捕获特征图和梯度
        self._register_hooks()
    
    def _register_hooks(self):
        # 前向钩子：捕获特征图
        def forward_hook(module, input, output):
            self.feature_map = output.detach()
        
        # 反向钩子：捕获梯度
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        # 为目标层注册钩子
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, image, target_class=None):
        # 前向传播获取模型输出
        self.model.eval()
        with torch.enable_grad():
            output = self.model(image)
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # 反向传播计算梯度
        self.model.zero_grad()
        class_loss = output[:, target_class].sum()
        class_loss.backward()
        
        # 计算梯度的全局平均池化（得到特征图权重）
        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        
        # 权重加权求和特征图，得到原始热力图
        heatmap = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        
        # ReLU激活（只保留正贡献，负贡献置0）
        heatmap = nn.functional.relu(heatmap)
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 上采样到原始图像尺寸（使用双线性插值）
        heatmap = nn.functional.interpolate(
            heatmap, 
            size=(image.shape[2], image.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 转换为numpy数组
        heatmap = heatmap.squeeze().cpu().numpy()
        
        return heatmap
    
    @staticmethod
    def visualize_heatmap(original_image, heatmap, alpha=0.5, cmap='jet'):
        # 处理原始图像：统一转换为NumPy数组
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().detach()  
            if original_image.dim() == 3:  
                original_image = original_image.permute(1, 2, 0)  
            original_image = original_image.numpy()  
        elif isinstance(original_image, Image.Image):
            original_image = np.array(original_image, dtype=np.uint8)
        elif not isinstance(original_image, np.ndarray):
            raise TypeError(f"不支持的图像类型：{type(original_image)}，支持 [Tensor/PIL Image/NumPy数组]")
        
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # 处理热力图
        cmap = plt.get_cmap(cmap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # 去除alpha通道
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 调整热力图尺寸匹配原始图像
        heatmap_colored = cv2.resize(
            heatmap_colored, 
            (original_image.shape[1], original_image.shape[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 图像叠加（原始图像 + 彩色热力图）
        overlay_image = cv2.addWeighted(
            original_image, 
            1 - alpha, 
            heatmap_colored, 
            alpha, 
            0
        )
        
        # 转换回PIL Image格式
        overlay_image = Image.fromarray(overlay_image)
        
        return overlay_image

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

# MultiModalDataset
class MultiModalDataset(Dataset):
    def __init__(self, data, text_tokenizer, image_transform=None):
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.label2idx = {'positive':0, 'neutral':1, 'negative':2}
        self.text_max_len = 32
        self.keep_original_image = True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 文本处理
        input_ids = None
        attention_mask = None
        if self.text_tokenizer is not None:
            text = sample['txt_content']
            text_encoding = self.text_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.text_max_len,
                return_tensors='pt'
            )
            input_ids = text_encoding['input_ids'].squeeze(0)
            attention_mask = text_encoding['attention_mask'].squeeze(0)
        
        original_image = None
        if self.keep_original_image:
            img_content = sample['jpg_content']
            if isinstance(img_content, torch.Tensor):
                original_image = img_content.clone()  
            elif isinstance(img_content, np.ndarray):
                original_image = img_content.copy()   
            else:
                original_image = img_content 
        
        # 处理模型输入的图像
        image = sample['jpg_content']  
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        # 标签处理
        if sample['tag'] is None:
            label = torch.tensor(0, dtype=torch.long)  # 占位标签，测试集不参与训练/评估
        else:
            label = self.label2idx[sample['tag']]
            label = torch.tensor(label, dtype=torch.long)
        
        # 返回结果适配两种分支
        return_dict = {
            'image': image,
            'original_image': original_image, 
            'label': label,
            'guid': sample['guid']
        }
        if self.text_tokenizer is not None:
            return_dict['input_ids'] = input_ids
            return_dict['attention_mask'] = attention_mask
        
        return return_dict

# 文本分支
class TextBranch(nn.Module):
    def __init__(
        self, 
        num_classes=3, 
        local_bert_path='bert-base-uncased',
        freeze_embeddings=True,
        freeze_encoder_layers=0,
        hidden_dim=256,
        dropout_prob=0.3,
        activation='relu'
    ):  
        super().__init__()
        self.bert = BertModel.from_pretrained(local_bert_path, cache_dir=None)
        
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_encoder_layers > 0:
            for idx, (name, param) in enumerate(self.bert.named_parameters()):
                if 'encoder.layer.' in name:
                    name_parts = name.split('.')
                    try:
                        layer_index = name_parts.index('layer')
                        layer_num = int(name_parts[layer_index + 1])
                    except (ValueError, IndexError):
                        continue
                    # 冻结小于 freeze_encoder_layers 的层
                    if layer_num < freeze_encoder_layers:
                        param.requires_grad = False
               
        if activation.lower() == 'relu':
            self.act = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数：{activation}，可选['relu', 'gelu', 'tanh']")
        
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            self.act,
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_hidden)
        probs = torch.softmax(logits, dim=1)
        return probs

# 图像分支
class ImageBranch(nn.Module):
    def __init__(
        self, 
        num_classes=3, 
        local_resnet_path='./local_weights/resnet18-f37072fd.pth',
        freeze_resnet_layers=6,
        hidden_dim=256,
        dropout_prob=0.3
    ):  
        super().__init__()
        self.resnet = resnet18(weights=None)
        local_weights = torch.load(local_resnet_path)  
        self.resnet.load_state_dict(local_weights)
        
        # 拆分ResNet：提取layer4（最后一个卷积层），便于Grad-CAM注册钩子
        self.resnet_conv = nn.Sequential(*list(self.resnet.children())[:-2])  
        self.resnet_avgpool = self.resnet.avgpool  
        self.layer4 = self.resnet_conv[-1]  
        
        # 冻结指定层参数
        resnet_params = list(self.resnet_conv.parameters())
        for param in resnet_params[:freeze_resnet_layers]:
            param.requires_grad = False
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, image):
        # 前向传播：卷积层 -> 平均池化 -> 全连接层
        features = self.resnet_conv(image)
        features = self.resnet_avgpool(features)
        logits = self.fc(features)
        probs = torch.softmax(logits, dim=1)
        return probs

# 训练函数
def train_single_branch(model, dataloader, criterion, optimizer, device, epoch, model_type):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        desc=f"[{model_type}] Epoch {epoch+1:02d}",
        leave=False
    )
    
    for batch_idx, batch in progress_bar:
        if isinstance(model, TextBranch):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            probs = model(input_ids, attention_mask)
        elif isinstance(model, ImageBranch):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            probs = model(images)
        else:
            raise ValueError("仅支持TextBranch/ImageBranch")
        
        loss = criterion(torch.log(probs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# 验证损失函数
def calculate_valid_loss(model, dataloader, criterion, device):
    model.eval()
    total_valid_loss = 0.0
    
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="计算验证损失",
        leave=False
    )
    
    with torch.no_grad():
        for batch_idx, batch in pbar:  
            if isinstance(model, TextBranch):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                probs = model(input_ids, attention_mask)
            elif isinstance(model, ImageBranch):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                probs = model(images)
            else:
                raise ValueError("仅支持TextBranch/ImageBranch")
            
            loss = criterion(torch.log(probs), labels)
            total_valid_loss += loss.item()
            
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
    
    avg_valid_loss = total_valid_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_valid_loss

# 评估函数
def evaluate_single_branch(model, dataloader, device, label2idx):
    model.eval()
    all_probs = []
    all_labels = []
    all_guids = []
    idx2label = {v:k for k,v in label2idx.items()}
    
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="执行模型评估",
        leave=False
    )
    
    with torch.no_grad():
        for batch_idx, batch in pbar:  
            if isinstance(model, TextBranch):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                probs = model(input_ids, attention_mask)
                labels = batch['label'].to(device)
                guids = batch['guid']
                
                all_probs.append(probs)
                all_labels.append(labels)
                all_guids.extend(guids)
            elif isinstance(model, ImageBranch):
                images = batch['image'].to(device)
                probs = model(images)
                labels = batch['label'].to(device)
                guids = batch['guid']
                
                all_probs.append(probs)
                all_labels.append(labels)
                all_guids.extend(guids)
            else:
                raise ValueError("仅支持TextBranch/ImageBranch")
            
            processed_samples = (batch_idx + 1) * dataloader.batch_size
            total_samples = len(dataloader.dataset)
            pbar.set_postfix(processed=f"{processed_samples}/{total_samples}")
    
    all_probs = torch.cat(all_probs, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    
    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)
    
    results = {}
    for guid, prob, pred_idx, true_idx in zip(all_guids, all_probs, preds, all_labels):
        results[guid] = {
            'probs': prob,
            'pred_label': idx2label[pred_idx],
            'true_label': idx2label[true_idx]
        }
    
    return acc, results

# 晚期融合策略
def late_fusion_weighted(text_results, image_results, weight_text=0.5, weight_image=0.5):
    fusion_results = {}
    common_guids = set(text_results.keys()) & set(image_results.keys())
    idx2label = {0:'positive', 1:'neutral', 2:'negative'}
    
    for guid in common_guids:
        text_prob = text_results[guid]['probs']
        image_prob = image_results[guid]['probs']
        true_label = text_results[guid]['true_label']
        
        fusion_prob = weight_text * text_prob + weight_image * image_prob
        fusion_pred_idx = np.argmax(fusion_prob)
        fusion_pred_label = idx2label[fusion_pred_idx]
        
        fusion_results[guid] = {
            'fusion_prob': fusion_prob,
            'fusion_pred_label': fusion_pred_label,
            'true_label': true_label,
            'text_prob': text_prob,
            'image_prob': image_prob,
            'text_pred_label': text_results[guid]['pred_label'],
            'image_pred_label': image_results[guid]['pred_label']
        }
    
    preds = [v['fusion_pred_label'] for v in fusion_results.values()]
    trues = [v['true_label'] for v in fusion_results.values()]
    fusion_acc = accuracy_score(trues, preds)
    
    return fusion_acc, fusion_results

def late_fusion_vote(text_results, image_results):
    fusion_results = {}
    common_guids = set(text_results.keys()) & set(image_results.keys())
    idx2label = {0:'positive', 1:'neutral', 2:'negative'}
    
    for guid in common_guids:
        text_pred = text_results[guid]['pred_label']
        image_pred = image_results[guid]['pred_label']
        text_prob = text_results[guid]['probs']
        image_prob = image_results[guid]['probs']
        true_label = text_results[guid]['true_label']
        
        text_pred_prob = np.max(text_prob)
        image_pred_prob = np.max(image_prob)
        
        if text_pred == image_pred:
            fusion_pred_label = text_pred
        else:
            fusion_pred_label = text_pred if text_pred_prob > image_pred_prob else image_pred
        
        fusion_results[guid] = {
            'fusion_pred_label': fusion_pred_label,
            'true_label': true_label,
            'text_pred_label': text_pred,
            'image_pred_label': image_pred,
            'text_pred_prob': text_pred_prob,
            'image_pred_prob': image_pred_prob
        }
    
    preds = [v['fusion_pred_label'] for v in fusion_results.values()]
    trues = [v['true_label'] for v in fusion_results.values()]
    fusion_acc = accuracy_score(trues, preds)
    
    return fusion_acc, fusion_results

# 绘图函数
def plot_training_curve(history, model_name):
    epochs = range(1, len(history['train_losses']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='训练损失')
    ax1.plot(epochs, history['valid_losses'], 'r--', label='验证损失')
    ax1.set_title(f'{model_name} - 损失变化对比')
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


# Grad-CAM可视化函数
def visualize_image_gradcam(image_model, dataloader, device, num_samples=3):
    # 初始化Grad-CAM（目标层为image_model的layer4）
    grad_cam = GradCAM(image_model, image_model.layer4)
    image_model.eval()  # 评估模式，确保预测结果稳定
    
    # 遍历数据加载器，获取样本并可视化
    sample_count = 0
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # 辅助函数：统一处理图像格式
    def process_image_for_plt(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"不支持的图像类型：{type(img)}，支持 [Tensor/PIL Image/NumPy数组]")
        
        # 归一化到[0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    for batch in dataloader:
        if sample_count >= num_samples:
            break
        
        images = batch['image'].to(device)
        original_images = batch['original_image']
        true_labels = batch['label'].cpu().numpy()  # 真实标签
        idx2label = {0:'positive', 1:'neutral', 2:'negative'}
        
        # 获取模型对当前批次图像的预测结果
        with torch.no_grad():
            pred_probs = image_model(images)  
            pred_idxs = torch.argmax(pred_probs, dim=1).cpu().numpy()  
            pred_labels = [idx2label[idx] for idx in pred_idxs]  
        
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
            
            single_image = images[i:i+1]  
            heatmap = grad_cam.generate_heatmap(single_image)
            
            original_img = original_images[i]
            processed_original = process_image_for_plt(original_img)
            
            overlay_img = GradCAM.visualize_heatmap(original_img, heatmap)
            processed_overlay = np.array(overlay_img) / 255.0  # 归一化到[0, 1]
            
            true_label = idx2label[true_labels[i]]
            pred_label = pred_labels[i]
            
            axes[sample_count, 0].imshow(processed_original)
            axes[sample_count, 0].set_title(f'Original Image\nTrue: {true_label} | Pred: {pred_label}')
            axes[sample_count, 0].axis('off')
            
            axes[sample_count, 1].imshow(processed_overlay)
            axes[sample_count, 1].set_title(f'Grad-CAM Overlay\nTrue: {true_label} | Pred: {pred_label}')
            axes[sample_count, 1].axis('off')
            
            sample_count += 1
    
    plt.tight_layout()
    plt.show()

# 训练+融合+消融实验+Grad-CAM可视化
if __name__ == "__main__":
    # 训练常数
    BATCH_SIZE = 16   #
    EPOCHS = 18    #
    PATIENCE = 4
    TEXT_LR = 2e-5
    IMAGE_LR = 1e-4
    IMAGE_SIZE = 224
    TEXT_MAX_LEN = 32
    FUSION_WEIGHT_TEXT = 0.5    #0.5
    FUSION_WEIGHT_IMAGE = 1 - FUSION_WEIGHT_TEXT
    LABEL2IDX = {'positive':0, 'neutral':1, 'negative':2}
    NUM_WORKERS = 0
    
    # 文本分支可调参数
    TEXT_FREEZE_EMBEDDINGS = True
    TEXT_FREEZE_ENCODER_LAYERS = 2    # 0-12（从浅到深）
    TEXT_HIDDEN_DIM = 256     # 256 Bert 编码器之后全连接层（FC 层）的隐藏层维度
    TEXT_DROPOUT_PROB = 0.3   # 0.3
    TEXT_ACTIVATION = 'gelu'
    
    # 图像分支可调参数
    IMAGE_FREEZE_RESNET_LAYERS = 2   # 0-5（从浅到深）
    IMAGE_HIDDEN_DIM = 256    # 256 ResNet18 全连接层（FC 层）的隐藏层维度
    IMAGE_DROPOUT_PROB = 0.3  # 0.3
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 加载数据
    train_data, valid_data, test_data = load_train_valid_test()
    print(f"数据加载完成：训练集{len(train_data)}，验证集{len(valid_data)}，测试集{len(test_data)}")
    
    # 数据预处理配置
    text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 图像变换
    train_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    # 验证/测试集图像变换
    val_test_image_transform = transforms.Compose([
        transforms.Normalize(mean=[0.6313, 0.6022, 0.5861], std=[0.3512, 0.3568, 0.3625])
    ])
    # 文本分支图像变换
    text_image_transform = None  

    # 文本分支数据集
    train_text_dataset = MultiModalDataset(train_data, text_tokenizer, text_image_transform)  
    valid_text_dataset = MultiModalDataset(valid_data, text_tokenizer, text_image_transform)
    train_text_loader = DataLoader(train_text_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_text_loader = DataLoader(valid_text_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 图像分支数据集
    train_image_dataset = MultiModalDataset(train_data, None, train_image_transform)  
    valid_image_dataset = MultiModalDataset(valid_data, None, val_test_image_transform)
    train_image_loader = DataLoader(train_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_image_loader = DataLoader(valid_image_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 训练文本分支
    print("\n===== 训练文本分支 =====")
    text_model = TextBranch(
        num_classes=3,
        freeze_embeddings=TEXT_FREEZE_EMBEDDINGS,
        freeze_encoder_layers=TEXT_FREEZE_ENCODER_LAYERS,
        hidden_dim=TEXT_HIDDEN_DIM,
        dropout_prob=TEXT_DROPOUT_PROB,
        activation=TEXT_ACTIVATION
    ).to(device)
    
    text_total_params, text_trainable_params = count_model_parameters(text_model)
    
    text_criterion = nn.NLLLoss()
    text_optimizer = optim.AdamW(text_model.parameters(), lr=TEXT_LR)
    
    best_text_acc = 0.0
    text_history = {'train_losses': [], 'valid_losses': [], 'valid_accs': []}
    
    no_improvement_count = 0
    start = time.time()
    for epoch in range(EPOCHS):
        train_loss = train_single_branch(
            text_model, train_text_loader, text_criterion, 
            text_optimizer, device, epoch, "Text"
        )
        valid_loss = calculate_valid_loss(text_model, valid_text_loader, text_criterion, device)
        valid_acc, _ = evaluate_single_branch(text_model, valid_text_loader, device, LABEL2IDX)
        
        text_history['train_losses'].append(train_loss)
        text_history['valid_losses'].append(valid_loss)
        text_history['valid_accs'].append(valid_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失：{train_loss:.4f} | 验证损失：{valid_loss:.4f} | 验证准确率：{valid_acc:.4f}")
        
        # 早停
        if valid_acc > best_text_acc:
            best_text_acc = valid_acc
            torch.save(text_model.state_dict(), 'best_text_model.pth')
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  早停：{no_improvement_count}/{PATIENCE}")
        if no_improvement_count > PATIENCE:
            print(f"早停触发！连续 {PATIENCE} 轮验证准确率无有效提升，终止训练")
            break  
    end = time.time()
    print(f"训练文本耗时{end-start}")


    plot_training_curve(text_history, "文本分支（Bert）")
    text_model.load_state_dict(torch.load('best_text_model.pth'))
    val_text_acc, val_text_results = evaluate_single_branch(text_model, valid_text_loader, device, LABEL2IDX)
    print(f"文本分支验证集准确率（消融实验）：{val_text_acc:.4f}")
    
    # 训练图像分支
    print("\n===== 训练图像分支 =====")
    image_model = ImageBranch(
        num_classes=3,
        freeze_resnet_layers=IMAGE_FREEZE_RESNET_LAYERS,
        hidden_dim=IMAGE_HIDDEN_DIM,
        dropout_prob=IMAGE_DROPOUT_PROB
    ).to(device)
    
    image_total_params, image_trainable_params = count_model_parameters(image_model)
    
    image_criterion = nn.NLLLoss()
    image_optimizer = optim.AdamW(image_model.parameters(), lr=IMAGE_LR)
    
    best_image_acc = 0.0
    image_history = {'train_losses': [], 'valid_losses': [], 'valid_accs': []}
    
    no_improvement_count = 0
    start = time.time()
    for epoch in range(EPOCHS):
        train_loss = train_single_branch(
            image_model, train_image_loader, image_criterion, 
            image_optimizer, device, epoch, "Image"
        )
        valid_loss = calculate_valid_loss(image_model, valid_image_loader, image_criterion, device)
        valid_acc, _ = evaluate_single_branch(image_model, valid_image_loader, device, LABEL2IDX)
        
        image_history['train_losses'].append(train_loss)
        image_history['valid_losses'].append(valid_loss)
        image_history['valid_accs'].append(valid_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失：{train_loss:.4f} | 验证损失：{valid_loss:.4f} | 验证准确率：{valid_acc:.4f}")
        
        if valid_acc > best_image_acc:
            best_image_acc = valid_acc
            torch.save(image_model.state_dict(), 'best_image_model.pth')
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  早停：{no_improvement_count}/{PATIENCE}")
        if no_improvement_count > PATIENCE:
            print(f"早停触发！连续 {PATIENCE} 轮验证准确率无有效提升，终止训练")
            break 
    end = time.time()
    print(f"训练图像耗时{end-start}")
    
    plot_training_curve(image_history, "图像分支（ResNet18）")
    image_model.load_state_dict(torch.load('best_image_model.pth'))
    val_image_acc, val_image_results = evaluate_single_branch(image_model, valid_image_loader, device, LABEL2IDX)
    print(f"图像分支验证集准确率（消融实验）：{val_image_acc:.4f}")
    
    # 晚期融合策略
    print("\n===== 晚期融合（验证集） =====")
    fusion_weighted_acc, fusion_weighted_results = late_fusion_weighted(
        val_text_results, val_image_results, FUSION_WEIGHT_TEXT, FUSION_WEIGHT_IMAGE
    )
    print(f"加权平均融合（text={FUSION_WEIGHT_TEXT}, image={FUSION_WEIGHT_IMAGE}）验证集准确率：{fusion_weighted_acc:.4f}")
    
    fusion_vote_acc, fusion_vote_results = late_fusion_vote(val_text_results, val_image_results)
    print(f"投票机制融合验证集准确率：{fusion_vote_acc:.4f}")
    
    # 消融实验结果汇总
    print("\n===== 消融实验结果汇总（验证集） =====")
    print(f"仅文本模态：{val_text_acc:.4f}")
    print(f"仅图像模态：{val_image_acc:.4f}")
    print(f"加权融合模态：{fusion_weighted_acc:.4f}")
    print(f"投票融合模态：{fusion_vote_acc:.4f}")
    
    # 消融实验结果汇总
    print("\n===== 消融实验结果汇总（验证集） =====")
    print(f"仅文本模态：{val_text_acc:.4f}")
    print(f"仅图像模态：{val_image_acc:.4f}")
    print(f"加权融合模态：{fusion_weighted_acc:.4f}")
    print(f"投票融合模态：{fusion_vote_acc:.4f}")

    print("\n" + "="*60)
    print("===== 详细分类报告（Classification Report） =====")
    print("="*60)

    # 标签映射
    idx2label = {0:'positive', 1:'neutral', 2:'negative'}
    labels = ['positive', 'neutral', 'negative']  

    # 仅文本模态的分类报告
    print("\n1. 仅文本模态（Text Branch）")
    print("-"*50)
    text_trues = [v['true_label'] for v in val_text_results.values()]
    text_preds = [v['pred_label'] for v in val_text_results.values()]
    print(classification_report(text_trues, text_preds, target_names=labels, digits=4))

    # 仅图像模态的分类报告
    print("\n2. 仅图像模态（Image Branch）")
    print("-"*50)
    image_trues = [v['true_label'] for v in val_image_results.values()]
    image_preds = [v['pred_label'] for v in val_image_results.values()]
    print(classification_report(image_trues, image_preds, target_names=labels, digits=4))

    # 加权平均融合的分类报告
    print("\n3. 加权平均融合（Weighted Late Fusion）")
    print("-"*50)
    weighted_trues = [v['true_label'] for v in fusion_weighted_results.values()]
    weighted_preds = [v['fusion_pred_label'] for v in fusion_weighted_results.values()]
    print(classification_report(weighted_trues, weighted_preds, target_names=labels, digits=4))

    # 投票机制融合的分类报告
    print("\n4. 投票机制融合（Vote Late Fusion）")
    print("-"*50)
    vote_trues = [v['true_label'] for v in fusion_vote_results.values()]
    vote_preds = [v['fusion_pred_label'] for v in fusion_vote_results.values()]
    print(classification_report(vote_trues, vote_preds, target_names=labels, digits=4))

    # Grad-CAM热力图可视化（验证集）
    print("\n===== 图像分支Grad-CAM热力图可视化 =====")
    visualize_image_gradcam(image_model, valid_image_loader, device, num_samples=9)