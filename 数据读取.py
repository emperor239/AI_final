import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.sans-serif'] = ['SimHei'] 

def clean_social_media_text(text):
    text = re.sub(r'\bRT\b', ' ', text)                        # 过滤转发标记"RT"
    text = re.sub(r'@\w+', ' ', text)                          # 过滤用户提及"@xxx"）
    text = re.sub(r'htt\S*', ' ', text, flags=re.IGNORECASE)   # 过滤链接
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)                # 仅保留字母、数字、空格
    text = re.sub(r'\s+', ' ', text).strip()                   # 过滤多余空格，并移除首尾空格
    text = text.lower()                                        # 转小写
    return text

def process_image(image_path, target_size=(224, 224), pad_color=(255, 255, 255), picture=None):
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    target_width, target_height = target_size

    # 取宽、高中较小的缩放比例，等比例缩放图像
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale) 
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 高质量缩放

    # 创建新的空白图像，并将缩放后的图像粘贴到中间
    new_img = Image.new('RGB', target_size, pad_color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))

    if picture == None:
        # 把PIL转化为tensor
        new_img = transforms.ToTensor()(new_img)
        return new_img
    
    if picture == "PIL":
        return new_img


def load_train_valid_test(known_path="train.txt", unknown_path="test_without_label.txt", data_dir="data", test_size=0.2, random_state=42, picture=None):
    i = 0
    # 先处理训练集和验证集
    known = pd.read_csv(known_path)
    full_data = []
    for idx, row in known.iterrows():
        guid = str(int(row['guid'])) 
        tag = row['tag']
        
        jpg_path = os.path.join(data_dir, f"{guid}.jpg")
        txt_path = os.path.join(data_dir, f"{guid}.txt")
        
        jpg_content = process_image(image_path=jpg_path, picture=picture)
        txt_content = None
        with open(txt_path, 'r', encoding='utf-8', errors="ignore") as f:
            txt_content = f.read().strip()
        txt_content = clean_social_media_text(txt_content)
        
        full_data.append({
            'guid': guid,
            'jpg_content': jpg_content,
            'txt_content': txt_content,
            'tag': tag,
        })

        i += 1
        # print(i)
    
    train, valid = train_test_split(
        full_data,
        test_size=test_size,
        random_state=random_state,
        stratify=[item['tag'] for item in full_data]  # 按tag分层抽样，保证训练/验证集标签分布一致
    )
    
    # 再处理测试集
    unknown = pd.read_csv(unknown_path)
    test = []
    for idx, row in unknown.iterrows():
        guid = str(int(row['guid'])) 
        tag = row['tag']
        
        jpg_path = os.path.join(data_dir, f"{guid}.jpg")
        txt_path = os.path.join(data_dir, f"{guid}.txt")
        
        jpg_content = process_image(image_path=jpg_path, picture=picture)
        txt_content = None
        with open(txt_path, 'r', encoding='utf-8', errors="ignore") as f:
            txt_content = f.read().strip()
        txt_content = clean_social_media_text(txt_content)
        
        test.append({
            'guid': guid,
            'jpg_content': jpg_content,
            'txt_content': txt_content,
            'tag': tag,
        })

        i += 1
        # print(i)
    
    return train, valid, test

def plot_ndarray_image(img_np):
    img_np = np.transpose(img_np, (1, 2, 0))
    plt.imshow(img_np)
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 计算图像均值和方差
def calculate_image_mean_std(image_list):
    channel_sum = torch.zeros(3)    # 各通道像素值总和
    channel_sum_sq = torch.zeros(3) # 各通道像素值平方和
    total_pixels = 0                # 总像素个数
    
    for idx, img_tensor in enumerate(image_list):
        # 提取三通道数据（img_tensor: [3, H, W]）
        # 展平为 [3, H*W]，方便按通道求和
        img_flatten = img_tensor.view(3, -1)
        
        channel_sum += img_flatten.sum(dim=1)
        channel_sum_sq += (img_flatten ** 2).sum(dim=1)
        total_pixels += img_flatten.shape[1]
    
    channel_means = channel_sum / total_pixels
    channel_vars = (channel_sum_sq / total_pixels) - (channel_means ** 2)
    channel_stds = torch.sqrt(channel_vars)
    
    channel_means = channel_means.numpy()
    channel_stds = channel_stds.numpy()
    
    return channel_means, channel_stds


if __name__ == "__main__":
    train, valid, test = load_train_valid_test()
    print(f"训练集{len(train)}条，验证集{len(valid)}条，测试集{len(test)}条")
    
    print(f"训练集")
    print(train[100].get("guid"))
    plot_ndarray_image(train[100].get("jpg_content"))
    print(train[100].get("txt_content"))
    
    print(f"验证集")
    print(valid[100].get("guid"))
    plot_ndarray_image(valid[100].get("jpg_content"))
    print(valid[100].get("txt_content"))
       
    print(f"测试集")
    print(test[100].get("guid"))
    plot_ndarray_image(test[100].get("jpg_content"))
    print(test[100].get("txt_content"))

    all_images = []
    for item in train:
        all_images.append(item['jpg_content'])
    img_means, img_stds = calculate_image_mean_std(all_images)
    print("图像三通道（R/G/B）均值和方差统计结果")
    print(f"mean={list(img_means.round(4))}, std={list(img_stds.round(4))}")
    
    