from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

clip_model_name = "openai/clip-vit-base-patch32"  
local_clip_save_path = "./local_clip_model"  

# CLIP 预训练模型权重和配置
CLIPModel.from_pretrained(clip_model_name).save_pretrained(local_clip_save_path)
# CLIP 图像处理器
CLIPProcessor.from_pretrained(clip_model_name).save_pretrained(local_clip_save_path)
# CLIP 文本分词器
CLIPTokenizer.from_pretrained(clip_model_name).save_pretrained(local_clip_save_path)

print(f"CLIP 模型已完整保存到本地路径：{local_clip_save_path}")