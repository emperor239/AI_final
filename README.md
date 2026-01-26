bert模型的准备
创建文件夹bert-base-uncased
进入https://huggingface.co/google-bert/bert-base-uncased/tree/main
下载
config.json
model.safetensors
tokenizer_config.json
tokenizer.json
vocab.txt
将以上文件放入bert-base-uncased文件夹

resnet18模型的准备
创建文件夹local_weights
进入https://download.pytorch.org/models/resnet18-f37072fd.pth
会下载得到resnet18-f37072fd.pth
将以上文件放入local_weights文件夹

clip模型的准备
创建文件夹local_clip_model
运行“下载CLIP.py”

blip模型的准备
创建文件夹local_blip_model
进入https://hf-mirror.com/Salesforce/blip-image-captioning-base/tree/main
下载
preprocessor_config.json
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
tokenizer.json
vocab.txt
将以上文件放入local_blip_model文件夹