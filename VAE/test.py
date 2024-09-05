import torch
from torchvision.utils import save_image
from main import VAE

# 加载模型
model = VAE().to('cpu')
model.load_state_dict(torch.load('vae.pth'))

# 从潜在空间中采样
with torch.no_grad():
    z = torch.randn(64, 20).to('cpu')  # 64是生成样本的数量，20是潜在向量的维度
    generated_images = model.decode(z).cpu()

# 保存生成图像
save_image(generated_images.view(64, 1, 28, 28), 'generated_sample.png')
