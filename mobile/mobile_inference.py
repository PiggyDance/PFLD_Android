import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# 加载.pt文件
model = torch.jit.load('./pfld_mobile.pt')

# 将模型设置为推理模式
model.eval()

# 定义图像转换操作
transform = torchvision.transforms.Compose([
    transforms.Resize([112, 112]),  # 调整图像大小
    torchvision.transforms.ToTensor(),
])

# 加载图像并应用转换
img = Image.open('input.jpg')
original_size = img.size  # (width, height)
input_tensor = transform(img)

# 增加批量维度 shape:(3,922,736) (channel,height,width)
input_batch = input_tensor.unsqueeze(0)

# 进行推理
with torch.no_grad():
    _, landmarks = model(input_batch)

landmarks = landmarks.numpy()  # shape:(1,196)
landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)  # landmark

# 如果 landmarks 是归一化到 0-1 之间的坐标
pre_landmark = landmarks[0] * [112, 112]

# 将 landmarks 从112x112图像映射回原始图像尺寸
pre_landmark[:, 0] = pre_landmark[:, 0] * (original_size[0] / 112.0)
pre_landmark[:, 1] = pre_landmark[:, 1] * (original_size[1] / 112.0)

# 将输入张量转换为图像并显示
show_img = input_tensor.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 顺序
show_img = (show_img * 255).astype(np.uint8)
show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR)  # 转换为 BGR 顺序

# 读取图像进行绘制
img_clone = cv2.imread("input.jpg")

# 绘制关键点
for (x, y) in pre_landmark.astype(np.int32):
    cv2.circle(img_clone, (x, y), 3, (255, 0, 0), -1)

cv2.imshow("show_img", img_clone)
cv2.waitKey(0)
