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
    # transforms.CenterCrop(224),  # 中心裁剪
    torchvision.transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 正规化
])

# 加载图像并应用转换
img = Image.open('input.jpg')
input_tensor = transform(img)

# 增加批量维度 shape:(3,922,736) (channel,height,width)
input_batch = input_tensor.unsqueeze(0)

# 进行推理
with torch.no_grad():
    _, landmarks = model(input_batch)

landmarks = landmarks.numpy()  # shape:(1,196)
landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)  # landmark

show_img = np.array(np.transpose(input_tensor.numpy(), (1, 2, 0)))
show_img = (show_img * 255).astype(np.uint8)
np.clip(show_img, 0, 255)

pre_landmark = landmarks[0] * [112, 112]

cv2.imwrite("show_img.jpg", show_img)
img_clone = cv2.imread("show_img.jpg")

for (x, y) in pre_landmark.astype(np.int32):
    cv2.circle(img_clone, (x, y), 1, (255, 0, 0), -1)
cv2.imshow("show_img.jpg", img_clone)
cv2.waitKey(0)
