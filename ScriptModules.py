import torch
import torchvision
from model import MstNet
from PIL import Image
from torchvision import transforms
import utils
import cv2
trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

model = MstNet()
model.load_state_dict(torch.load('./weights/mst_epoch.pkl', map_location='cpu')['model_state_dict'])
model.eval()
example = torch.rand(1, 3, 256, 256)
style = torch.rand(1, 3, 256, 256)
traced_script_module = torch.jit.trace(model, (example, style))
traced_script_module.save("./weights/model.pt")

model = torch.jit.load("./weights/mst.pt")

img = Image.open("./content.jpg").convert("RGB")
content = trans(img).unsqueeze(0)

img_s = Image.open('./style.jpg').convert("RGB")
style = trans(img_s).unsqueeze(0)


transfer = model(content, style)
transfer = utils.add_mean_std(transfer).detach().cpu().numpy()
transfer = (transfer.transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype('uint8')[0]


image = cv2.cvtColor(transfer, cv2.COLOR_BGR2RGB)

cv2.imshow('Style Transfer', image)
cv2.waitKey()