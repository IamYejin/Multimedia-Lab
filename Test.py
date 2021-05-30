import torch
import os
import cv2
import numpy as np
from PIL import Image
from google.colab import drive

def image_save(img, path):
  if isinstance(img, torch.Tensor):
    img = np.asarray(transforms.ToPILImage()(img))
  img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
  cv2.imwrite(path, img)

root_path = "/content/"
result_save_path = "/content/drive/MyDrive/..." # input best loss's model

test_dataset = ColorHintDataset(root_path, 128)
test_dataset.set_mode('testing')
print('Test length : ', len(test_dataset))

test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UnetGenerator().cuda()

# input best loss model
model_path = os.path.join('/content/drive/MyDrive/...[BestModel].pth')
model.load_state_dict(torch.load(model_path))

# =========================================================

def test(model, test_dataloader): 
  
  model.eval() # same as testing mode
  for i, data in enumerate(test_dataloader):
    l = data["l"].cuda()
    # print('\n===== l size =====\n', l.shape) # [1, 1, 128, 128]
    hint = data["hint"].cuda()
    # print('\n===== hint size =====\n', hint.shape) # [1, 2, 128, 128]
    mask = data["mask"].cuda()  # add mask

    file_name = data['file_name']

    with torch.no_grad():
      out = torch.cat((l, hint, mask), dim=1) # add mask
      pred_image = model(out)

      for idx in range(len(file_name)):
        image_save(pred_image[idx], os.path.join(result_save_path, file_name[idx]))

# =========================================================

test(model, test_dataloader)
