import torch
import torch.utils.data  as data
import os
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
import tqdm
from PIL import Image
import numpy as np

# Change to your data root directory
root_path = "/content/"
# Depend on runtime setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

train_dataset = ColorHintDataset(root_path, 128)
train_dataset.set_mode("training")

val_dataset = ColorHintDataset(root_path, 128)
val_dataset.set_mode("validation")

train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=True)

# ================== define helper function ==================

class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

# ================== define train and validation ==================

def train(model, train_dataloader, optimizer, criterion, epoch):
  print('\n[Training] epoch {} '.format(epoch))
  model.train()
  losses = AverageMeter()
  
  for i, data in enumerate(train_dataloader):
    
    # if use_cuda:
    l = data["l"].cuda()
    ab = data["ab"].cuda()
    hint = data["hint"].cuda()
    mask = data["mask"].cuda()  # Add mask

    
    # concat
    gt_image = torch.cat((l, ab), dim=1).cuda()
    #print('\n===== img size =====\n', gt_image.shape)
    hint_image = torch.cat((l, hint, mask), dim=1).cuda()  # Add mask
    #print('===== hint size =====\n', hint_image.shape)

    # run forward
    output_ab = model(hint_image)
    loss = criterion(output_ab, gt_image)
    losses.update(loss.item(), hint_image.size(0))

    # compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100==0:
      print('Train Epoch : [{}] [{} / {}]\tLoss{loss.val:.4f}'.format(epoch, i, len(train_dataloader),loss=losses))


def validation(model, train_dataloader, criterion, epoch):
  model.eval()
  losses = AverageMeter()
  
  for i, data in enumerate(val_dataloader):
    
    # if use_cuda:
    l = data["l"].cuda()
    ab = data["ab"].cuda()
    hint = data["hint"].cuda()
    mask = data["mask"].cuda() # Add mask


    # concat
    gt_image = torch.cat((l, ab), dim=1).cuda()
    #print('\n===== img size =====\n', gt_image.shape)
    hint_image = torch.cat((l, hint, mask), dim=1).cuda() # Add mask
    #print('===== hint size =====\n', hint_image.shape)

    # run model and store loss
    output_ab = model(hint_image)
    loss = criterion(output_ab, gt_image)
    losses.update(loss.item(), hint_image.size(0))

    if i%100==0:
      print('Validation Epoch : [{} / {}]\tLoss{loss.val:.4f}'.format(i, len(val_dataloader),loss=losses))
      
    gt_np = tensor2im(gt_image)
    #print('\n===== gt size =====\n', gt_np.shape)
    hint_np = tensor2im(output_ab)
    #print('===== hint size =====\n', hint_np.shape)

    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
    hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)

    os.makedirs('/content/ground_truth',exist_ok=True)
    cv2.imwrite('/content/ground_truth/gt_'+str(i)+'.jpg',gt_bgr)

    os.makedirs('/content/predictions',exist_ok=True)
    cv2.imwrite('/content/predictions/pred_'+str(i)+'.jpg',hint_bgr)
  
  return losses.avg

# ====================================================

model = UnetGenerator() # load model
print(model)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.00025)

epochs = 150
best_losses = 1e10

save_path = './Result'
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, 'validation_model.tar')

model.cuda()
          
for epoch in range(epochs):
  train(model, train_dataloader, optimizer, criterion, epoch)
  with torch.no_grad():
    val_losses = validation(model, val_dataloader, criterion, epoch)
    print('========== LR ==========\n')


  
  if best_losses > val_losses:
    best_losses = val_losses
    torch.save(model.state_dict(), '/content/drive/MyDrive/Myungji/L1Loss/L1Loss-epoch-{}-losses-{:.5f}.pth'.format(epoch + 1, best_losses))
    
