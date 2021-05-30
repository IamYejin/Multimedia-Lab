import torch
import torch.nn as nn
import torch.nn.functional as F

# U-net Based Model
class UnetGenerator(nn.Module):
  def __init__(self, norm_layer=nn.BatchNorm2d):
    super(UnetGenerator, self).__init__()

    self.model_down1 = nn.Sequential(
        nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),  # l, a, b, mask --> 4 input channels
        norm_layer(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(64),
    )
    
    self.model_down2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        norm_layer(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(128),
    )

    self.model_down3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
        norm_layer(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(256),
    )

    self.model_down4 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
        norm_layer(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(512),
    )

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    self.model_bridge = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
        norm_layer(1024),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(1024),
        )
    
    self.model_trans1 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        )
     
    self.model_up1 = nn.Sequential(
        nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        )
     
    self.model_trans2 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        )
     
    self.model_up2 = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        )
     
    self.model_trans3 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )
     
    self.model_up3 = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        )
     
    self.model_trans4 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ReLU(),
        )
     
    self.model_up4 = nn.Sequential(
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(64),
        )
     
    self.model_out = nn.Sequential(
        nn.Conv2d(64, 3, 3, 1, 1), # l, a, b --> 3 output channels
        nn.Tanh(),
        )

  def forward(self, input_lab):
    down1 = self.model_down1(input_lab)
    pool1 = self.pool(down1)
    down2 = self.model_down2(pool1)
    pool2 = self.pool(down2)
    down3 = self.model_down3(pool2)
    pool3 = self.pool(down3)
    down4 = self.model_down4(pool3)
    pool4 = self.pool(down4)

    bridge = self.model_bridge(pool4)

    trans1 = self.model_trans1(bridge)
    concat1 = torch.cat([trans1, down4], dim=1)
    up1 = self.model_up1(concat1)
    trans2 = self.model_trans2(up1)
    concat2 = torch.cat([trans2, down3], dim=1)
    up2 = self.model_up2(concat2)
    trans3 = self.model_trans3(up2)
    concat3 = torch.cat([trans3, down2], dim=1)
    up3 = self.model_up3(concat3)
    trans4 = self.model_trans4(up3)
    concat4 = torch.cat([trans4, down1], dim=1)
    up4 = self.model_up4(concat4)

    return self.model_out(up4)
    
