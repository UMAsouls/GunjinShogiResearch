import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvResBlock(nn.Module):
    #-100は壁
    def __init__(self, in_channels:int, out_channels: int, stride:int, kernel:int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels//2, kernel, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.c2 = nn.Conv2d(out_channels//2, out_channels, kernel, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
            # 論文の記述通り、kernel_size=1, stride=stride（ここでは2）で残差接続を変換
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) # DeepNashではBNも含む
            )
        else:
            self.shortcut = nn.Identity()
        
    
    def forward(self, x:torch.Tensor):
        res = self.shortcut.forward(x)
        
        out = self.c1.forward(x)
        out = self.bn1.forward(out)
        out = F.relu(out)
        
        out = self.c2.forward(out)
        out = self.bn2.forward(out)
        out = F.relu(out)
        
        out += res
        return F.relu(out)
        
        
class DeConvResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, stride:int, kernel:int = 3, out_pad: tuple[int,int] = (0,0)):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(in_channels, out_channels//2, kernel, stride, padding=1, bias=False, output_padding=out_pad)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.c2 = nn.ConvTranspose2d(out_channels//2, out_channels, kernel, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.sc = nn.ConvTranspose2d(in_channels, out_channels//2, 1, stride, bias=False, output_padding=out_pad)
        self.sbn = nn.BatchNorm2d(out_channels//2)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, output_padding=out_pad),
                nn.BatchNorm2d(out_channels) # DeepNashではBNも含む
            )
        else:
            self.shortcut = nn.Identity()
        
    
    def forward(self, x:torch.Tensor, skip:torch.Tensor):
        res = self.shortcut.forward(x)
        
        out = self.c1.forward(x)
        out = self.bn1.forward(out)
        out = F.relu(out)
        
        sp = self.sc.forward(skip)
        sp = self.sbn.forward(sp)
        
        out += sp
        
        out = self.c2.forward(out)
        out = self.bn2.forward(out)
        out = F.relu(out)
        
        out += res
        return F.relu(out)
    
class PyramidModule(nn.Module):
    def __init__(self, N:int, M:int, in_channels: int, mid_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        self.cbs1 = nn.ModuleList([ConvResBlock(in_channels, in_channels, 1) for i in range(N)])
        
        self.cb = ConvResBlock(in_channels, mid_channels, 2)
        
        self.cbs2 = nn.ModuleList([ConvResBlock(mid_channels, mid_channels, 1) for i in range(M)])
        self.dcbs1 = nn.ModuleList([DeConvResBlock(mid_channels, mid_channels, 1) for i in range(M)])
        
        self.dcb = DeConvResBlock(mid_channels, in_channels, 2, out_pad=(1,0))
        
        self.dcbs2 = nn.ModuleList([DeConvResBlock(in_channels, in_channels, 1) for i in range(N)])
        
        
    def forward(self, x: torch.Tensor):
        out = self.c1.forward(x)
        out = self.bn1.forward(out)
        out = F.relu(out)
        
        for cb in self.cbs1:
            out = cb.forward(out)
            
        s1 = out
        
        out = self.cb.forward(out)
        s2 = out
        
        for cb in self.cbs2:
            out = cb.forward(out)
        s3 = out
        
        for dcb in self.dcbs1:
            out = dcb.forward(out, s3)
            
        out = self.dcb.forward(out,s2)
        
        for dcb in self.dcbs2:
            out = dcb.forward(out,s1)
            
        return out