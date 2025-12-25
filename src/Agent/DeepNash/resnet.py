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
        
        out = out + res
        return F.relu(out)
        
        
class DeConvResBlock(nn.Module):
    def __init__(
            self, in_channels:int, out_channels: int, 
            stride:int, kernel:int = 3, 
            padding: int = 1,
            sc_in_channels: int = 0
        ):
        super().__init__()
        self.stride = stride
        self.kernel = kernel
        self.padding = padding
        
        
        self.c1 = nn.ConvTranspose2d(in_channels, out_channels//2, kernel, stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//2)
        self.c2 = nn.ConvTranspose2d(out_channels//2, out_channels, kernel, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if sc_in_channels == 0: sc_in_channels = in_channels
        
        self.sc = nn.ConvTranspose2d(sc_in_channels, out_channels//2, 1, stride, bias=False)
        self.sbn = nn.BatchNorm2d(out_channels//2)
        
        self.has_shortcut = (stride != 1 or in_channels != out_channels)
        
        if self.has_shortcut:
            # shortcutも動的paddingが必要なため、Sequentialではなく個別に定義
            self.shortcut_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut = nn.Identity()
        
    
    def forward(self, x:torch.Tensor, skip:torch.Tensor):
        # 目標サイズはskip接続のサイズ
        target_h, target_w = skip.shape[2], skip.shape[3]
        h_in, w_in = x.shape[2], x.shape[3]
        
        # --- c1 (Main Path Upsampling) の output_padding 計算 ---
        # H_out = (H_in - 1)*stride - 2*padding + kernel + output_padding
        # output_padding = Target - ((H_in - 1)*stride - 2*padding + kernel)
        h_out_raw = (h_in - 1) * self.stride - 2 * self.padding + self.kernel
        w_out_raw = (w_in - 1) * self.stride - 2 * self.padding + self.kernel
        
        op_h = target_h - h_out_raw
        op_w = target_w - w_out_raw
        
        out = F.conv_transpose2d(
            x, self.c1.weight, self.c1.bias, 
            self.c1.stride, self.c1.padding, 
            output_padding=(op_h, op_w), 
            groups=self.c1.groups, dilation=self.c1.dilation
        )
        out = self.bn1.forward(out)
        out = F.relu(out)
        
        # --- sc (Skip Connection) の padding / output_padding 計算 ---
        # 入力サイズ(skip) == 目標サイズ(target)
        # stride > 1 の場合、サイズを維持するためにpaddingが必要
        # (H - 1)*S - 2P + K + OP = H
        # 2P - OP = H(S-1) - S + K
        # K=1 なので 2P - OP = H(S-1) - S + 1
        
        rhs_h = target_h * (self.stride - 1) - self.stride + 1
        p_h_sc = (rhs_h + 1) // 2
        op_h_sc = 2 * p_h_sc - rhs_h
        
        rhs_w = target_w * (self.stride - 1) - self.stride + 1
        p_w_sc = (rhs_w + 1) // 2
        op_w_sc = 2 * p_w_sc - rhs_w
        
        sp = F.conv_transpose2d(
            skip, self.sc.weight, self.sc.bias,
            self.sc.stride, padding=(p_h_sc, p_w_sc),
            output_padding=(op_h_sc, op_w_sc),
            groups=self.sc.groups, dilation=self.sc.dilation
        )
        sp = self.sbn.forward(sp)
        
        out = out + sp
        
        out = self.c2.forward(out)
        out = self.bn2.forward(out)
        out = F.relu(out)
        
        # --- shortcut の output_padding 計算 ---
        if self.has_shortcut:
            # shortcutはxを入力とし、targetサイズへ変換 (c1と同様)
            # kernel=1, padding=0
            h_out_sc_raw = (h_in - 1) * self.stride + 1
            w_out_sc_raw = (w_in - 1) * self.stride + 1
            
            op_h_res = target_h - h_out_sc_raw
            op_w_res = target_w - w_out_sc_raw
            
            res = F.conv_transpose2d(
                x, self.shortcut_conv.weight, self.shortcut_conv.bias,
                self.shortcut_conv.stride, self.shortcut_conv.padding,
                output_padding=(op_h_res, op_w_res),
                groups=self.shortcut_conv.groups, dilation=self.shortcut_conv.dilation
            )
            res = self.shortcut_bn.forward(res)
        else:
            res = x
        
        out = out + res
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
        
        self.dcb = DeConvResBlock(
            mid_channels, in_channels, 2, sc_in_channels=in_channels
        )
        
        self.dcbs2 = nn.ModuleList([DeConvResBlock(in_channels, in_channels, 1) for i in range(N)])
        
        
    def forward(self, x: torch.Tensor):
        out = self.c1.forward(x)
        out = self.bn1.forward(out)
        out = F.relu(out)
        
        s1 = out
        for cb in self.cbs1:
            out = cb.forward(out)
        
        s2 = out
        out = self.cb.forward(out)
        
        s3 = out
        for cb in self.cbs2:
            out = cb.forward(out)
        
        for dcb in self.dcbs1:
            out = dcb.forward(out, s3)
            
        out = self.dcb.forward(out,s2)
        
        for dcb in self.dcbs2:
            out = dcb.forward(out,s1)
            
        return out