'''
build network for img ALPR here
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .subModel import Backbone, Neck, Head,LPR_Head

class testNet(nn.Module):
    def __init__(self, char_classNum=69,maxLength=8):
        super().__init__()
        self.backbone=Backbone.__lprrr_no_M_S(char_classNum=128)
        self.neck=Neck.bbox(128,char_classNum*maxLength)
    def forward(self, img: torch.Tensor):
        x=self.backbone(img)
        x=self.neck(x) 
        return x
    pass
class Box_TRde(nn.Module):
    def __init__(self, classNum_char:int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            Backbone.resnet18(),
            Backbone.Tokenizer_resFM(),
        )
        self.head=Head.TRde(classNum_char)
        return
    def forward(self,x:torch.Tensor):
        x=self.backbone(x)
        x=self.head(x)
        return x
    pass

class TR_ende_net(nn.Module):
    def __init__(self, classNum_char:int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            Backbone.resnet18(),
            Backbone.Tokenizer_resFM(),
        )
        self.head=Head.TRende_fmQ_res(classNum_char)
        return
    def forward(self,x):
        x=self.backbone(x)
        x=self.head(x)
        return x
    pass
class B_conv_H_TRende_fmQ(nn.Module):
    def __init__(self, classNum_char:int) -> None:
        super().__init__()
        self.backbone = Backbone.JustConv(classNum_char)
        self.neck=Neck.Tokenizer_JustConv(classNum_char)
        self.head=Head.TRende_fmQ_res(classNum_char)
        return
    def forward(self,x):
        fm=self.backbone(x)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits

class B_conv_H_conv(nn.Module):
    def __init__(self, classNum_char:int) -> None:
        super().__init__()
        self.backbone = Backbone.Conv_325_k_2(classNum_char)
        self.neck=Neck.Tokenizer_newConv(classNum_char)
        self.head=Head.charConv(classNum_char)
        return
    def forward(self,x):
        fm=self.backbone(x)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits
    pass 
class B_pre_H_TR(nn.Module):
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone = Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,288,16,nhead=1)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    pass
class B_pre_H_156TR(nn.Module):
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone = Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,288,16,nhead=15,nEnLayers=6,nDelayers=6)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    pass
class B_fix_H_TR(nn.Module):
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone = Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,288,16,nhead=15,nEnLayers=6,nDelayers=6)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    
    def freeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = False

    def unfreeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = True
    def train(self, mode: bool = True) :
        super().train(mode)
        self.freeze_bone()
        return 
    pass

class B_pre_H_156TR_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone = Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,288,8,nhead=15,nEnLayers=6,nDelayers=6)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    pass
class B_freez_H_156TR_CE(B_pre_H_156TR_CE):
    def freeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = False

    def unfreeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = True
        return
    
    def train(self, mode: bool = True) :
        super().train(mode)
        self.freeze_bone()
        return 

class B_pre_H_156TR_noRes_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone = Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lprTRfmde_no_res(classNum_char,288,8,nhead=15,nEnLayers=6,nDelayers=6)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    pass
class B_freez_H_156TR_noRes_CE(B_pre_H_156TR_noRes_CE):
    def freeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = False

    def unfreeze_bone(self):
        for param in self.bone.parameters():
            param.requires_grad = True
        return
    
    def train(self, mode: bool = True) :
        super().train(mode)
        self.freeze_bone()
        return 
    pass
class B_Cs2_H_tr_ctc(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,270,16,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head(fm)
        return logits #B,C,N
    pass
class B_Cs2_H_tr_dePosEmbed_ctc(B_Cs2_H_tr_ctc):
    def __init__(self, classNum_char: int = 75) -> None:
        super().__init__(classNum_char)
        self.head=Head.TR_fmde_de_pos_embed(classNum_char,270,16,nhead=15,nEnLayers=2,nDelayers=2)
    pass
class B_Cs2_H_tr_sinPosEncode_ctc(B_Cs2_H_tr_ctc):
    def __init__(self, classNum_char: int = 75) -> None:
        super().__init__(classNum_char)
        self.head=Head.TR_fmQ_ende_sinPosEncode(classNum_char,270,16,nhead=15,nEnLayers=2,nDelayers=2)
    pass

class B_Cs2_H_tr_mask_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_mask_lpos(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char) #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class B_Cs2_H_tr_atAttn_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class B_Cs2_H_tr_atAttn_4layers_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=4,nDelayers=4)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class B_Cs2_H_tr_atAttn_6_CE(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=6,nDelayers=6)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class B_Cs2_N_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.FPN()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class img_B_Cs2_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,855,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class Img_d4_B_Cs32_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.C_32x256_fm_1x8(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,230,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class ImgD4_B_LP32x256_C32_FM2x16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.LP32x256_C32x32_FM2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,855,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class ImgD4_B_resnet18_FM2x16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,855,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass


class ImgD4_B_res18stnresS16g270tanh_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_resS16g270tanh()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass


class STNtach_Bres18_stng132resLnet_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g132_Lnet(stn_detach=False)
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N

class STNLnetTR_Bres50_stng132LnetTR_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet50_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g132_LnetTR()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    
class STNLnetTR_Bres18_stng132LnetTR_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g132_LnetTR()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N

class STNg132_Bres50_stng132resLnet_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet50_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g132_Lnet()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N

class STNg132_Bres18_stng132resLnet_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g132_Lnet()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    
class STNLnet_Bres18_stnresLnet_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270_Lnet()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N

class STNprojective_Bres18_stnprojective_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STNprojective_s16g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N


############ search exp models ########
#-------------stn init-----------------
class STN_Iinit_ImgD4_B_res18stnS16g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270_Iinit()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
#-------------stn location-------------
class __base_ImgD4_B_res18stnS16g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class STN8_16_Bres50STNs8s16g66_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet50_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s8s16g66_8c_res50()
        self.head=Head.TR_seqQ_atAttn(classNum_char,66,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm = self.bone[0][0:6](img)  # s8 fm[128,37,90]
        fm=self.neck[0](fm) #STN
        fm = self.bone[0][6](fm) #s16
        fm = self.bone[1](fm)
        fm = self.neck[1:](fm) #STN&flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass

class STN8_16_Bres18STNs8s16g66_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s8s16g66()
        self.head=Head.TR_seqQ_atAttn(classNum_char,66,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm = self.bone[0][0:6](img)  # s8 fm[128,37,90]
        fm=self.neck[0](fm) #STN
        fm = self.bone[0][6](fm) #s16
        fm = self.bone[1](fm)
        fm = self.neck[1:](fm) #STN&flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass


class STN8_B_res18stnS8g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s8g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm = self.bone[0][0:6](img)  # s8 fm[128,37,90]
        fm=self.neck[0](fm) #STN
        fm = self.bone[0][6](fm) #s16
        fm = self.bone[1](fm)
        fm = self.neck[1](fm) #flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass

class STN4_B_res18stnS4g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s4g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm = self.bone[0][0:5](img)  # s4 fm[64,73,180]
        fm=self.neck[0](fm) #STN
        fm = self.bone[0][5:](fm) #s16
        fm = self.bone[1](fm)
        fm = self.neck[1](fm) #flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass

class STN0_B_res18stnS0g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s0g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.neck[0](img) #STN
        fm=self.bone(fm)
        # fm = self.bone[0][0:5](img)  # s4 fm[64,73,180]
        # fm = self.bone[0][5:](fm) #s16
        # fm = self.bone[1](fm)
        fm = self.neck[1](fm) #flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass
class ImgD4_B_res18stnS16g270tanh_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270tanh()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

class ImgD4_B_res18stnS0g270res_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s0g270_res()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.neck[0](img) #STN
        fm=self.bone(fm)
        fm = self.neck[1](fm) #flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass
#---------------STN tach-------------
class STNtach_ImgD4_B_res18stnS16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270(stn_detach=False)
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
#---------------no STN---------------
class noSTN_ImgD4_B_res50_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet50_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,855,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class noSTN_ImgD4_B_res18_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=Head.TR_seqQ_atAttn(classNum_char,855,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass

#---------------pos embed------------
class lr2dPos_Bres18_stnS16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_attn_lr2dPos(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class sin2dPos_Bres18_stnS16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_attn_sin2dPos(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class sinPos_Bres18_stnS16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_attn_sinPos(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
#-----------decoder input-----------
class fmRes_Bres18_stnS16g270_H_tratt_fmRes(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        self.spatialDense = nn.Sequential(
            nn.Linear(270, 8),
            nn.BatchNorm1d(num_features=classNum_char),
            nn.ReLU(),
            )
        self.channelDense = nn.Linear(classNum_char, classNum_char)
        self.normL2=nn.BatchNorm1d(classNum_char)
        return
    def forward(self,img,tgt_):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_BCN=self.spatialDense(fm)
        tgt=tgt_BCN.permute(0,2,1)#B,N,C
        # tgt=torch.zeros((*tgt.shape,self.classNum_char))#B,N,C
        logits=self.head.forward(fm,tgt)#B,C,N
        logits=logits.permute(0,2,1)#B,N,C
        logits=self.channelDense(logits)
        logits=logits.permute(0,2,1)#B,C,N
        logits=self.normL2(logits)
        logits=F.relu(logits,True)
        return tgt_BCN+logits #B,C,N
    pass
class fmde_Bres18_stnS16g270_H_tratt_fmde(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        self.spatialDense = nn.Sequential(
            nn.Linear(270, 8),
            nn.BatchNorm1d(num_features=classNum_char),
            nn.ReLU(),
            )
        return
    def forward(self,img,tgt_):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt=self.spatialDense(fm)#B,C,N
        tgt=tgt.permute(0,2,1)#B,N,C
        # tgt=torch.zeros((*tgt.shape,self.classNum_char))#B,N,C
        logits=self.head.forward(fm,tgt)
        return logits #B,C,N
    pass
class pos_Bres18_stnS16g270_H_tr_at_pos(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt=torch.zeros((*tgt.shape,self.classNum_char),device=tgt.device)#B,N,C
        logits=self.head.forward(fm,tgt)
        return logits #B,C,N
    pass
class RNN_Bres18_stnS16g270_H_rnn(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.RNN_ende(classNum_char,nEnLayers=2,nDelayers=4)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
#----------------tr/tr attn------------
class oTR_Bres18_stnS16g270_H_tr(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
#----------------loss----------------
class base_Bres18_stnS16g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    def forward_theta(self,img,tgt):
        fm=self.bone(img)
        fm,theta=self.neck[0].forward_gen_fm_theta(fm)
        fm=self.neck[-1](fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits,theta #B,C,N
    pass

class CTC_ImgD4_Bres18_stnS16g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        
        self.head=LPR_Head.lpr_TR_fmde(classNum_char,270,16,15,2,2)
        return
    def forward(self,img):
        fm=self.bone(img)
        fm=self.neck(fm)
        logits=self.head.forward(fm)
        return logits #B,C,N
    pass
#--------------backbone---------------
class res50_Bres50_stnS16_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet50_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class ImgD4_B_effb0_stnS16g270_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.effitientNet_s16(classNum_char)
        self.neck=Neck.STN_s16g270()
        self.head=Head.TR_seqQ_atAttn(classNum_char,270,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm=self.bone(img)
        fm=self.neck(fm)
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float() #B,N,C
        logits=self.head.forward(fm,tgt_one_hot)
        return logits #B,C,N
    pass
class STN8_16_Bres18STNs8s16g66Adap_H_tr_at(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.classNum_char=classNum_char
        self.bone = Backbone.resnet18_LP32x256_FM2x16(classNum_char)
        self.neck=Neck.STN_s8s16g66_Adaptive()
        self.head=Head.TR_seqQ_atAttn(classNum_char,66,8,nhead=15,nEnLayers=2,nDelayers=2)
        return
    def forward(self,img,tgt):
        fm = self.bone[0][0:6](img)  # s8 fm[128,37,90]
        fm=self.neck[0](fm) #STN
        fm = self.bone[0][6](fm) #s16
        fm = self.bone[1](fm)
        fm = self.neck[1:](fm) #STN&flatten
        tgt_one_hot = F.one_hot(tgt, self.classNum_char).float()  # B,N,C
        logits = self.head.forward(fm, tgt_one_hot)
        return logits  # B,C,N
    pass
