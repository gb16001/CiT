import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .subModel import Backbone, Neck, LPR_Head


class PRE_B_C_k2x16_H_no(nn.Module):
    '''
    img(116,240)=>fm(2,8)
    '''
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone=Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.pool=nn.MaxPool2d(2)
        # self.head=Head.PRE_lpr_TR_fmde(char_classNum=classNum_char)
        return
    def forward(self,x):
        x=self.bone(x)
        x=self.pool(x)
        x=self.neck(x)
        
        return x
    pass


class PRE_B_C_k2_H_TR8(nn.Module):
    '''
    img(116,240)=>fm(2,8)=>LP_logits(74,16)
    '''
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone=Backbone.Conv_325_k_2(classNum_char)
        self.neck=Neck.flate()
        # self.pool=nn.AdaptiveMaxPool2d((1,8))
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char)
        return
    def forward(self,x):
        x=self.bone(x)
        
        x=self.neck(x)
        x=self.head(x)
        return x
    pass

class B_C_k2x16_H_no_CTC(nn.Module):
    '''
    img(116,240)=>fm(2,16)
    '''
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone=Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.pool=nn.MaxPool2d((2,1))
        return
    def forward(self,x):
        x=self.bone(x)
        x=self.pool(x)
        x=self.neck(x)
        return x
    pass
class B_C_k2x16_H_TR_CTC(nn.Module):
    '''
    img(116,240)=>fm(2,16)=>seq:16
    '''
    def __init__(self, classNum_char:int=69) -> None:
        super().__init__()
        self.bone=Backbone.Conv325_k2_x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,x):
        x=self.bone(x)
        x=self.neck(x)
        x=self.head(x)
        return x # B,C,N
    pass

class B_Cs2_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.C_128x256_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_Cs32_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.C_32x256_fm_1x8(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=8,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_Cs32fm32_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.C_32x256_fm_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_Cs32fm32_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.C_32x256_fm_2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass
class B_LP32x256_FM2x16_STN_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_FM2x16_STN(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_LP32x256_FM2x16_STN_detach_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_FM2x16_STN(classNum_char,stn_detach=True)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_LP32x256_FM2x16_STN_d0_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_FM2x16_STN_0(classNum_char,stn_detach=True)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_LP32x256_FM2x16_STN_d4_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_FM2x16_STN_4(classNum_char,stn_detach=True)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_LP32x256_FM2x16_STN_d8_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_FM2x16_STN_8(classNum_char,stn_detach=True)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_LP32FM32C32_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.LP32x256_C32x32_FM2x16(classNum_char)
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass


class B_resnet18_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.resnet18_LP32x256_FM2x16(classNum_char,)# pre_weights=None
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_resnet18_init_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.resnet18_LP32x256_FM2x16(classNum_char,pre_weights=None)# 
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass


class B_effiNetb0_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.effitientNet_s16(classNum_char,)# pre_weights=None
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass

class B_resnet50_H_tr_CTC(nn.Module):
    def __init__(self, classNum_char:int=75) -> None:
        super().__init__()
        self.bone=Backbone.resnet50_LP32x256_FM2x16(classNum_char,)# pre_weights=None
        self.neck=Neck.flate()
        self.head=LPR_Head.lpr_TR_fmde(char_classNum=classNum_char,fm_len=32,LP_len=16)
        return
    def forward(self,img):
        img=self.bone(img)
        img=self.neck(img)
        img=self.head(img)
        return img
    pass


def __test():
    classNum_char=75
    batchSize=32
    maxLength=8
    imgSize=(128//4, 256) #(128, 256) #LP=  box= (348,720)
    img=torch.randn(batchSize,3,*imgSize)
    LP = torch.randint(0, classNum_char, (batchSize,maxLength))
    print(LP.shape)

    net=B_effiNetb0_H_tr_CTC(classNum_char)

    # forward
    LP_hat=net(img).permute(2,0,1) # B,C,N=>N,B,C
    print(LP_hat.shape)


    # 模拟输入长度 (所有输入长度相同)
    input_lengths = torch.full(size=(batchSize,), fill_value=LP_hat.size(0), dtype=torch.long)

    # 模拟目标长度
    target_lengths = torch.randint(1, maxLength, (batchSize,), dtype=torch.long)

    # 将目标拼接成一维数组
    # targets = torch.cat([LP[i, :target_lengths[i]] for i in range(batchSize)])

    # 定义 CTC 损失
    criterion = nn.CTCLoss(blank=0)
    # criterion = nn.CrossEntropyLoss()
    # 计算损失
    loss = criterion.forward(LP_hat.log_softmax(2), LP,input_lengths, target_lengths)
    loss.backward()
    print(loss)
if __name__=="__main__":
    __test()
    pass