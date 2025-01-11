import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .baseBlock import *



class Backbone:

    @staticmethod
    def effitientNet_s16(char_classNum:int=75,pre_weights=models.EfficientNet_B0_Weights.DEFAULT):
        effiNet_b0 = models.efficientnet_b0(weights=pre_weights)
        # s0-s16
        effilayers = list(effiNet_b0.features.children())[:6]
        # effiNet_b0 = nn.Sequential(*effilayers)
        conv_k1 = nn.Sequential(
            nn.Conv2d(112, char_classNum, 1),
            nn.BatchNorm2d(char_classNum),
            nn.ReLU(True),
        )
        return nn.Sequential(*effilayers,conv_k1)
    @staticmethod
    def resnet18_LP32x256_FM2x16(char_classNum:int=75,pre_weights=models.ResNet18_Weights.DEFAULT):
        resnet18 = models.resnet18(weights=pre_weights)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-3])
        conv_k1 = nn.Sequential(
            nn.Conv2d(256, char_classNum, 1),
            nn.BatchNorm2d(char_classNum),
            nn.ReLU(True),
        )
        return nn.Sequential(resnet18,conv_k1)
    @staticmethod
    def resnet50_LP32x256_FM2x16(char_classNum:int=75,pre_weights=models.ResNet50_Weights.DEFAULT):
        resnet50 = models.resnet50(weights=pre_weights)
        resnet50 = nn.Sequential(*list(resnet50.children())[:-3])
        conv_k1 = nn.Sequential(
                    nn.Conv2d(1024, char_classNum, 1),
                    nn.BatchNorm2d(char_classNum),
                    nn.ReLU(True),
        )
        return nn.Sequential(resnet50,conv_k1)
    @staticmethod
    def LP32x256_C32x32_FM2x16(char_classNum:int=75):
        '''Cs32
        img:32,256 to fm:2,16
        '''
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,81,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum)
            )
    @staticmethod
    def LP32x256_FM2x16_STN_8(char_classNum:int=75,stn_detach:bool=False):
        '''Cs32
        LP img:32,256 to fm:2,16
        '''
        L_net=nn.Sequential(
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            nn.Conv2d(75, 1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.Flatten(),
        )
        L_channels=32
        outSize=[4,32]
        stn = STN(L_net,L_channels,outSize,detach=stn_detach)
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            stn,
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            )
    @staticmethod
    def LP32x256_FM2x16_STN_4(char_classNum:int=75,stn_detach:bool=False):
        '''Cs32
        LP img:32,256 to fm:2,16
        '''
        L_net=nn.Sequential(
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            nn.Conv2d(75, 1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.Flatten(),
        )
        L_channels=32
        outSize=[8,64]
        stn = STN(L_net,L_channels,outSize,detach=stn_detach)
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            stn,
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            )
    @staticmethod
    def LP32x256_FM2x16_STN_0(char_classNum:int=75,stn_detach:bool=False):
        '''Cs32
        LP img:32,256 to fm:2,16
        '''
        L_net=nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            nn.Conv2d(75, 1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.Flatten(),
        )
        L_channels=32
        outSize=[32,256]
        stn = STN(L_net,L_channels,outSize,detach=stn_detach)
        return nn.Sequential(
            stn,
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            )
    @staticmethod
    def LP32x256_FM2x16_STN(char_classNum:int=75,stn_detach:bool=False):
        '''Cs32
        LP img:32,256 to fm:2,16
        '''
        L_net=nn.Sequential(
            nn.Conv2d(75, 1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.Flatten(),
        )
        L_channels=32
        outSize=[2,16]
        stn = STN(L_net,L_channels,outSize,detach=stn_detach)
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            stn
            )
    @staticmethod
    def C_32x256_fm_2x16(char_classNum:int=75):
        '''Cs32
        img:32,256 to fm:2,16
        '''
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            )
    @staticmethod
    def C_32x256_fm_1x8(char_classNum:int=75):
        '''Cs32
        img:32,256 to fm:1,8
        '''
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),# s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),# s4
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # s8
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,81,3,2,1), # s16
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,2,1), # s32
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),

            # nn.Conv2d(81,char_classNum,(3,1),(2,1),(1,0)), #2,16
            # nn.ReLU(),
            )
    @staticmethod
    def C_128x256_2x16(char_classNum:int=75):
        '''Cs2
        img:128,256 to fm:2,16
        '''
        return nn.Sequential(
            nn.Conv2d(3,9,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,2,1),
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1), # 16,32
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,81,3,2,1), # 8,16
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,81,3,(2,1),1), #4,16
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,(3,1),(2,1),(1,0)), #2,16
            nn.ReLU(),
            )
    @staticmethod
    def newConv(char_classNum:int=69):
        return nn.Sequential(
            nn.Conv2d(3,9,3,3,1),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,3,3,1),
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(81,64,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(64,char_classNum,3,(2,1),1),
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
        )
    @staticmethod
    def newConv_3_2_5(char_classNum:int=69):
        return nn.Sequential(
            nn.Conv2d(3,9,7,5,3), #s5
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,27,5,3,1),#s3
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.MaxPool2d(3,2,1),#s2
            nn.Conv2d(81,char_classNum,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
            nn.MaxPool2d(3,(2,1),1),#s 2,1
            # nn.Conv2d(64,char_classNum,3,(2,1),1),
            # nn.ReLU(),
            # nn.BatchNorm2d(char_classNum),
        )
    @staticmethod
    def Conv_325_k_2(char_classNum:int=69):
        return nn.Sequential(
            nn.Conv2d(3,9,9,5,4), #s5
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(3,(2,1),1),#s 2,1
            nn.Conv2d(9,27,5,3,1),#s3
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,2,1),#s2
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
        )
    @staticmethod
    def Conv325_k2_x16(char_classNum:int=69):
        return nn.Sequential(
            nn.Conv2d(3,9,9,5,4), #s5
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(3,(2,1),1),#s 2,1
            nn.Conv2d(9,27,5,3,1),#s3
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.Conv2d(27,81,3,(2,1),1),#s2
            nn.ReLU(),
            nn.BatchNorm2d(81),
            nn.Conv2d(81,char_classNum,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(char_classNum),
        )
    @staticmethod
    def JustConv(char_classNum=74): # LPRRRnet backbone
        return nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            nn.Conv2d(9,27,3,1),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(27,81,3,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.Conv2d(81,char_classNum,(1,5),1,(0,1)),# expand kernel feild, to fix O recognition
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    @staticmethod
    def resnet18(pretrained: bool = True):
        """
        [1,3,116,240]=>[1,512, 4, 8]
        """
        if pretrained:
            model = models.resnet18(
                
                weights=None
                #weights=models.ResNet18_Weights.IMAGENET1K_V1
            )  # weights=models.ResNet18_Weights.DEFAULT
        else:
            model = models.resnet18()

        # 删除最后的 avgpool 和 fc 层
        model = nn.Sequential(*list(model.children())[:-2])
        return model
    class Tokenizer_resFM(nn.Module):
        '''
        box=>resnet18=>[9, 1, 1024],1024=2*8*46
        '''
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.l1=nn.Sequential(
            nn.Conv2d(512,64,1,groups=64),
            nn.AdaptiveMaxPool2d((6,24))
        )
            return
        def forward(self,x):
            x=self.l1(x)
            # 使用 unfold 拆分块
            x_unfolded = x.unfold(2, 2, 2).unfold(3, 8, 8)
            # 重塑为 [bz, 64, 9, 16]
            x_C_LP_char = x_unfolded.contiguous().view(x_unfolded.size(0), 64, 9, 16)
            x_C_LP_char = x_C_LP_char.permute(2,0,3,1)
            # 使用 torch.flatten 将后两个维度展平
            x_LP_char = torch.flatten(x_C_LP_char, start_dim=2)
            return x_LP_char.contiguous()
    @staticmethod
    def __res_fm_tokenizer():
        """
        [1, 512, 4, 8]=>[1,64,2,8]

        """
        return nn.Sequential(
            nn.Conv2d(512,64,1,groups=64),
            nn.AdaptiveMaxPool2d((6,24))
            # nn.MaxPool2d((2,1),stride=(2,1))
        )

    @staticmethod
    def __lprrr_no_M_S(char_classNum=74):
        return nn.Sequential(
            nn.Conv2d(3,9,3,1),
            nn.BatchNorm2d(num_features=9),
            nn.ReLU(),
            nn.Conv2d(9,27,3,1),
            nn.BatchNorm2d(num_features=27),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(27,81,3,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(81,81,(3,3),1,groups=9),
            nn.BatchNorm2d(num_features=81),
            nn.Conv2d(81,81,1,1),
            nn.BatchNorm2d(num_features=81),
            nn.ReLU(),
            nn.Conv2d(81,char_classNum,(1,5),1,(0,1)),# expand kernel feild, to fix O recognition
            nn.BatchNorm2d(num_features=char_classNum),
            nn.ReLU(),
        )
    pass


class Neck:
    class Lnet_res_TR(nn.Module):
        def __init__(self, char_classNum=75):
            super().__init__()
            self.bone=nn.Sequential(models.resnet.BasicBlock(char_classNum,64,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(char_classNum,64,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s32
            models.resnet.BasicBlock(64,32,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(64,32,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s64
            models.resnet.BasicBlock(32,16,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(32,16,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s128.[3,6]
                              )
            encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.1)
            self.posEncoding=PosEncode.learnPosEncoding(16,18)
            self.head=nn.TransformerEncoder(encoder_layer, num_layers=2)
            return
        def forward(self,fm):
            fm_s:torch.Tensor=self.bone.forward(fm)
            fm_s=fm_s.flatten(2,-1)
            fm_t=fm_s.permute(2,0,1)#(N,B,C)
            fm_t_pos=fm_t+self.posEncoding(fm_t)
            y=self.head.forward(fm_t_pos)#(18,B,16)
            return y[0]#(B,16)
    @staticmethod
    def Lnet_res_v3(char_classNum):
        return nn.Sequential(# s16
            models.resnet.BasicBlock(char_classNum,64,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(char_classNum,64,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s32
            models.resnet.BasicBlock(64,32,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(64,32,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s64
            models.resnet.BasicBlock(32,16,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(32,16,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s128.[3,6]
            nn.Conv2d(16,8,1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
            )
    @staticmethod
    def Lnet_res_v2(char_classNum):
        return nn.Sequential(# s16
            models.resnet.BasicBlock(char_classNum,64,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(char_classNum,64,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s32
            models.resnet.BasicBlock(64,32,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(64,32,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s64
            models.resnet.BasicBlock(32,16,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(32,16,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s128.[3,6]
            nn.Conv2d(16,8,1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d(3),
            nn.Flatten(),
            )
    @staticmethod
    def STN_s16g132_LnetTR(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_res_TR(char_classNum)
        L_channels=16
        outSize=[6,22]
        theta_0=STN.gen_Theta0_relative((-1/3,-0.5),(1/3,0.5))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STN_s16g132_Lnet(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_res_v3(char_classNum)
        L_channels=72
        outSize=[6,22]
        theta_0=STN.gen_Theta0_relative((-1/3,-0.5),(1/3,0.5))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())

    
    @staticmethod
    def STN_s16g270_Lnet(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
            models.resnet.BasicBlock(char_classNum,64,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(char_classNum,64,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s32
            models.resnet.BasicBlock(64,32,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(64,32,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s64
            models.resnet.BasicBlock(32,16,2,
                                     downsample=nn.Sequential(
                                  nn.Conv2d(32,16,3,1,1),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(3,2,1),
                              ),norm_layer=nn.BatchNorm2d),#s128.[3,6]
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            )
        L_channels=16
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STNprojective_s16g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN_projective.gen_Theta0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN_projective(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STN_s8s16g66_8c_res50(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s8 fm[128,37,90]
            nn.Conv2d(512,128,3,2,1), #s16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,32,3,2,1), #s32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=1), # channel down=>fm[2,2,9]
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=144
        outSize=[12,90]
        theta_0=STN.gen_Theta_0(inSize=(37,90),p1=(12,0),p2=(24,90))
        stnS8 = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)

        L_net=nn.Sequential(#[6,45] =s3=> [2,15]
            nn.Conv2d(char_classNum,32,5,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,8,1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=240
        outSize=[3,22]
        stnS16=STN(L_net,L_channels,outSize)
        return nn.Sequential(stnS8,stnS16,Neck.flate())
    @staticmethod
    def STN_s8s16g66(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s8 fm[128,37,90]
            nn.Conv2d(128,64,3,2,1), #s16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[12,90]
        theta_0=STN.gen_Theta_0(inSize=(37,90),p1=(12,0),p2=(24,90))
        stnS8 = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)

        L_net=nn.Sequential(#[6,45] =s3=> [2,15]
            nn.Conv2d(char_classNum,32,5,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,2,1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=60
        outSize=[3,22]
        stnS16=STN(L_net,L_channels,outSize)
        return nn.Sequential(stnS8,stnS16,Neck.flate())
    @staticmethod
    def STN_s16g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STN_s16g270_Iinit(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        stn = STN(L_net,L_channels,outSize,detach=stn_detach)
        return nn.Sequential(stn,Neck.flate())

    @staticmethod
    def STN_s0g270_res(char_classNum:int=75,stn_detach:bool=True):
        resnet18=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18=nn.Sequential(*list(resnet18.children())[:-2])
        L_net=nn.Sequential(#img(3,348,720)
            resnet18, #s32,fm(512,10,23)
            nn.Conv2d(512,128,3,2,1), #s64
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,32,3,2,1), #s128
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.ReLU(True),
            nn.BatchNorm2d(2),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[96,720]
        theta_0=STN.gen_Theta_0(inSize=(290,720),p1=(96,0),p2=(192,720))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    
    @staticmethod
    def STN_s0g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s0 fm[32,290, 720]
            nn.Conv2d(3,9,5,2,2), #s2
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9,64,3,2,1), #s4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,3,2,1), #s8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,3,2,1), #s16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.ReLU(True),
            nn.BatchNorm2d(2),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[96,720]
        theta_0=STN.gen_Theta_0(inSize=(290,720),p1=(96,0),p2=(192,720))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    
    @staticmethod
    def STN_s4g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s4 fm[64,73,180]
            nn.Conv2d(64,128,3,2,1), #s8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,3,2,1), #s16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.ReLU(True),
            nn.BatchNorm2d(2),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[24,180]
        theta_0=STN.gen_Theta_0(inSize=(73,180),p1=(24,0),p2=(48,180))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    
    @staticmethod
    def STN_s8g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s8 fm[128,37,90]
            nn.Conv2d(128,64,3,2,1), #s16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.ReLU(True),
            nn.BatchNorm2d(2),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[12,90]
        theta_0=STN.gen_Theta_0(inSize=(37,90),p1=(12,0),p2=(24,90))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    @staticmethod
    def __STN_s16g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.ReLU(),
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.ReLU(True),
                nn.BatchNorm2d(2),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    
    @staticmethod
    def STN_s16g270tanh(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.ReLU(),
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.ReLU(True),
                nn.BatchNorm2d(2),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN_tanh(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
    @staticmethod
    def STN_resS16g270tanh(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                resnetBasicBlock(char_classNum,64,2),#s32
                resnetBasicBlock(64,32,2),#s64
                resnetBasicBlock(32,16,2),#s128

                nn.Conv2d(16, 2, kernel_size=1),
                nn.ReLU(True),
                nn.BatchNorm2d(2),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN_tanh(L_net,L_channels,outSize,detach=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate()) 
        
    class FPN(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.conv= nn.Sequential(
                nn.Conv2d(75,75,5,3,2),
                nn.ReLU(),
                nn.BatchNorm2d(75)
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(75, 75, 5, stride=3),
                nn.ReLU(),
                nn.BatchNorm2d(75)
            )
            return 
        def forward(self,fm):
            y=self.conv(fm)
            y=self.deconv(y)
            y:torch.Tensor=fm+y[...,1:-1,1:-1]
            return y.flatten(-2,-1)
        pass 
    @staticmethod
    def flate():
        return nn.Flatten(-2,-1)
    class Tokenizer_newConv(nn.Module):
        def __init__(self, charNum:int=69) -> None:
            super().__init__()
            return 
        def forward(self,x:torch.Tensor):
            x=x.flatten(2)
            x=x.permute(2,0,1)
            return x
        pass 
    class Tokenizer_JustConv(nn.Module):
        '''
        box=>resnet18=>[9, 1, 1024],1024=2*8*46
        '''
        def __init__(self, charNum:int=69) -> None:
            super().__init__()
            self.l1=nn.Sequential(
            nn.Conv2d(charNum,64,1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveMaxPool2d((6,24)),
        )
            return
        def forward(self,x):
            x=self.l1(x)
            # 使用 unfold 拆分块
            x_unfolded = x.unfold(2, 2, 2).unfold(3, 8, 8)
            # 重塑为 [bz, 64, 9, 16]
            x_C_LP_char = x_unfolded.contiguous().view(x_unfolded.size(0), 64, 9, 16)
            x_C_LP_char = x_C_LP_char.permute(2,0,3,1)
            # 使用 torch.flatten 将后两个维度展平
            x_LP_char = torch.flatten(x_C_LP_char, start_dim=2)
            return x_LP_char.contiguous()
    @staticmethod
    def bbox(in_c,out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,(8,3),(8,3))
        )
    pass

from .detr_TR import Transformer_at_attn as TR_at_attn
class Head:
    class TR_atAttn(nn.Module):
        # TODO: search tag
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=1) -> None:
            '''
            no decoder causal mask
            '''
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            # pos encoding
            self.posEncode_en = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.LP_len
            )
            # detr at attn transformer
            self.transformer = TR_at_attn(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=nEnLayers,
                num_decoder_layers=nDelayers,
                dim_feedforward=self.d_model * 2,
            )
            return

        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            '''
            fm=fm.permute(2,0,1)
            tgt=tgt.permute(1,0,2)#N,B,C
            en_pos,de_pos=self.posEncode_en(fm),self.posEncode_de(tgt)
            logits,_=self.transformer.forward(src=fm,tgt=tgt,en_pos_embed=en_pos,de_pos_embed=de_pos)
            return logits.permute(1,2,0) #N,B,C=>B,C,N
        pass 

    class TR_seqQ_atAttn(nn.Module):
        # TODO: search tag
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            # pos encoding
            self.posEncode_en = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.LP_len
            )
            # detr at attn transformer
            self.transformer = TR_at_attn(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=nEnLayers,
                num_decoder_layers=nDelayers,
                dim_feedforward=self.d_model * 2,
            )
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.LP_len)
            self.register_buffer('tgt_mask', tgt_mask)
            return

        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            '''
            fm=fm.permute(2,0,1)
            tgt=tgt.permute(1,0,2)#N,B,C
            en_pos,de_pos=self.posEncode_en(fm),self.posEncode_de(tgt)
            tgt_mask=self.tgt_mask if self.training else self.tgt_mask[0:tgt.shape[0],0:tgt.shape[0]]
            logits,_=self.transformer.forward(src=fm,tgt=tgt,en_pos_embed=en_pos,de_pos_embed=de_pos,tgt_mask=tgt_mask)
            return logits.permute(1,2,0) #N,B,C=>B,C,N
        pass 
    class TR_seqQ_attn_sinPos(TR_seqQ_atAttn):
        def __init__(self, char_classNum = 75, fm_len = 16, LP_len = 8, nhead = 2, nEnLayers = 2, nDelayers = 1):
            super().__init__(char_classNum, fm_len, LP_len, nhead, nEnLayers, nDelayers)
            # pos encoding
            self.posEncode_en = PosEncode.sinePosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.sinePosEncoding(
                self.d_model, max_len=self.LP_len
            )
            return
    class TR_seqQ_attn_sin2dPos(TR_seqQ_atAttn):
        def __init__(self, char_classNum = 75, fm_len = 16, LP_len = 8, nhead = 2, nEnLayers = 2, nDelayers = 1):
            super().__init__(char_classNum, fm_len, LP_len, nhead, nEnLayers, nDelayers)
            # pos encoding
            self.posEncode_en = PosEncode.sinePosEncoding_2D(
                self.d_model,height=6,width=45
            )
            self.posEncode_de = PosEncode.sinePosEncoding(
                self.d_model, max_len=self.LP_len
            )
            return
    class TR_seqQ_attn_lr2dPos(TR_seqQ_atAttn):
        def __init__(self, char_classNum = 75, fm_len = 16, LP_len = 8, nhead = 2, nEnLayers = 2, nDelayers = 1):
            super().__init__(char_classNum, fm_len, LP_len, nhead, nEnLayers, nDelayers)
            # pos encoding
            self.posEncode_en = PosEncode.LearnPosEncoding_2D(
                self.d_model,height=6,width=45
            )
            self.posEncode_de = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.LP_len
            )
            return


    class TR_seqQ_mask_lpos(nn.Module):
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            # transformer
            self.posEncode_en = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.LP_len
            )
            self.transformer = nn.Transformer(
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=nEnLayers, 
                num_decoder_layers=nDelayers, 
                dim_feedforward=self.d_model*2, 
            )
            self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.LP_len)
            return
        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            '''
            fm=fm.permute(2,0,1)
            tgt=tgt.permute(1,0,2)#N,B,C
            fm=fm+self.posEncode_en(fm)
            tgt=tgt+self.posEncode_de(tgt)
            logits=self.transformer.forward(fm,tgt,tgt_mask=self.tgt_mask,tgt_is_causal=True)

            return logits.permute(1,2,0) #N,B,C=>B,C,N
        pass 
    TR_seqQ=TR_seqQ_mask_lpos
    class TR_fmQ_ende_sinPosEncode(nn.Module):
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            #
            self.spatialDense = nn.Sequential(
            nn.Linear(fm_len, LP_len),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
            )
            # transformer
            self.posEncode_en = PosEncode.sinePosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.sinePosEncoding(
                self.d_model, max_len=self.LP_len
            )
            self.transformer = nn.Transformer(
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=nEnLayers, 
                num_decoder_layers=nDelayers, 
                dim_feedforward=self.d_model*2, 
            )

            self.channelDense = nn.Linear(self.d_model, char_classNum)
            # =nn.Linear(char_classNum*2,char_classNum,)
            self.normL2=nn.BatchNorm1d(char_classNum)
            return
        def forward(self,fm):
            '''
            fm:(bz,ch,16)
            '''
            # fm=fm.permute(2,0,1)
            B,C,N = fm.size()
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            logits_1s=self.spatialDense(fm)
            logits_2t=fm.permute(2,0,1).contiguous()# (N,B,C)
            # add pos embed
            logits_2t=self.posEncode_en(logits_2t)
            logits_1t=self.posEncode_de(logits_1s.permute(2,0,1).contiguous())
            # Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            y_hat=self.transformer.forward(logits_2t,logits_1t, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask

            y_hat = self.channelDense(y_hat)
            y_hat=y_hat.permute(1,2,0) # B,C,N
            y_hat=F.relu(y_hat)
            y_hat=self.normL2(y_hat)
            return y_hat+logits_1s 
        pass 
    class TR_fmde_de_pos_embed(nn.Module):
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            #
            self.spatialDense = nn.Sequential(
            nn.Linear(fm_len, LP_len),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
            )
            # transformer
            self.posEmbeds_en = nn.Parameter(torch.randn(fm_len, self.d_model))
            self.posEmbeds_de = nn.Parameter(torch.randn(LP_len, self.d_model))
            self.transformer = nn.Transformer(
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=nEnLayers, 
                num_decoder_layers=nDelayers, 
                dim_feedforward=self.d_model*2, 
            )
            # self.q_token = nn.Parameter(torch.randn(self.LP_len, 1,self. d_model))
            self.channelDense = nn.Linear(self.d_model, char_classNum)
            # =nn.Linear(char_classNum*2,char_classNum,)
            self.normL2=nn.BatchNorm1d(char_classNum)
            return
        def forward(self,fm):
            '''
            fm:(bz,ch,16)
            '''
            # fm=fm.permute(2,0,1)
            B,C,N = fm.size()
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            logits_1s=self.spatialDense(fm)
            logits_2t=fm.permute(2,0,1).contiguous()# (N,B,C)
            # add pos embed
            pos_embeds_en = self.posEmbeds_en.unsqueeze(1).expand(-1, B, -1)
            pos_embeds_de = self.posEmbeds_de.unsqueeze(1).expand(-1, B, -1)
            logits_2t = logits_2t + pos_embeds_en
            logits_1t=logits_1s.permute(2,0,1).contiguous()+pos_embeds_de
            # Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            y_hat=self.transformer.forward(logits_2t,logits_1t, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask

            y_hat = self.channelDense(y_hat)
            y_hat=y_hat.permute(1,2,0) # B,C,N
            y_hat=F.relu(y_hat)
            y_hat=self.normL2(y_hat)
            return y_hat+logits_1s 
        pass 
    class TRende(nn.Module):
        def __init__(self, classNum_char: int,d_model:int=1024,LP_len:int=8,fm_len:int=9) -> None:
            super().__init__()
            self.d_LP,self.LP_len,self.FM_len=d_model,LP_len,fm_len
            self.d_char=self.d_LP//self.LP_len
            # Positional embeddings
            self.pos_embeds = nn.Parameter(torch.randn(fm_len, d_model))
            # Transformer Model
            self.transformer = nn.Transformer(
                d_model=d_model, 
                nhead=8, 
                num_encoder_layers=2, 
                num_decoder_layers=2, 
                dim_feedforward=2048, 
            )
            self.q_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.fc = nn.Linear(128, classNum_char)
            return
        def forward(self,fm):
            # Add positional embeddings
            N,B,C = fm.size() 
            # if N != self.FM_len:
            #     raise ValueError(f"The input seq length={N}!=fm len={self.FM_len}.")
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            # Expand the positional embeddings to match the batch size
            pos_embeds = self.pos_embeds.unsqueeze(1).expand(-1, B, -1)
            fm = fm + pos_embeds
            Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            out_embed=self.transformer.forward(fm,Q, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask
            out_embed=out_embed.squeeze(0)
            out_embed = out_embed.view(out_embed.size(0), self.LP_len, self.d_char)
            logits = self.fc(out_embed)
            return logits
        pass 
    class TRende_fmQ_res(nn.Module):
        def __init__(self, classNum_char: int,d_model:int=1024,LP_len:int=8,fm_len:int=9,nhead:int=1) -> None:
            super().__init__()
            self.d_LP,self.LP_len,self.FM_len=d_model,LP_len,fm_len
            self.d_char=self.d_LP//self.LP_len
            # Positional embeddings
            self.pos_embeds = nn.Parameter(torch.randn(fm_len, d_model))
            # Transformer Model
            self.transformer = nn.Transformer(
                d_model=d_model, 
                nhead=nhead, 
                num_encoder_layers=1, 
                num_decoder_layers=1, 
                dim_feedforward=d_model*2, 
            )
            self.Q_spacial_fc=nn.Sequential(
                nn.Linear(self.FM_len,1),
                # nn.BatchNorm1d(num_features=d_model),
                nn.ReLU(),
            )

            self.embed2token = nn.Sequential(
                nn.Linear(128, classNum_char),
                nn.LayerNorm(classNum_char),
                nn.ReLU()
            )

            return
        def forward(self,fm):
            # Add positional embeddings
            N,B,C = fm.size() 
            # if N != self.FM_len:
            #     raise ValueError(f"The input seq length={N}!=fm len={self.FM_len}.")
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            # Expand the positional embeddings to match the batch size
            pos_embeds = self.pos_embeds.unsqueeze(1).expand(-1, B, -1)
            fm = fm + pos_embeds
            # gen Q
            Q=fm.permute(1,2,0) #Q (B,C,N)
            Q=self.Q_spacial_fc(Q)
            Q=Q.permute(2,0,1) #Q (N,B,C)
            # Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            out_embed=self.transformer.forward(fm,Q, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask
            out_embed=out_embed+Q
            out_embed=out_embed.squeeze(0)
            out_embed = out_embed.view(out_embed.size(0), self.LP_len, self.d_char)
            logits = self.embed2token(out_embed)
            return logits
        pass 
    class TRde(nn.Module):
        def __init__(self, classNum_char: int) -> None:
            super().__init__()
            decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=8)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.q_token = nn.Parameter(torch.randn(1, 1, 1024))
            self.fc = nn.Linear(128, classNum_char)
            return
        def forward(self, H):
            Q = self.q_token.expand(-1, H.size(1), -1)  # 1,B,C
            output: torch.Tensor = self.decoder(Q, H)
            output = output.squeeze(0)
            output = output.view(output.size(0), 8, 128)
            logits = self.fc(output)
            return logits
    class RNN_ende(nn.Module):
        def __init__(self, char_classNum:int=75,nEnLayers:int=2,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model=char_classNum
            self.rnn_en=nn.GRU(char_classNum,char_classNum*2,nEnLayers,bidirectional=True)
            self.rnn_de=nn.GRU(char_classNum,char_classNum*2,nDelayers)
            self.channelDense=nn.Sequential(
                nn.Linear(char_classNum*2,char_classNum)
            )
            return

        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            '''
            fm=fm.permute(2,0,1)
            tgt=tgt.permute(1,0,2)#N,B,C
            _,memory=self.rnn_en.forward(fm)
            logits,_=self.rnn_de.forward(tgt,memory)
            logits=self.channelDense.forward(logits)
            return logits.permute(1,2,0) #N,B,C=>B,C,N
        pass
        class charTR(nn.Module):
            '''
            [100,bz,69]=>[8,bz,69]
            '''
            def __init__(self, classNum_char: int) -> None:
                super().__init__()
                return 
            def forward(self,x):
                return 
            pass 
    class charConv(nn.Module):
        def __init__(self, lassNum_char: int=69) -> None:
            super().__init__()
            self.l1= nn.Sequential(
            nn.Conv1d(lassNum_char,lassNum_char,3,2),
            nn.ReLU(),
            nn.BatchNorm1d(lassNum_char),
            nn.Conv1d(lassNum_char,lassNum_char,3,2),
            nn.ReLU(),
            nn.BatchNorm1d(lassNum_char),
            nn.Conv1d(lassNum_char,lassNum_char,3,3,1),
            nn.ReLU(),
        )
            return
        def forward(self,x):
            x=x.permute(1,2,0) # B,C,N
            x=self.l1(x)
            x=x.permute(0,2,1) #B,N,C
            return x
        pass
    pass

class LPR_Head:
    class lpr_TR_fmde(nn.Module):
        def __init__(self, char_classNum:int=74,fm_len:int=16,LP_len:int=8,nhead:int=1,nEnLayers:int=1,nDelayers:int=1) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            #
            self.spatialDense = nn.Sequential(
            nn.Linear(fm_len, LP_len),
            nn.BatchNorm1d(num_features=char_classNum),
            nn.ReLU(),
            )
            # transformer
            self.pos_embeds = nn.Parameter(torch.randn(fm_len, self.d_model))
            self.transformer = nn.Transformer(
                d_model=self.d_model, 
                nhead=self.nhead, 
                num_encoder_layers=nEnLayers, 
                num_decoder_layers=nDelayers, 
                dim_feedforward=self.d_model*2, 
            )
            # self.q_token = nn.Parameter(torch.randn(self.LP_len, 1,self. d_model))
            self.channelDense = nn.Linear(self.d_model, char_classNum)
            # =nn.Linear(char_classNum*2,char_classNum,)
            self.normL2=nn.BatchNorm1d(char_classNum)
            return
        def forward(self,fm):
            '''
            fm:(bz,ch,16)
            '''
            # fm=fm.permute(2,0,1)
            B,C,N = fm.size()
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            logits_1s=self.spatialDense(fm)
            logits_2t=fm.permute(2,0,1).contiguous()# (N,B,C)
            logits_1t=logits_1s.permute(2,0,1).contiguous()
            # Expand the positional embeddings to match the batch size
            pos_embeds = self.pos_embeds.unsqueeze(1).expand(-1, B, -1)
            logits_2t = logits_2t + pos_embeds
            # Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            y_hat=self.transformer.forward(logits_2t,logits_1t, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask
            
            y_hat = self.channelDense(y_hat)
            y_hat=y_hat.permute(1,2,0) # B,C,N
            y_hat=self.normL2(y_hat)
            y_hat=F.relu(y_hat)
            return y_hat+logits_1s 
        pass 
    class lprTRfmde_no_res(lpr_TR_fmde):
        def forward(self,fm):
            '''
            fm:(bz,ch,16)
            '''
            # fm=fm.permute(2,0,1)
            B,C,N = fm.size()
            assert N == self.FM_len,f"The input seq length={N}!=fm len={self.FM_len}."

            logits_1s=self.spatialDense(fm)
            logits_2t=fm.permute(2,0,1).contiguous()# (N,B,C)
            logits_1t=logits_1s.permute(2,0,1).contiguous()
            # Expand the positional embeddings to match the batch size
            pos_embeds = self.pos_embeds.unsqueeze(1).expand(-1, B, -1)
            logits_2t = logits_2t + pos_embeds
            # Q = self.q_token.expand(-1, B, -1)  # 1,B,C
            y_hat=self.transformer.forward(logits_2t,logits_1t, tgt_mask=None)# tgt_is_causal=True ,TO use look-ahead mask
            
            y_hat = self.channelDense(y_hat)
            y_hat=y_hat.permute(1,2,0) # B,C,N
            y_hat=self.normL2(y_hat)
            y_hat=F.relu(y_hat)
            return y_hat
        pass 
    pass 


