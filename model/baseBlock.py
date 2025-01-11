import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import torch.nn.init as init


class STN(nn.Module):
    @staticmethod
    def gen_Theta_0(inSize,p1,p2):
        """
        outSize:(y,x)
        p1,p2:(y,x)rectfangle clip down=object box pos in origin img
        """
        assert len(inSize)==len(p1)==len(p2)
        p1=[2*p1[i]/inSize[i]-1 for i in range(len(p1))]
        p2=[2*p2[i]/inSize[i]-1 for i in range(len(p2))]
        return STN.gen_Theta0_relative(p1, p2) 

    @staticmethod
    def gen_Theta0_relative(p1, p2):
        '''
        p1,p2:(y,x)~(-1,1).rectfangle crop down=object box relative pos in origin img
        '''
        y, x = [(p2[i] + p1[i]) / 2 for i in range(len(p1))]
        b, a = ((p2[i] - p1[i]) / 2 for i in range(len(p1)))
        return [a,0,x,
                0,b,y]#grid point:(x,y)～[-1,1]
    def __init__(self,L_net:nn.Module,L_channels:int,outSize,theta_0=[1, 0, 0, 0, 1, 0],detach:bool=True,fc_loc_init="zeros"):
        '''
        STN=localization net+grid affine+sample
        L_net:fm=>(bz,L_channels)=(fc_loc)=>theta:(bz,6)
        grid affine:theta=>pos grid
        theta_0:initial theta
        detach: detach L_net backprop to fm
        '''
        super(STN, self).__init__()
        self.outSize,self.detach=outSize,detach
        # fm=>(bz,L_channels)
        self.localization = L_net
        # [bz,L_channels]=>[bz,6]
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2)
        )
        if fc_loc_init=="normal":
            init.normal_(self.fc_loc[0].weight, mean=0.0, std=1e-1)
        elif fc_loc_init=='kaiming':
            init.kaiming_normal_(self.fc_loc[0].weight, nonlinearity='relu')
        elif fc_loc_init=="xavier":
            init.xavier_normal_(self.fc_loc[0].weight, gain=1.0)
        elif fc_loc_init=="zeros":
            self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))

    def forward(self, x):
        # predict theta
        xs = self.localization(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # gen affine_grid
        grid = F.affine_grid(theta, [*x.shape[0:2],*self.outSize])
        # sample
        x = F.grid_sample(x, grid)
        return x
    def forward_gen_fm_theta(self,x):
        # predict theta
        xs = self.localization(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # gen affine_grid
        grid = F.affine_grid(theta, [*x.shape[0:2],*self.outSize])
        # sample
        x = F.grid_sample(x, grid)
        return x,theta
    pass

class STN_projective(nn.Module):
    Theta_identical = [1, 0, 0,
                       0, 1, 0,
                       0, 0]
    @staticmethod
    def gen_Theta0(inSize,p1,p2):
        return STN.gen_Theta_0(inSize,p1,p2)+[0,0]
    @staticmethod
    def gen_Theta0_relative(p1, p2):
        return STN.gen_Theta0_relative(p1, p2)+[0,0]
    @staticmethod
    def gen_grid(H:int, W:int):
        "gen original grid"
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))  # [H, W]
        # x, y = x.flatten(), y.flatten()  # [N]
        ones = torch.ones_like(x)  # [H,W]
        grid = torch.stack([x, y, ones], dim=0)
        return grid# [3,W,H]
    def __init__(self,L_net:nn.Module,L_channels:int,outSize,theta_0=Theta_identical,detach:bool=True):
        '''
        projective transform STN, instead of affin trans
        L_net:fm=>(bz,L_channels)=(fc_loc)=>theta:(bz,8)
        grid gen:theta=>pos grid
        theta_0:initial theta
        detach: detach L_net backprop to fm
        '''
        super().__init__()
        self.outSize,self.detach=outSize,detach
        # fm=>(bz,L_channels)
        self.localization = L_net
        # [bz,L_channels]=>[bz,8]
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 8)
        )
        # init theta=theta_0
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))
        self.register_buffer('grid_origin',self.gen_grid(*self.outSize),persistent=False)
        return
    def forward(self, x):
        # predict theta
        xs = self.localization(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)
        theta = torch.concat((theta, torch.ones(theta.size(0), 1,device=theta.device)), dim=-1)
        theta = theta.view(-1, 3, 3)
        # gen affine_grid
        grid = self.projective_tranform(theta,self.grid_origin)
        # sample
        x = F.grid_sample(x, grid)
        return x
    @staticmethod
    def projective_tranform(theta,grid):
        '''
        theta:(B,3,3),grid:(3,W,H)
        '''
        tgtGrid=torch.einsum('bij,jxy->bxyi',theta,grid)
        # tgtGrid=theta@grid#B,3,H,W
        # tgtGrid.permute(0,2,3,1)
        tgtGrid_2d=tgtGrid[...,:2]/tgtGrid[...,2:]#p=(x,y)/z
        return tgtGrid_2d #(B,H,W,2)
    pass 

class STN_tanh(STN):
    def __init__(self, L_net, L_channels, outSize, theta_0=[1, 0, 0, 0, 1, 0], detach = False):
        super().__init__(L_net, L_channels, outSize, theta_0, detach)
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2,bias=False),
            nn.Tanh()
        )
        # init theta=theta_0
        self.fc_loc[0].weight.data.zero_()
        self.fc_bias=nn.Parameter(torch.tensor([1,0,0,
         0,1,0],dtype=torch.float))
        return
    def forward(self, x):
        # predict theta
        xs = self.localization(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)+self.fc_bias
        theta = theta.view(-1, 2, 3)
        # gen affine_grid
        grid = F.affine_grid(theta, [*x.shape[0:2],*self.outSize])
        # sample
        x = F.grid_sample(x, grid)
        return x

class resnetBasicBlock(nn.Module):
    def __init__(self, inChannel:int,outChannel:int,stride:int=2):
        super().__init__()
        self.basicBlock = models.resnet.BasicBlock(
            inChannel,outChannel,stride,
            downsample=nn.Sequential(
                nn.Conv2d(inChannel, outChannel, 3, stride, 1), 
                nn.BatchNorm2d(outChannel)
                ),
        )
        return
    def forward(self,x):
        return self.basicBlock(x)
class PosEncode:

    class LearnPosEncoding_2D(nn.Module):
        def __init__(self, d_model, height, width):
            """
            初始化 Learnable 2D Positional Encoding 模块。
            :param d_model: 每个位置的特征维度。
            :param max_height: 最大高度，用于定义位置编码参数的范围。
            :param max_width: 最大宽度，用于定义位置编码参数的范围。
            """
            super().__init__()
            if d_model % 2 == 0:
                dim_x=dim_y=d_model // 2
            else:
                dim_y=d_model // 2
                dim_x=d_model // 2+1
            
            self.d_model = d_model
            self.max_height = height
            self.max_width = width
            
            # 定义 learnable 的 pos_x 和 pos_y
            self.pos_x = nn.Parameter(torch.randn(width, dim_x))  
            self.pos_y = nn.Parameter(torch.randn(height, dim_y))  

        def forward(self, x):
            """
            生成并返回与输入匹配的 2D 位置编码。
            :param x: 输入张量，形状为 (seq_len, batch_size, d_model)。
            :return: 2D 位置编码，形状为 (seq_len, batch_size, d_model)。
            """
            seq_len = x.size(0)
            # batch_size = x.size(1)

            # 获取位置编码并拼接
            pos_x = self.pos_x.unsqueeze(0).repeat(self.max_height, 1, 1)  # (height, width, d_model//2)
            pos_y = self.pos_y.unsqueeze(1).repeat(1, self.max_width, 1)  # (height, width, d_model//2)
            pos_encoding = torch.cat([pos_x, pos_y], dim=-1)  # (height, width, d_model)
            
            # 展平为 (seq_len, d_model)
            pos_encoding = pos_encoding.view(-1, self.d_model)
            
            return pos_encoding[:seq_len].unsqueeze(1)  # 根据序列长度返回

    class learnPosEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000) -> None:
            super().__init__()
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

            return
        def forward(self,x):
            # x = x + self.pe[:x.size(0), :]
            return self.pe[:x.size(0), :]
        pass 

    class sinePosEncoding_2D(nn.Module):
        def __init__(self, d_model, height, width):
            super().__init__()
            self.height = height
            self.width = width
            self.d_model = d_model

            # Create a grid of shape (height, width)
            pos_w = torch.arange(0, width).unsqueeze(0).repeat(height, 1)
            pos_h = torch.arange(0, height).unsqueeze(1).repeat(1, width)

            # Calculate position encodings based on sine and cosine functions
            dim_w=d_model//2
            dim_h=d_model-dim_w
            dim_w_s=dim_w//2
            dim_w_c=dim_w-dim_w_s
            dim_h_s=dim_h//2
            dim_h_c=dim_h-dim_h_s
            dims_w = torch.arange(0, dim_w , 1).unsqueeze(0).unsqueeze(0)
            dims_h = torch.arange(0, dim_h , 1).unsqueeze(0).unsqueeze(0)
            # div_terms = self.gen_terms(d_model, dims)
            pos_w = pos_w.unsqueeze(-1).repeat(1, 1, dim_w)
            pos_h = pos_h.unsqueeze(-1).repeat(1, 1, dim_h)
            # Apply sin to even indices, cos to odd indices
            pos_encoding_w = torch.zeros(height, width, dim_w)
            pos_encoding_h = torch.zeros(height, width, dim_h)
            
            pos_encoding_w[:, :, 0::2] = torch.sin(pos_w * self.gen_terms(dim_w_s,dims_w))[..., 0::2]
            pos_encoding_w[:, :, 1::2] = torch.cos(pos_w * self.gen_terms(dim_w_c,dims_w))[..., 1::2]
            pos_encoding_h[:, :, 0::2] = torch.sin(pos_h * self.gen_terms(dim_h_s,dims_h))[..., 0::2]
            pos_encoding_h[:, :, 1::2] = torch.cos(pos_h * self.gen_terms(dim_h_c,dims_h))[..., 1::2]

            # Concatenate along the depth (last dimension)
            pos_encoding = torch.cat([pos_encoding_w, pos_encoding_h], dim=-1)
            pos_encoding=pos_encoding.view(height*width,d_model)
            self.register_buffer('pe', pos_encoding.unsqueeze(1))

        def gen_terms(self, dim_sine, dims):
            return torch.exp(dims * -(math.log(10000.0) / (dim_sine)))

        def forward(self, x):
            # Add position encoding to input x
            return self.pe[:x.size(0), :]
    class sinePosEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000): # , dropout=0
            super().__init__()
            # self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = (
                torch.cos(position * div_term)
                if d_model % 2 == 0
                else torch.cos(position * div_term)[..., :-1]
            )
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # x = x + self.pe[:x.size(0), :]
            # return self.dropout(x)
            return self.pe[:x.size(0), :]
