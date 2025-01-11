# here we test models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import model

def __test_CE():
    classNum_char=75
    batchSize=32
    maxLength=8
    imgSize=(1160//4,720) #LP=  box= (348,720)
    net=model.STN8_16_Bres18STNs8s16g66_H_tr_at(classNum_char)
    LP = torch.randint(0, classNum_char, (batchSize,maxLength))
    LP = torch.cat((torch.zeros(batchSize, 1, dtype=LP.dtype), LP), dim=1)
    print(f'lp:{LP.shape}')
    img=torch.randn(batchSize,3,*imgSize)
    # forward
    logits=net.forward(img,LP[:,:-1]) # B,C,N
    print(logits.shape)
    # backprop
    criterion = nn.CrossEntropyLoss()
    loss=criterion(logits, LP[:,1:])
    loss.backward()
    print(loss)

def __test_CTC():
    classNum_char=75
    batchSize=32
    maxLength=8
    imgSize=(348,720) #LP=  box= (348,720)
    net=model.CTC_ImgD4_Bres18_stnS16g270_H_tr_at(classNum_char)
    # for name, param in net.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    
    img=torch.randn(batchSize,3,*imgSize)
    # forward
    LP_hat=net(img) # B,C,N
    LP_hat = LP_hat.permute(2,0,1)  # B,C,N=>N,B,C
    LP_hat=LP_hat.log_softmax(2)
    print(LP_hat.shape)

    LP = torch.randint(0, classNum_char, (batchSize,maxLength))
    print(LP.shape)

    criterion = nn.CTCLoss()
    input_len=torch.tensor(batchSize*[16])
    tgt_len=torch.tensor(batchSize*[8])
    loss=criterion.forward(LP_hat, LP,input_len,tgt_len)
    loss.backward()
    print(loss)

    # print("X:", X)
    # print("Y:", Y)
    # X,Y=__gen_seq()
    # y_hat,_=net.head(X,H_0)
    # y_hat=y_hat.permute(0,2,1)
    # criterion = nn.CrossEntropyLoss()
    # loss=criterion(y_hat, Y)
    # loss.backward()
    # print(loss)
def __gen_seq(classNum_char,maxLength,batchSize):
    # 生成随机序列
    X = torch.randint(0, classNum_char, (maxLength, batchSize))
    Y = torch.randint(0, classNum_char, (maxLength, batchSize))
    # 确保X的第一个元素是73
    X[0, :] = 73
    # 确保Y的最后一个元素是73
    Y[-1, :] = 73
    # 确保X的后面的元素与Y的前面的元素相同
    X[1:, :] = Y[:-1, :]
    return X,Y
if __name__=="__main__":
    __test_CE()
    pass