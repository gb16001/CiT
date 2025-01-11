from CBLdata import CBLDataLoader,CBLdata2iter,CBLPRD_CE_1Layer_LP,CBLPRD_CE
from CBLdata.CBLchars import CHARS,CHARS_DICT
from dynaconf import Dynaconf
import torch.nn as nn
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import math

import model
class LPRtrain():
    @staticmethod
    def creat_dataset(args):
        train_dataset = CBLPRD_CE_1Layer_LP(args.CBLtrain, args.img_size, args.lpr_max_len)
        test_dataset = CBLPRD_CE_1Layer_LP(args.CBLval, args.img_size, args.lpr_max_len)
        epoch_size = len(train_dataset) // args.train_batch_size
        return train_dataset,test_dataset,epoch_size
    @staticmethod
    def creat_dataloader(args):
        train_dataset,test_dataset,epoch_size=LPRtrain.creat_dataset(args)
        train_iter=CBLdata2iter( train_dataset,
                args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=None
            )
        test_iter=CBLdata2iter(
            test_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=None
        )
        return train_iter,test_iter
    @staticmethod
    def creat_net(model_name):
        try:
            NetClass = getattr(model, model_name)
        except:
            raise ValueError(f"args.model_name={model_name} can't find in model")
        net = NetClass(len(CHARS))

        print(f"Successful to build {model_name} network!")
        return net
    @staticmethod
    def init_net_weight(net:nn.Module,pretrained_model):
        if pretrained_model:
            # load pretrained model
            net.load_state_dict(torch.load(pretrained_model))
            print("load pretrained model successful!")
        elif False:
            # TODO backbone load_state_dict,container weights_init
            None
        else:
            # for idx,m in enumerate(lprnet.modules()):
            #     m.apply(norm_init_weights)
            print("defult initial net weights")
        return 
    @staticmethod
    def creat_lossFunc():
        ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
        CE_loss=nn.CrossEntropyLoss() 
        return CE_loss
    @staticmethod
    def creat_optim(net,args):
        optimizer = (
            optim.RMSprop(
                net.parameters(),
                lr=args.learning_rate,
                alpha=args.RMS_alpha,
                eps=1e-08,
                momentum=args.RMS_momentum,
                weight_decay=args.weight_decay,
            )
            if args.optim == "RMS"
            else optim.Adam(
                net.parameters(),
                lr=args.learning_rate,
                betas=args.adam_betas,
                weight_decay=args.weight_decay,
            )
        )
        return optimizer
    def __init__(self, conf_file: str) -> None:
        # super().__init__(conf_file)
        args = self.args = Dynaconf(settings_files=[conf_file])
        self.train_loader, self.val_loader=self.creat_dataloader(args)
        self.net=self.creat_net(args.model_name)
        self.init_net_weight(self.net,args.pretrained_model)
        self.criterion=self.creat_lossFunc()
        self.optimizer=self.creat_optim(self.net,self.args)
        os.makedirs(args.weight_save_folder, exist_ok=True)
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.net.to(self.device)
        if self.args.tboard:
            self.Tboard_writer = SummaryWriter(args.tb_log_dir)
            # tboard write net graph
            self.Tboard_writer.add_graph(self.net, torch.randn(1,3,*args.img_size).cuda()) if args.tb_net_graph else None
        else:
            self.Tboard_writer =None

        # Continue training for additional epochs
        self.global_step=init_step=args.init_step
        self.add_epochs:int = args.add_epochs
        return
    def train_epochs(self):
        for local_epoch in range(self.add_epochs):
            self.net.train()
            self.train_epoch(local_epoch)
            pass
        self.eval()
        self.save_pth(self.net,self.args,f"{self.global_step}_final")
        return

    def train_epoch(self, epoch):

        for i, batchData in enumerate(self.train_loader):
            start_time = time.time()
            loss = self.train_step(batchData)
            self.global_step += 1
            end_time = time.time()

            # periodic print,save,eval
            if  i % self.args.step_p_tbWrite == 0:
                print(f"epoch={epoch},global_step={self.global_step},loss={loss:.4f},batch time={end_time - start_time:.4f},lr={None}") 
                self.Tboard_writer.add_scalar('train/loss', loss, self.global_step) if self.Tboard_writer is not None else None
                
            # eval and Histogram
            if self.global_step%self.args.step_p_val==0:
                self.eval()
                self.TB_add_weightHistogram(self.net,self.Tboard_writer) if self.args.tb_weight_Histogram else None
            # save pth
            self.save_pth(self.net,self.args,self.global_step) if self.global_step%self.args.step_p_save==0 else None
        return 

    def train_step(self, batchData):
        img, LP_lables, len,LP_class = batchData
        img = img.to(self.device)
        LP_lables = LP_lables.to(self.device)
        logits = self.net(img)
        logits = logits  # B,C,N
        loss = self.criterion(logits, LP_lables)
        # Backward pass (calculate gradients)
        self.optimizer.zero_grad()
        if math.isfinite(loss.item()):
            loss.backward()
            # 在反向传播后进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip) if self.args.grad_clip else None
        else:
            raise Exception(f"Loss={loss.item()}, stopping training")

        self.optimizer.step()  # Update weights
        return loss.item()
    def eval(self):
        # TODO eval net every step_p_val
        start_time=time.time()
        self.net.eval()
        LP_yesNum_eval,LP_allNum_eval=0,0
        for i,batch in enumerate(self.val_loader):
            char_yesNum_step,char_allNum_step,LP_yesNum_step,LP_allNum_step=self.eval_step(batch,TB_save_str=True if i==0 else False)
            LP_yesNum_eval+=LP_yesNum_step
            LP_allNum_eval+=LP_allNum_step
            pass
        
        LP_errRate=1-(LP_yesNum_eval/LP_allNum_eval)
        
        self.Tboard_writer.add_scalar('eval/LP_errRate', LP_errRate, self.global_step)
        end_time=time.time()
        self.net.train()
        print(f"eval: LP_errRate={LP_errRate:.4f}, step={self.global_step}, time={(end_time-start_time):.4f}s")
        return LP_errRate

    def eval_step(self,batch,TB_save_str:bool=True):
        img, LP_labels, len,LP_class = batch
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        logits = self.net(img)
        logits = logits # .permute(0, 2, 1)  # B,C,N
        labels_hat = logits.argmax(dim=1)  # B, N
        char_match_mat = (labels_hat == LP_labels)
        correct_char_count = (char_match_mat).sum().item()
        total_char_count = LP_labels.numel()
        correct_lp_count = (char_match_mat).all(dim=1).sum().item()  # 完全匹配的车牌数量
        total_lp_count = LP_labels.size(0)
        # char_correct_rate = correct_char_count / total_char_count
        if TB_save_str:
            # 将预测和真实标签翻译为字符串
            predicted_strs = ["".join([CHARS[idx] for idx in pred]) for pred in labels_hat]
            actual_strs = ["".join([CHARS[idx] for idx in label]) for label in LP_labels]
            # 记录到 TensorBoard
            for i, (pred_str, actual_str) in enumerate(zip(predicted_strs, actual_strs)):
                self.Tboard_writer.add_text(f'Prediction_{i}/compare', f'Predicted: {pred_str} | Truth: {actual_str}',self.global_step)
        return correct_char_count,total_char_count,correct_lp_count,total_lp_count,

    def TB_add_weightHistogram(self,net,Tboard_writer):

        for name,param in net.named_parameters():
            Tboard_writer.add_histogram(name,param.cpu().data.numpy(),self.global_step)
        return 

    @staticmethod
    def save_pth(net,args,stepNum,printPath:bool=True):
        file_path = os.path.join(args.weight_save_folder, f"model_s{stepNum}.pth")
        torch.save(net.state_dict(),file_path)
        print(f"save pth:{file_path}") if printPath else None
        return file_path
    pass

class LPRtrain_CTC(LPRtrain):
    @staticmethod
    def creat_lossFunc():
        ctc_loss = nn.CTCLoss(blank=CHARS_DICT['-'], reduction='mean') # reduction: 'none' | 'mean' | 'sum'
        return ctc_loss

    def __init__(self, conf_file: str) -> None:
        super().__init__(conf_file)
        return
    def train_step(self, batchData):
        img, LP_lables, length_LP,LP_class = batchData
        img = img.to(self.device)
        LP_lables = LP_lables.to(self.device)
        length_LP=length_LP.to(self.device)
        logits = self.net(img)
        logits = logits.permute(2,0,1)  # B,C,N=>N,B,C
        N,B,C=logits.shape
        input_lengths = torch.full(size=(B,), fill_value=N, dtype=torch.long,device=self.device)
        loss = self.criterion(logits.log_softmax(2), LP_lables,input_lengths,length_LP)
        # Backward pass (calculate gradients)
        self.optimizer.zero_grad()
        if math.isfinite(loss.item()):
            loss.backward()
            # 在反向传播后进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip) if self.args.grad_clip else None
        else:
            raise Exception(f"Loss={loss.item()}, stopping training")

        self.optimizer.step()  # Update weights
        return loss.item()
    @staticmethod
    def CTC_seq_decode(inSeq,blank_token:int=0):
        # Remove consecutive duplicates and blanks (assuming blank index is 0)
        outSeq = []
        prev = None
        for t in inSeq:
            if t != prev and t != blank_token:
                outSeq.append(CHARS[t.item()])
            prev = t
            pass
        outStr = ''.join(outSeq)
        return outStr
    @staticmethod
    def CTC_batch_decode(inTensor,blank_token:int=0):
        out_strList=[]
        for seq in inTensor:
            out_strList.append(LPRtrain_CTC.CTC_seq_decode(seq,blank_token))
            pass
        return out_strList
    @staticmethod
    def ctc_decode(tensor,blank_token:int=0):
        decoded = []
        for seq in tensor:
            new_seq = []
            prev_token = -1
            for token in seq:
                if token != prev_token and token != blank_token:
                    new_seq.append(token.item())
                prev_token = token
            decoded.append(new_seq)
        return decoded
    @staticmethod
    def trim_padding(labels):
        trimmed = []
        for seq in labels:
            # Remove trailing zeros (padding)
            seq = seq.tolist()
            if 0 in seq:
                seq = seq[:seq.index(0)]
            trimmed.append(seq)
        return trimmed
    def eval_step(self,batch,TB_save_str:bool=True):
        img, LP_labels, length,LP_class = batch
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        logits = self.net(img)
        logits = logits # .permute(0, 2, 1)  # B,C,N
        log_probs = logits.log_softmax(dim=1) 
        labels_hat = log_probs.argmax(dim=1)  # B,N 
        # Decode predictions and remove padding
        labels_hat_CTCdecoded = self.ctc_decode(labels_hat)
        LP_labels_trimmed = self.trim_padding(LP_labels)

        # Now compare each sequence in the batch
        all_LP_num = len(labels_hat_CTCdecoded)
        match_LP_num = 0
        for i in range(all_LP_num):
            if labels_hat_CTCdecoded[i] == LP_labels_trimmed[i]:
                match_LP_num += 1
        
        if TB_save_str:
            # 将预测和真实标签翻译为字符串
            predicted_strs = ["".join([CHARS[idx] for idx in pred]) for pred in labels_hat_CTCdecoded]
            actual_strs = ["".join([CHARS[idx] for idx in label]) for label in LP_labels_trimmed]
            # 记录到 TensorBoard
            for i, (pred_str, actual_str) in enumerate(zip(predicted_strs, actual_strs)):
                self.Tboard_writer.add_text(f'Prediction_{i}/compare', f'Predicted: {pred_str} | Truth: {actual_str}',self.global_step)
        return None,None,match_LP_num,all_LP_num,

    pass



if __name__=="__main__":
    trainer=LPRtrain_CTC("configs/args_pretrain.yaml")
    trainer.train_epochs()
