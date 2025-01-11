# -*- coding: utf-8 -*-

from dataset import CCPD_img_resize,CCPDdataset_img,CCPDdataset_crop,dataset2loader
import dataset
from dataset.chars import CHARS
from torch.utils.data import *
from torch.utils.tensorboard import SummaryWriter

from torch import optim
import torch.nn as nn
import numpy as np
import torch
import time
import os
from dynaconf import Dynaconf

import model
import math

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    TODO:del this func, use optim.lr_scheduler.StepLR :torch api instead.
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class Trainer:
    @staticmethod
    def init_net_backbone_weight(net: nn.Module, pretrained_pth):
        return Trainer_CTC.init_net_backbone_weight(net,pretrained_pth)
    @staticmethod
    def creat_dataloader(args):
        train_dataset = CCPDdataset_crop(args.CCPD_train, args.lpr_max_len)
        val_dataset = CCPDdataset_crop(args.CCPD_val, args.lpr_max_len)
        train_loader = dataset2loader(
            train_dataset,
            args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = dataset2loader(
            val_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return train_loader, val_loader
    @staticmethod
    def creat_net(args):
        try:
            NetClass = getattr(model, args.model_name)
        except:
            raise ValueError(f"args.model_name={args.model_name} can't find in model")
        net = NetClass(len(CHARS))

        print(f"Successful to build {args.model_name} network!")
        return net

    @staticmethod
    def creat_lossFunc(args):
        if args.lossFunc == "CEloss":
            return nn.CrossEntropyLoss()
        elif args.lossFunc == "FocalLoss":
            return model.FocalLoss()
        else:
            raise ValueError("lossFunc error:only CElos/FocalLoss supported")
        
    @staticmethod
    def creat_optim(net,args):
        Trainer_CTC.creat_optim(net,args)
        return Trainer_CTC.creat_optim(net,args)
    @staticmethod
    def init_net_weight(net:nn.Module,args):
        if args.pretrained_model:
            # load pretrained model
            net.load_state_dict(torch.load(args.pretrained_model))
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
    def Tb_add_graph(Tboard_writer,net,args):
        Tboard_writer.add_graph(net,torch.randn(1,3,384,720))
        return
    @staticmethod
    def creat_lr_scheduler(optimizer,args):
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule_step_size, gamma=args.gamma)
    
    def __init__(self,conf_file:str) -> None:
        # load config args
        args = self.args = Dynaconf(settings_files=[conf_file])
        # get dataset loader
        self.train_loader, self.val_loader=self.creat_dataloader(args)
        # init net
        self.net=self.creat_net(args)
        self.init_net_weight(self.net,args)
        print(self.init_net_backbone_weight(self.net,self.args.backbone_pth)) if self.args.backbone_pth else None
        # def loss function
        self.CE_loss=self.creat_lossFunc(args)
        # define optimizer
        self.optimizer=self.creat_optim(self.net,self.args)
        self.scheduler = self.creat_lr_scheduler(self.optimizer,self.args)
        # other preparations
        os.makedirs(args.weight_save_folder, exist_ok=True)
        if args.grad_scale:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.args.tboard:
            self.Tboard_writer = SummaryWriter(args.tb_log_dir)
            # tboard write net graph
            self.Tb_add_graph(self.Tboard_writer,self.net,self.args) if args.tb_net_graph else None
        else:
            self.Tboard_writer =None

        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.net.to(self.device)

        # Continue training for additional epochs
        self.global_step=init_step=args.init_step
        self.add_epochs:int = args.add_epochs
        return

    def eval(self):
        start_time=time.time()
        self.net.eval()
        LP_yesNum_eval,LP_allNum_eval=0,0
        with torch.no_grad():  # 禁用梯度计算，减少显存占用
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
        img, LP_labels, len = batch
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        logits = self.net(img)
        logits = logits # .permute(0, 2, 1)  # B,C,N
        labels_hat, correct_char_count, total_char_count, correct_lp_count, total_lp_count = self.eval_check_output(LP_labels, logits)
        if TB_save_str:
            self._Tb_add_LPtxt(self.Tboard_writer,self.global_step,labels_hat,LP_labels)
        return correct_char_count,total_char_count,correct_lp_count,total_lp_count
    @staticmethod
    def eval_check_output(LP_labels, logits):
        labels_hat = logits.argmax(dim=1)  # B, N
        correct_char_count, total_char_count, correct_lp_count, total_lp_count = Trainer.check_labels(LP_labels, labels_hat)
        return labels_hat,correct_char_count,total_char_count,correct_lp_count,total_lp_count

    @staticmethod
    def check_labels(LP_labels, labels_hat):
        char_match_mat = (labels_hat == LP_labels)
        correct_char_count = (char_match_mat).sum().item()
        total_char_count = LP_labels.numel()
        correct_lp_count = (char_match_mat).all(dim=1).sum().item()  # 完全匹配的车牌数量
        total_lp_count = LP_labels.size(0)
        return correct_char_count,total_char_count,correct_lp_count,total_lp_count
    @staticmethod
    def _Tb_add_LPtxt(Tboard_writer,global_step,labels_hat,LP_labels):
        # 将预测和真实标签翻译为字符串
        predicted_strs = ["".join([CHARS[idx] for idx in pred]) for pred in labels_hat]
        actual_strs = ["".join([CHARS[idx] for idx in label]) for label in LP_labels]
        # 记录到 TensorBoard
        for i, (pred_str, actual_str) in enumerate(zip(predicted_strs, actual_strs)):
            Tboard_writer.add_text(f'Prediction_{i}/compare', f'Predicted: {pred_str} | Truth: {actual_str}',global_step)

    def train_epochs(self):
        for local_epoch in range(self.add_epochs):
            self.net.train()
            self.train_epoch(local_epoch)
            self.scheduler.step()
            pass
        err=self.eval()
        self.save_pth(self.net,self.args,f"{self.global_step}_err{err:.4f}_final")
        return

    def train_epoch(self, epoch):

        # stream = torch.cuda.Stream()
        # with torch.cuda.stream(stream):
        start_time = time.time()
        for i, batchData in enumerate(self.train_loader):

            loss = self.train_step(batchData)
            self.global_step += 1
            # loss = 0.1 * self.train_step(batchData) + 0.9 * loss  # add inrtia =0.9
            end_time = time.time()
            self._period_record(epoch, i, start_time, loss, end_time)
            start_time =time.time()
        return 

    def _period_record(self, epoch, i, start_time, loss,
     end_time):
        '''periodic print,save,eval'''
        if  i % self.args.step_p_tbWrite == 0:
            self.__print_TB_add_step(epoch, start_time, loss, end_time)
        if self.global_step%self.args.step_p_val==0:
            self.eval()
            self.TB_add_weightHistogram(self.net,self.Tboard_writer) if self.args.tb_weight_Histogram else None
        self.save_pth(self.net,self.args,self.global_step) if self.global_step%self.args.step_p_save==0 else None
        return

    def __print_TB_add_step(self, epoch, start_time, loss, end_time):
        print(f"epoch={epoch},global_step={self.global_step},loss={loss:.4f},batch time={end_time - start_time:.4f},lr={self.scheduler.get_last_lr()}") 
        self.Tboard_writer.add_scalar('train/loss', loss, self.global_step) if self.Tboard_writer is not None else None

    def train_step(self, batchData):
        img, LP_lables, len = batchData
        img = img.to(self.device)
        LP_lables = LP_lables.to(self.device)
        logits = self.net(img) # B,C,N
        loss = self.CE_loss(logits, LP_lables)
        # Backward pass (calculate gradients)
        self.optimizer.zero_grad()
        if math.isfinite(loss.item()):#25ms
            loss.backward()
            # 在反向传播后进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip) if self.args.grad_clip else None
        else:
            raise Exception(f"Loss={loss.item()}, stopping training")
        self.optimizer.step()  # Update weights
        return loss.item()

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

from pretrain import LPRtrain_CTC
class Trainer_CTC(LPRtrain_CTC):

    @staticmethod
    def creat_dataloader(args):
        return Trainer.creat_dataloader(args)

    @staticmethod
    def creat_optim(model, args):
        optimizer = optim.Adam(
            [
                {
                    "params": model.bone.parameters(),
                    "lr": args.backbone_lr,
                    'betas': args.adam_betas,
                    'weight_decay': args.weight_decay,
                },
                {
                    "params": model.neck.parameters(),
                    "lr": args.learning_rate,
                    'betas': args.adam_betas,
                    'weight_decay': args.weight_decay,
                },
                {
                    "params": model.head.parameters(),
                    "lr": args.learning_rate,
                    'betas': args.adam_betas,
                    'weight_decay': args.weight_decay,
                },
            ]
        )
        return optimizer

    @staticmethod
    def init_net_backbone_weight(net: nn.Module, pretrained_pth):

        pretrained_state_dict = torch.load(pretrained_pth)
        # load state_dict in bone
        bone_state_dict = {k.replace('bone.', ''): v for k, v in pretrained_state_dict.items() if k.startswith('bone')}
        net.bone.load_state_dict(bone_state_dict)

        return f"backbone init by {pretrained_pth}"
    def __init__(self, conf_file: str) -> None:
        super().__init__(conf_file)
        print(self.init_net_backbone_weight(self.net,self.args.backbone_pth)) if self.args.backbone_pth else None
        return
    def train_step(self, batchData):
        img, LP_lables, length_LP = batchData
        img = img.to(self.device)
        LP_lables = LP_lables.to(self.device)
        length_LP=length_LP.to(self.device)
        if not self.args.autocast:
            loss_item = self.for_backword_fp32(img, LP_lables, length_LP)
        else:
            loss_item = self.for_backword_autocast(img, LP_lables, length_LP)
        return loss_item
    def for_backword_autocast(self, img, LP_lables, length_LP):
        with torch.cuda.amp.autocast():
            loss = self.forward(img, LP_lables, length_LP)
        loss_item=loss.item()
        # Backward pass (calculate gradients)
        if self.args.grad_scale:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            self.backprop(loss, loss_item)
            self.optimizer.step()
        return loss_item

    def forward(self, img, LP_lables, length_LP):
        logits = self.net(img)
        logits = logits.permute(2,0,1)  # B,C,N=>N,B,C
        N,B,C=logits.shape
        input_lengths = torch.full(size=(B,), fill_value=N, dtype=torch.long,device=self.device)
        loss = self.criterion(logits.log_softmax(2), LP_lables,input_lengths,length_LP)
        return loss
    def for_backword_fp32(self, img, LP_lables, length_LP):
        loss = self.forward(img, LP_lables, length_LP)
        loss_item=loss.item()
        # Backward pass (calculate gradients)
        self.optimizer.zero_grad()
        self.backprop(loss, loss_item)
        self.optimizer.step()
        return loss_item

    def backprop(self, loss, loss_item):
        if math.isfinite(loss_item):
            loss.backward()
            # grid clip
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip) if self.args.grad_clip else None
        else:
            raise Exception(f"Loss={loss_item}, stopping training")

    def eval_step(self,batch,TB_save_str:bool=True):
        img, LP_labels, length = batch
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        with torch.no_grad():
            logits = self.net(img) # B,C,N
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
            # num to string
            predicted_strs = ["".join([CHARS[idx] for idx in pred]) for pred in labels_hat_CTCdecoded]
            actual_strs = ["".join([CHARS[idx] for idx in label]) for label in LP_labels_trimmed]
            # write to TensorBoard
            for i, (pred_str, actual_str) in enumerate(zip(predicted_strs, actual_strs)):
                self.Tboard_writer.add_text(f'Prediction_{i}/compare', f'Predicted: {pred_str} | Truth: {actual_str}',self.global_step)
        return None,None,match_LP_num,all_LP_num,
    pass

class Trainer_CE(Trainer):
    @staticmethod
    def Tb_add_graph(Tboard_writer, net, args):
        Tboard_writer.add_graph(net,torch.randn(1,3,384,720),torch.randint(0, len(CHARS), (1,args.lpr_max_len)))
        return  
    def train_step(self, batchData):
        img, LP_lables, len = batchData
        img=img.to(self.device,non_blocking=True)
        # img.to(non_blocking=True,)
        LP_lables = LP_lables.to(self.device,non_blocking=True)
        tgt = self.build_tgt(LP_lables)
        if not self.args.autocast:
            loss_item = self.forword_backprop_fp32(img, LP_lables, tgt)
        else:
            loss_item = self.forword_backprop_autocast(img, LP_lables, tgt)
        return loss_item 

    def forword_backprop_autocast(self, img, LP_lables, tgt):
        with torch.cuda.amp.autocast():
            logits = self.net(img,tgt) # B,C,N
            loss = self.CE_loss(logits, LP_lables)
        loss_item=loss.item()
        if not math.isfinite(loss.item()):
            raise Exception(f"Loss={loss.item()}, stopping training")
        # Backward pass (calculate gradients)
        if self.args.grad_scale:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            self.backprop(loss, loss_item)
            self.optimizer.step()
        return loss_item

    def forword_backprop_fp32(self, img, LP_lables, tgt):
        logits = self.net(img,tgt) # B,C,N
        loss = self.CE_loss(logits, LP_lables)
        loss_item=loss.item()
        # Backward pass (calculate gradients)
        self.optimizer.zero_grad()
        self.backprop(loss, loss_item)
        self.optimizer.step()
        return loss_item

    def backprop(self, loss, loss_item):
        if math.isfinite(loss_item): # loss_item!=torch.nan: 
            loss.backward()
            # grid clip
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip) if self.args.grad_clip else None
        else:
            raise Exception(f"Loss={loss.item()}, stopping training")#loss.detach().item()
    
    def build_tgt(self, LP_labels):
        tgt=torch.cat((torch.zeros(LP_labels.size(0), 1, dtype=LP_labels.dtype,device=self.device),LP_labels[...,:-1] ), dim=1)
        return tgt
    
    def eval_step(self,batch,TB_save_str:bool=True):
        img, LP_labels, len = batch
        LP_labels, logits = self._eval_step_forward(img, LP_labels)  # B,C,N
        labels_hat, correct_char_count, total_char_count, correct_lp_count, total_lp_count = self.eval_check_output(LP_labels, logits)
        if TB_save_str:
            self._Tb_add_LPtxt(self.Tboard_writer,self.global_step,labels_hat,LP_labels)
        return correct_char_count,total_char_count,correct_lp_count,total_lp_count

    def _eval_step_forward(self, img, LP_labels):
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        tgt=self.build_tgt(LP_labels)
        logits = self.net(img,tgt)
        return LP_labels,logits
    pass

class Trainer_img_CE(Trainer_CE):
    @staticmethod
    
    def creat_dataloader(args):

        train_dataset = CCPDdataset_img(args.CCPD_train, args.lpr_max_len)
        val_dataset = CCPDdataset_img(args.CCPD_val, args.lpr_max_len)
        train_loader = dataset2loader(
            train_dataset,
            args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = dataset2loader(
            val_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return train_loader, val_loader
    pass

class Trainer_imgResize_CE(Trainer_CE):
    @staticmethod
    def creat_dataloader(args):
        train_dataset = CCPD_img_resize(args.CCPD_train, args.lpr_max_len,imgSize=args.img_size)
        val_dataset = CCPD_img_resize(args.CCPD_val, args.lpr_max_len,imgSize=args.img_size)
        train_loader = dataset2loader(
            train_dataset,
            args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = dataset2loader(
            val_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return train_loader, val_loader
    pass
class Trainer_imgAugment_CE(Trainer_CE):
    @staticmethod
    def creat_dataloader(args):
        train_dataset = CCPD_img_resize(args.CCPD_train, args.lpr_max_len,imgSize=args.img_size,PreprocFun=dataset.CCPDloader.PreprocFuns.strong_augment(args.img_size))
        val_dataset = CCPD_img_resize(args.CCPD_val, args.lpr_max_len,imgSize=args.img_size)
        train_loader = dataset2loader(
            train_dataset,
            args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = dataset2loader(
            val_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return train_loader, val_loader
    pass

class Trainer_CE_profiler(Trainer_CE):
    # can remove ce 
    def train_epoch(self, epoch ,profiler_step:int=100,profiler_needed:bool=True):
        if not profiler_needed:
            return super().train_epoch()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=18, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "runs/ALPR/your_model_profiler_experiment"
            ),
            record_shapes=True,
            with_stack=True,
        ) as profiler:
            for i, batchData in enumerate(self.train_loader):
                start_time = time.time()
                loss = self.train_step(batchData)
                self.global_step += 1
                end_time = time.time()
                self._period_record(epoch, i, start_time, loss, end_time)
                # 记录 profiler
                profiler.step()
                if i>profiler_step :
                    break 
                pass
            print("profiler finish")
        exit()

        return 

def train_a_conf(Trainer_imgResize_CE, conf_file:str='configs/args_train.yaml'):
    trainer=Trainer_imgResize_CE(conf_file)
    trainer.train_epochs()

def train_configs(Trainer_imgResize_CE, train_a_conf, conf_files):
    for confFile in conf_files:
        print(f"training: {confFile}")
        train_a_conf(Trainer_imgResize_CE, confFile)
 
if __name__ == "__main__":
    # train_a_conf(Trainer_CTC,"configs/full_train/args_train-CTC.yaml")

    conf_files = [
'configs/args_train.yaml'
                  ]
    
    train_configs(Trainer_imgResize_CE, train_a_conf, conf_files)
    # train_configs(Trainer_imgAugment_CE, train_a_conf, conf_files)
    

