import dataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from dynaconf import Dynaconf
import dataset.chars
from train import Trainer_CE
import time
import cv2
import numpy as np
import os

class Evaluator_CE(Trainer_CE):
    @staticmethod
    def creat_dataloader(args):
        val_dataset = dataset.CCPD_img_resize(args.CCPD_val, args.lpr_max_len,imgSize=args.img_size)
        val_loader = dataset.dataset2loader(
            val_dataset,
            args.test_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return val_loader
    
    def __init__(self,conf_file:str,new_CCPD_val=None) -> None:
        # load config args
        args = self.args = Dynaconf(settings_files=[conf_file])
        if new_CCPD_val is not None:
            args.CCPD_val=new_CCPD_val
        # get dataset loader
        self.val_loader=self.creat_dataloader(args)
        self.valSet_stepNum=self.val_loader.__len__()
        # init net
        self.net=self.creat_net(args)
        self.init_net_weight(self.net,args)
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.net.to(self.device)
        return
    def eval(self):
        start_time=time.time()
        self.net.eval()
        LP_yesNum_eval,LP_allNum_eval=0,0
        for i,batch in enumerate(self.val_loader):
            char_yesNum_step,char_allNum_step,LP_yesNum_step,LP_allNum_step=self.eval_step(batch)
            LP_yesNum_eval+=LP_yesNum_step
            LP_allNum_eval+=LP_allNum_step
            print(f"{i}/{self.valSet_stepNum}",end='\r')
            pass

        LP_errRate=1-(LP_yesNum_eval/LP_allNum_eval)
        
        end_time=time.time()
        self.net.train()
        print(f"eval: LP_errRate={LP_errRate:.4f}, time={(end_time-start_time):.4f}s")
        return LP_errRate
    def eval_step(self,batch,TB_save_str:bool=True):
        img, LP_labels, len = batch
        # forward
        LP_labels, labels_hat = self._eval_step_forward(img, LP_labels)  # B,C,N
        # logits=logits.softmax(dim=1)
        
        correct_char_count, total_char_count, correct_lp_count, total_lp_count = self.check_labels(LP_labels, labels_hat)
        
        return correct_char_count,total_char_count,correct_lp_count,total_lp_count
    def _eval_step_forward(self, img, LP_labels):
        '''
         1 by 1 infer here
        '''
        img, LP_labels=img.to(self.device),LP_labels.to(self.device)
        bz=img.shape[0]
        tgt=torch.zeros(bz,1,dtype=LP_labels.dtype,device=self.device)
        for _ in range(self.args.lpr_max_len):
            # forward
            with torch.no_grad():
                logits = self.net(img,tgt)
                 
            # last token, argmax
            next_token_logits = logits[:, :,-1]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # concat
            tgt = torch.cat([tgt, next_token_id], dim=-1)
        return LP_labels,tgt[...,1:]
    def _eval_step_forward_label_input(self, img, LP_labels):
        """
        input LP number to TR decoder
        """
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        tgt=self.build_tgt(LP_labels)
        logits = self.net(img,tgt)# B,C,N
        return LP_labels,logits.argmax(dim=1) 
    def _eval_step_forward_label_input_gen_theta(self, img, LP_labels):
        """
        input LP number to TR decoder
        """
        img = img.to(self.device)
        LP_labels = LP_labels.to(self.device)
        tgt=self.build_tgt(LP_labels)
        logits,theta = self.net.forward_theta(img,tgt)# B,C,N
        return LP_labels,logits.argmax(dim=1) ,theta
    @staticmethod
    def eval_check_output(LP_labels, tgt):
        return Trainer_CE.check_labels(LP_labels, tgt)
    def build_tgt(self, LP_labels):
        if LP_labels.dim() == 1:  # 1D case
            tgt = torch.cat((
                torch.zeros_like(LP_labels[0], dtype=LP_labels.dtype, device=self.device).unsqueeze(0),
                LP_labels[:-1]
            ), dim=0)
        else:  # ND case
            tgt = torch.cat((
                torch.zeros_like(LP_labels[..., 0], dtype=LP_labels.dtype, device=self.device).unsqueeze(-1),
                LP_labels[..., :-1]
            ), dim=-1)
        return tgt
    pass


class Evaluate_img:
    def __init__(self):
        self.evaluator=Evaluator_CE("configs/args_eval.yaml")
        # self.evaluator.build_tgt =Evaluate_img.build_tgt.__get__(self.evaluator)
        self.PreprocFun=dataset.CCPDloader.PreprocFuns.resize(self.evaluator.args.img_size)
        pass
    def eval_img(self,imgFile,LP_numbers):
        img_int = Image.open(imgFile)
        img_tensor = self.PreprocFun(img_int)
        LP_label =torch.tensor([dataset.chars.CHARS_DICT[c] for c in LP_numbers]) 
        img_tensor=img_tensor.unsqueeze(0)
        LP_label=LP_label.unsqueeze(0)
        LP_labels, labels_hat =self.evaluator._eval_step_forward_label_input(img_tensor,LP_label)
        LPnumber_hat = self.labels2chars(labels_hat)
        return LPnumber_hat

    def eval_img_gen_theta(self, imgFile, LP_numbers):
        img_int = Image.open(imgFile)
        img_tensor = self.PreprocFun(img_int)
        LP_label = torch.tensor([dataset.chars.CHARS_DICT[c] for c in LP_numbers])
        img_tensor = img_tensor.unsqueeze(0)
        LP_label = LP_label.unsqueeze(0)
        LP_labels, labels_hat, theta = (
            self.evaluator._eval_step_forward_label_input_gen_theta(
                img_tensor, LP_label
            )
        )
        LPnumber_hat = self.labels2chars(labels_hat)
        return LPnumber_hat, theta
    @staticmethod
    def theta2corner(theta,cornerPoints=None):
        theta=theta.to("cpu")[0]
        defaultCorners=torch.tensor([[-1,-1],
                     [1,-1],
                     [-1,1],
                     [1,1],])
        if cornerPoints==None:
            cornerPoints=defaultCorners
        else:
            assert(cornerPoints.shape==(4,2),f"cornerPoints.shape=={cornerPoints.shape},should be (4,2)")
            pass
        cornerPoints=torch.concat((cornerPoints,torch.ones(cornerPoints.size(0),1)),dim=-1)
        return cornerPoints@theta.T
    @staticmethod
    def corners_drow_img(imgFile:str,corners,outFile:str="display.jpg"):
        # Load the image
        img = Image.open(imgFile)
        width, height = img.size

        # Convert the relative coordinates from the tensor to pixel coordinates
        # Scale the relative positions (-1, 1) to actual pixel positions
        corners = corners.detach().numpy()  # Convert tensor to numpy array
        pixel_corners = []

        for corner in corners:
            x_pixel = (corner[0] + 1) / 2 * width  # Scale X from [-1, 1] to [0, width]
            y_pixel = (corner[1] + 1) / 2 * height  # Scale Y from [-1, 1] to [0, height]
            pixel_corners.append((x_pixel, y_pixel))

        # Create a drawing object
        draw = ImageDraw.Draw(img)

        # Draw the parallelogram by connecting the four corners
        # draw.polygon(pixel_corners, outline="red", fill=None)  # You can change the color
        Evaluate_img._draw_polygon_(pixel_corners, draw)

        # Display the image with the parallelogram
        img.save(outFile)
        return

    @staticmethod
    def _draw_polygon_(pixel_corners, draw):
        # Draw the top edge (yellow)
        draw.line([pixel_corners[0], pixel_corners[1]], fill="yellow", width=3)
        # Draw the bottom edge (yellow)
        draw.line([pixel_corners[2], pixel_corners[3]], fill="yellow", width=3)

        # Draw the left edge (green)
        draw.line([pixel_corners[0], pixel_corners[3]], fill="green", width=3)
        # Draw the right edge (green)
        draw.line([pixel_corners[1], pixel_corners[2]], fill="green", width=3)
    
    @staticmethod
    def labels2chars( labels_hat):
        LPnumber_hat = ["".join([dataset.chars.CHARS[idx] for idx in pred]) for pred in labels_hat]
        return LPnumber_hat
    
    pass
class HeatmapDrawer:
    @staticmethod
    def show_heatmap_on_img_Light(imgFile, fm_probs, outFile="heatmap_overlay.jpg"):
        # 打开原图像
        original_image = Image.open(imgFile).convert('RGB')
        plt.clf()
        img_np = np.array(original_image)
        # 将图像转换为 HSV 色彩空间
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        # 调整 heatmap 的大小，使其与原图相同
        fm_probs_resized = cv2.resize(fm_probs, (hsv_img.shape[1], hsv_img.shape[0]))
        # 归一化 heatmap，以确保它在合理的范围内
        fm_probs_resized = np.clip(fm_probs_resized, 0, 1)  # 限制在[0, 1]区间
        # 将 heatmap 映射到亮度通道（V通道）
        hsv_img[..., 2] = np.uint8(fm_probs_resized * 255)  # fm_probs 直接控制亮度通道
        # 将修改后的图像转换回 RGB 色彩空间
        final_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        # 保存结果
        final_image = Image.fromarray(final_img)
        final_image.save(outFile)

        # # 显示叠加后的图像
        # plt.imshow(final_img)
        # plt.axis('off')
        # plt.show()
    @staticmethod
    def show_heatmap_on_img_add(imgFile, fm_probs,outFile:str="heatmap_overlay.jpg"):
        original_image=Image.open(imgFile)
        plt.clf()
        # plt.figure(figsize=(8, 8))
        plt.imshow(original_image) 
        # 叠加heatmap，设置透明度alpha值为0.5
        
        fm_probs_resized = cv2.resize(fm_probs, original_image.size)
        plt.imshow(fm_probs_resized, cmap='viridis', alpha=0.5)
        # plt.imshow(fm_probs, cmap='viridis')  # 颜色映射，可以选择其他方案
        # plt.title("Argmax Heatmap")
        plt.axis('off')
        # plt.colorbar()
        plt.savefig(outFile, bbox_inches='tight', pad_inches=0)
        return 
    @staticmethod
    def show_heatmap(imgFile, fm_probs,outFile:str="heatmap_overlay.jpg"):
        original_image=Image.open(imgFile)
        # 清除当前图形，以防止重复绘制
        plt.clf()
        # plt.figure(figsize=(8, 8))
        # plt.imshow(original_image) 
        # 叠加heatmap，设置透明度alpha值为0.5
        fm_probs_resized = cv2.resize(fm_probs, original_image.size)
        plt.imshow(fm_probs_resized, cmap='RdYlBu_r')#'viridis'
        # plt.imshow(fm_probs, cmap='viridis')  # 颜色映射，可以选择其他方案
        # plt.title("Argmax Heatmap")
        plt.axis('off')
        plt.colorbar()
        plt.savefig(outFile, bbox_inches='tight', pad_inches=0)
        return 
    pass


def eval_dataset(Evaluator_CE,CCPD_val=None):
    evaluator=Evaluator_CE("configs/args_eval.yaml",CCPD_val)
    with torch.no_grad():
        errRate= evaluator.eval()
    return errRate

def infer_img_LP( imgFile:str,LP_numbers:str):
    eval0=Evaluate_img()
    LP_hat=eval0.eval_img(imgFile,LP_numbers)
    print(LP_hat)
    return

def infer_img_STN_crop( imgFile:str,LP_numbers:str,outFile:str="display.jpg"):
    eval0=Evaluate_img()
    theta = {}  # 使用字典存储输出
    def hook_fn(module, input, output):
    # 这里可以处理或保存output
        print(output,"fuck you!!!")
        theta["stn0_theta"]=output.view(-1,2,3)
        return
    # 假设 model 是你的神经网络实例
    # eval0.evaluator.net.neck[0].fc_loc
    # layer_name =   # 替换为具体的层名或索引
    hook = eval0.evaluator.net.neck[0].fc_loc.register_forward_hook(hook_fn)
    LP_hat=eval0.eval_img(imgFile,LP_numbers)
    hook.remove()
    grid_corners= eval0.theta2corner(theta["stn0_theta"])
    print(LP_hat,theta,grid_corners)
    eval0.corners_drow_img(imgFile,grid_corners[[0,1,3,2]],outFile)
    return LP_hat,theta,grid_corners

def infer_img_heatmap( imgFile:str,LP_numbers:str,outFile:str="display.jpg"):
    eval0=Evaluate_img()
    tensorBox = {}  # 使用字典存储输出
    def hook_fn(module, input, output):
    # 这里可以处理或保存output
        # print(output,"fuck you!!!")
        tensorBox["bone_fm"]=output.detach()
        return
    def hook_fn1(module, input, output):
    # 这里可以处理或保存output
        # print(output,"fuck you!!!")
        tensorBox["neck_fm"]=output.detach()
        return
    
    hook = eval0.evaluator.net.bone.register_forward_hook(hook_fn)
    hook1=eval0.evaluator.net.neck[0].register_forward_hook(hook_fn1)
    LP_hat=eval0.eval_img(imgFile,LP_numbers)
    hook.remove()
    hook1.remove()
    fm, fm1 = tensorBox["bone_fm"], tensorBox["neck_fm"]
    
    # torch.argmax()
    fm_probs,fm_classes = torch.max(fm.softmax(dim=1), dim=1)  # 在通道维度上做 argmax

    # 如果 batch_size > 1，选择第一个样本（batch 维度）
    fm_probs = fm_probs[0].cpu().numpy()  # 转换为 NumPy 数组
    HeatmapDrawer.show_heatmap_on_img_add(imgFile, fm_probs,outFile.replace('.jpg', '-add.jpg'))
    HeatmapDrawer.show_heatmap_on_img_Light(imgFile, fm_probs,outFile.replace('.jpg', '-light.jpg'))
    HeatmapDrawer.show_heatmap(imgFile, fm_probs,outFile)
    # plt.show()
    return LP_hat,tensorBox



def infer_img_visualization(infer_img_STN_crop, infer_img_heatmap, imgFile, LP_numbers):
    # infer_img( imgFile,LP_numbers)
    infer_img_heatmap( imgFile,LP_numbers,f"./img/heatmap/{os.path.basename(imgFile)}")
    infer_img_STN_crop(imgFile,LP_numbers,f"./img/STNmap/{os.path.basename(imgFile)}")

def CCPD_ranking(Evaluator_CE, eval_dataset):
    testSets=[
        'dataset/CCPD/test_CCPD2019_ccpd_base.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_blur.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_challenge.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_db.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_fn.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_rotate.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_tilt.csv',
        'dataset/CCPD/test_CCPD2019_ccpd_weather.csv',
        'dataset/CCPD/test_CCPD2020_ccpd_green.csv',
        ]
    errRates=[]
    for testset in testSets:
        print(f"testing: {testset}")
        errRate=eval_dataset(Evaluator_CE,testset)
        errRates.append(errRate)
    print(errRates)

if __name__=="__main__":
    # errRate=eval_dataset(Evaluator_CE)
    # CCPD_ranking(Evaluator_CE, eval_dataset)
    imgFile:str="dataset/CCPD/CCPD2019/ccpd_base/018156531.jpg"
    LP_numbers:str="皖A86H07-"

    imgFile:str="dataset/CCPD/CCPD2019/ccpd_rotate/012008791.jpg"
    LP_numbers:str="皖A332Z4-"

    testSamples=[['dataset/CCPD/CCPD2019/ccpd_base/018000000.jpg','皖AJD523-'],
    ['dataset/CCPD/CCPD2019/ccpd_blur/017000001.jpg','皖AH1W30-'],
    ['dataset/CCPD/CCPD2019/ccpd_tilt/011000001.jpg','皖A780X1-'],
    ['dataset/CCPD/CCPD2019/ccpd_challenge/016000000.jpg','闽ARB205-'],
    ['dataset/CCPD/CCPD2020/ccpd_green/test/002000000.jpg','皖AD18217'],]

    for imgSample in testSamples:
        imgFile,LP_numbers = imgSample
        infer_img_visualization(infer_img_STN_crop, infer_img_heatmap, imgFile, LP_numbers)
    pass


