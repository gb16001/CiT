"""
from csv, batch dataloader img => tensor
"""
import numpy as np
import torch
from torch.utils.data import *
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataset.chars import CHARS_DICT

class PreprocFuns:
    @staticmethod
    def strong_augment(imgSize):
        '''imgSize:(y,x)
        '''
        # 定义增强管道
        strong_augment = transforms.Compose(
            [
                transforms.Resize(imgSize),
                # color augment
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # 调整色彩
                # transforms.RandomGrayscale(p=0.1),
                # shape augment
                # transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转
                # transforms.RandomVerticalFlip(p=0.1),
                # transforms.RandomRotation(degrees=30),  # 随机旋转
                # transforms.RandomResizedCrop(size=(224, 224)), # 随机裁剪后调整大小
                transforms.RandomApply(
                    [transforms.RandomAffine(
                        degrees=(-30, 30),  # 随机旋转角度范围
                        translate=(0.3, 0.3),  # 随机平移范围（宽度、高度的比例）
                        scale=(0.8, 1.2),  # 随机缩放范围
                        shear=(-30, 30),  # 随机剪切角度范围
                        fill=0,  # 填充像素值（0 表示黑色）
                    )],
                    p=0.8  # 应用 RandomAffine 的概率
                ),
                # transforms.RandomPerspective(
                #     distortion_scale=0.5, p=0.5
                # ),  # 随机透视变换
                # tensor augment
                transforms.ToTensor(),  # 转为张量
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),  # 标准化imagenet distribution
                # transforms.RandomErasing(p=0.5),               # 随机擦除
            ]
        )
        return strong_augment
    @staticmethod
    def resize(imgSize):
        '''imgSize:(y,x)
        '''
        return transforms.Compose(
                [
                    transforms.Resize(imgSize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )
    pass

class CCPD_img_resize(Dataset):
    '''
    origin img size=(720,1160), y/4=>
    OutimgSize=(720,290)
    '''
    defaltOutImgSize=(290,720)
    @staticmethod
    def rescale(img_int,size):
        return img_int.resize(size)
    
    def __init__(self, csvFile, lpr_max_len, PreprocFun=None, shuffle = False, imgSize=defaltOutImgSize):
        self.df = pd.read_csv(csvFile)
        # 获取列的位置
        keys=["filename","CCPD_path","license_plate"]
        self.col_indexes=[self.df.columns.get_loc(key) for key in keys]
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        self.lp_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = transforms.Compose(
                [
                    transforms.Resize(imgSize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )
        return
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename,CCPD_path,license_plate = self.df.iloc[index, self.col_indexes]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}" # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        img_tensor = self.PreprocFun(img_int)
        license_plate =  license_plate.ljust(self.lp_max_len, '-')# license_plate.len = 7 or 8
        try:
            LP_label =torch.tensor([CHARS_DICT[c] for c in license_plate]) 
        except KeyError as e:
            import warnings
            warnings.warn(f"Character {e.args[0]} not found in CHARS_DICT. Assigning default value 0.")
            LP_label = torch.tensor([CHARS_DICT.get(c, 0) for c in license_plate])
        # if len(LP_label)>8:
        #     print(license_plate)
        # print(img_tensor.shape,LP_label.shape)

        return (
            img_tensor,
            LP_label,
            len(LP_label),            
        )
    
    pass
class OtherDataset(CCPD_img_resize):
    def __init__(self, csvFile, lpr_max_len, PreprocFun=None, shuffle=False, imgSize=CCPD_img_resize.defaltOutImgSize):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize)
        return
    def __getitem__(self, index):
        filename,CCPD_path,license_plate = self.df.iloc[index, self.col_indexes]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}" # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        img_tensor = self.PreprocFun(img_int)
        license_plate =  license_plate.ljust(self.lp_max_len, '-')# license_plate.len = 7 or 8
        try:
            LP_label =torch.tensor([CHARS_DICT[c] for c in license_plate]) 
        except KeyError as e:
            import warnings
            warnings.warn(f"Character {e.args[0]} not found in CHARS_DICT. Assigning default value 0.")
            LP_label = torch.tensor([CHARS_DICT.get(c, 0) for c in license_plate])
        return (
            img_tensor,
            LP_label,
            len(LP_label),            
        )

class CCPDdataset_crop(Dataset):
    def __init__(self,csvFile:str, lpr_max_len:int,PreprocFun=None, shuffle: bool = False,LP_size_x:int=240,LP_size_y:int=116) -> None:
        super().__init__()
        self.LP_size_x,self.LP_size_y=LP_size_x,LP_size_y
        self.df = pd.read_csv(csvFile)
        # 获取列的位置
        keys=["filename","CCPD_path","license_plate",'center_x','center_y']
        # self.filename_index = self.df.columns.get_loc("filename")
        # self.CCPD_path_index = self.df.columns.get_loc("CCPD_path")
        # self.license_plate_index = self.df.columns.get_loc("license_plate")
        # self.col_index=[self.filename_index,self.CCPD_path_index,self.license_plate_index]
        self.col_index=[self.df.columns.get_loc(key) for key in keys]
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        # self.img_size = imgSize
        self.lp_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = transforms.Compose(
                [
                    # transforms.Resize(imgSize[::-1]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )
        return
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename,CCPD_path,license_plate,center_x,center_y = self.df.iloc[index, self.col_index]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}" # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)

        # box crop
        start_x, start_y, end_x, end_y=self.calcu_box_vertex(center_x,center_y,self.LP_size_x,self.LP_size_y)
        sub_img = img_int.crop((start_x, start_y, end_x, end_y))
        # 显示裁剪后的子图
        # sub_img.show()
        # sub_img.save("cropped_image.jpg")
        img_tensor = self.PreprocFun(sub_img)
        license_plate =  license_plate.ljust(self.lp_max_len, '-')# license_plate.len = 7 or 8
        
        LP_label =torch.tensor([CHARS_DICT[c] for c in license_plate]) 

        return (
            img_tensor,
            LP_label,
            len(LP_label),            
        )
    @staticmethod
    def calcu_box_vertex(center_x, center_y, block_x:int,block_y:int,img_width=1160, img_height=720):
        # # 计算块的宽度和高度
        # block_width = img_width // 10
        # block_height = img_height // 3
        
        # 确定中心点所在的块
        cx = center_x // block_x
        cy = center_y // block_y
        
        # 计算 sy，范围只能是 (0,sx,1)
        sx = max(0, min(1, cx - 1))
        
        # 计算 sx，范围是 [0,sy,7]
        sy = max(0, min(7, cy - 1))
        
        # 计算最终的图像坐标
        start_x = sx * block_x
        start_y = sy * block_y
        end_x = start_x + 3 * block_x
        end_y = start_y + 3 * block_y
        
        return start_x, start_y, end_x, end_y
    pass
class CCPDdataset_img(CCPDdataset_crop):
    
    def __getitem__(self, index):
        filename,CCPD_path,license_plate,center_x,center_y = self.df.iloc[index, self.col_index]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}" # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        img_tensor = self.PreprocFun(img_int)
        license_plate =  license_plate.ljust(self.lp_max_len, '-')# license_plate.len = 7 or 8
        
        LP_label =torch.tensor([CHARS_DICT[c] for c in license_plate]) 

        return (
            img_tensor,
            LP_label,
            len(LP_label),            
        )
    pass

class CCPDdataset_clip_x10(CCPDdataset_crop):
    '''
    TODO:del this class, may never use again
    '''
    def __getitem__(self, index):
        filename,CCPD_path,license_plate,center_x,center_y = self.df.iloc[index, self.col_index]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}" # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        # 获取图像尺寸
        width, height = img_int.size

        # 设置裁切 stride 和 box 高度
        stride = 116
        box_height = 3 * stride
        box_width = width

        # 保存裁切后的图像
        box_images = []
        for y in range(0, height-2*stride, stride):
            # 定义每个 box 的坐标 (左, 上, 右, 下)
            box = (0, y, box_width, min(y + box_height, height))
            cropped_img = img_int.crop(box)
            box_images.append(cropped_img)
            # 如果你想要保存图片，可以使用：
            # cropped_img.save(f"delet/box_{y}.png")


        # # box crop
        # start_x, start_y, end_x, end_y=self.calcu_box_vertex(center_x,center_y,self.LP_size_x,self.LP_size_y)
        # sub_img = img_int.crop((start_x, start_y, end_x, end_y))
        # 显示裁剪后的子图
        # sub_img.show()
        # sub_img.save("cropped_image.jpg")
        
        img_tensor = torch.stack([self.PreprocFun(img) for img in box_images])
        license_plate =  license_plate.ljust(self.lp_max_len, '-')# license_plate.len = 7 or 8
        
        LP_label =torch.tensor([CHARS_DICT[c] for c in license_plate]) 

        return (
            img_tensor,
            LP_label,
            len(LP_label),            
        )

def collate_fn(batch):# TODO lpr func, fix to ccpd. add tensor to cuda function
    import torch

    imgs = []
    labels = []
    lengths = []
    lp_classes = []
    for _, sample in enumerate(batch):
        img, label, length, lp_class = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        lp_classes.append(lp_class)
    labels = np.asarray(labels).flatten().astype(int)

    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, lp_classes

def __custom_preproc_on_gpu(img):
    '''
    img:pid image, must run in 'spawn' worker
    mp.set_start_method('spawn')
    '''
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).cuda().float().div(255)
    img_tensor = img_tensor.half()  # 转为半精度
    return img_tensor

def dataset2loader(
    dataset: Dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    collate_fn=None,
):
    return DataLoader(
        dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn,
    )
def __test_module():
    dataset = CCPD_img_resize("dataset/CCPDanno_val_c.csv", lpr_max_len= 8,PreprocFun=PreprocFuns.strong_augment((290,720)))# PreprocFun=custom_preproc_on_gpu
    one_data = dataset[10]
    # dataset:list=>iter
    batch_iterator =dataset2loader(dataset,batch_size=32,num_workers=8,collate_fn=None)
    print(f'loader worker: b32*1')

    import time
    start_time = time.time()  # 记录开始时间
    for i, batch in enumerate(batch_iterator):
        print(i)
        pass
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    print(f"Elapsed time: {elapsed_time} seconds")
    return


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    __test_module()
    pass
