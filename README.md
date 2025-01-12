# CiT
Character is also token: Detection-Free Automatic License Plate Recognition Using Vision Transformers. 

fix ALPR with license plate string labels only. 

![graphical abstract](dataset/CCPD/g-abstract.png)

## Usage
### install requirements
`pip install -r requirements.txt`

### train backbone in LPR task (opt.)
- download CBLPRD dataset and put in CBLdata with csv annoation file
- run pretrain.py script
  - check configs/args_pretrain.yaml. make sure `CBLtrain;CBLvalmodel_name` keys is correct.
  - `python pretrain.py`

### Train 

- download [CCPD](https://github.com/detectRecog/CCPD) and put at `dataset/CCPD`
- generate csv file as exampled `dataset/CCPD/CCPDanno_train.csv`. 
filename,CCPD_path,license_plate is needed columns. 
- modify `configs/args_train.yaml`, make sure CCPD_train,CCPD_val and model_name is what you want.
- `python train.py`

### Eval
run eval.py with `configs/args_eval.yaml`

## Released weight and performance
### CCPD
CiT-res18, res50, noSTN, 2STN is released, in which 2STN performed best.
| Method | overAll | base | blur | CHLG  | db   | fn   | rotate | tilt  | WX    | green |
|---------------------|---------|------|------|-------|------|------|--------|-------|-------|-------|
| CiT-res18 (3.1M,12GFLOPs) | 4.6     | 0.5  | 7.0  | 13.1  | 7.9  | 8.6  | 3.1    | 10.3  | 2.5   | 10.6  |
| CiT-res50 (8.9M,28.2G)   | 4.2     | 0.6  | 6.9  | 12.1  | 8.8  | 7.1  | **1.3**| 9.1   | 1.9   | 10.6  |
| CiT-noSTN (3.1M,12G)     | 3.7     | 0.3  | 5.8  | 10.5  | 6.6  | 7.5  | 2.1    | 7.9   | 1.9   | 9.2   |
| CiT-2STN (3.2M,9.2G)     | 3.2     | **0.3**| **4.3**| **9.6**| 6.4  | 6.1  | 1.7    | 6.6   | 1.5   | **8.3** |

### CLPD
|CiT-res18 | 31.0 |
|-|-|
|CiT-res18 (0 shot) | 60.3|

### CTPFSD
|CiT-res50-noSTN+augment | 39.1 |
|-|-|
|CiT-res50-noSTN (0 Shot) | 74.0 |

## Citation
```

```