# Multi-Feature Interaction and Degradation Estimation Transformer for Spectral Compressive Imaging

<div class="center-text">
     Jiaojiao Li ,&emsp;
     Ding Zhu ,&emsp;
     Rui Song ,&emsp;
     Haitao Xu ,&emsp;
     Yunsong Li ,&emsp;
     Qian Du &emsp;
</div>

This repo is the implementation of paper "Multi-Feature Interaction and Degradation Estimation Transformer for Spectral Compressive Imaging"

<i><strong><a target='_blank'>TCSVT 2025</a></strong></i>

[PDF](https://ieeexplore.ieee.org/document/10892248)

## Abstract

Coded Aperture Snapshot Spectral Imaging (CASSI) systems provide an efffcient approach to acquiring Hyperspectral Images (HSI), yet the reconstruction process still presents challenges. Traditional Deep Unfolding Networks (DUN) applied to CASSI often face constraints due to inadequate feature utilization and poor handling of multi-scale frequency domain information, leading to the loss of image detail and global information. Furthermore, most DUN methodologies oversimplify degrading factors and fail to account for issues such as distortions found in actual imaging, thus affecting accuracy and robustness. This paper presents MIDET, a novel DUN tailored for CASSI systems, which integrates the fusion of band information, spatial information, and multiscale information to meaningfully improve feature utilization and information interaction efffciency. Additionally, MIDET introduces a degradation-guided learning strategy and a frequency feature extraction module, enhancing the capability to handle real imaging distortions and preserve more details in HSI reconstruction. Experimental results demonstrate that MIDET signiffcantly outperforms existing technologies on both simulated and real datasets, effectively enhancing the quality of HSI reconstruction.

## Comparison with state-of-the-art methods

<div align=center>
<img src="https://github.com/xgiaogiao/MIDET/figures/fig0.png" width = "350" height = "300" alt="">
</div>
Comparison between reconstruction methods in terms of PSNR-FLOPS-Params. The vertical axis represents PSNR (dB), the horizontal axis represents FLOPS (G), and the radius of the circle corresponds to Params. The analysis results indicate that MIDET proposed in this study achieves optimal performance with lower values of parameter quantity and FLOPS.

### Model Zoo and Results

Download Model Zoo and Results ([Baidu Disk](https://pan.baidu.com/s/1aKLnHvAfqJbeykdO74Gixg?pwd=mdet), code: `mdet`) then put them into the corresponding folder "MIDET/checkpoints/" folder as the following form:

	|--checkpoints
	    |--simulation
	    |--real

## Architecture

### Multi-feature Aggregated and Degradation Estimation Unfolding Framework

<div align=center>
<img src="https://github.com/xgiaogiao/MIDET/figures/fig1.png" width = "700" height = "400" alt="">
</div>

The structure of Multi-feature Aggregated and Degradation Estimation Deep Unfolding Transformer(MIDET) with K Stage(iterations) for HSI reconstruction. (a) Nonlinear Normalized Subgradient Descent (NNSGD) Module (b) Proximal Mapping(PM) Module

### Sparse Dense Attention and Spectral Attention Block 

<div align=center>
<img src="https:https://github.com/xgiaogiao/MIDET/figures/fig2.png" width = "600" height = "400" alt="">
</div>

The structure of attention block. (a)Sparse Dense Self-Attention. It includes a sparse attention strategy (each group of labels comes from sparse regions in the image) and a dense attention strategy (each group of labels comes from dense regions in the image). (b)Spectral Wise Self-Attention. (c)Degradation Estimation Graph Interaction. (d)Degradation Estimation Vector Interaction. (e)Residual Frequency Fusion Module

## Usage

### 1. Create Environment

pip install -r requirements.txt

### 2. Data Preparation

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--MIDET
    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

### 3. Simulation Experiment
#### 3.1　Training
```
cd MIDET/simulation/train_code/ 
python train.py  --outf ./exp/MIDET_3stg/ --method midet_3

python train.py  --outf ./exp/MIDET_5stg/ --method midet_5

python train.py  --outf ./exp/MIDET_7stg/ --method midet_7

python train.py  --outf ./exp/MIDET_9stg/ --method midet_9
```
#### 3.2　Testing
```shell
cd MIDET/simulation/test_code/
python test.py  --outf ./exp/MIDET_3stg/ --method midet_3 --pretrained_model_path ./checkpoints/midet_3stg.pth

python test.py  --outf ./exp/MIDET_5stg/ --method midet_5 --pretrained_model_path ./checkpoints/midet_5stg.pth

python test.py  --outf ./exp/MIDET_7stg/ --method midet_7 --pretrained_model_path ./checkpoints/midet_7stg.pth

python test.py  --outf ./exp/MIDET_9stg/ --method midet_9 --pretrained_model_path ./checkpoints/midet_9stg.pth
```
The reconstrcuted HSIs will be output into `MIDET/simulation/test_code/exp/`. Then place the reconstructed results into `MIDET/simulation/test_code/Quality_Metrics/results` and run the following MATLAB command to calculate the PSNR and SSIM of the reconstructed HSIs.

    Run cal_quality_assessment.m

#### 3.3	Visualization

- Put the reconstruted HSI in `MIDET/visualization/` 

- Generate the RGB images of the reconstructed HSIs

```shell
 cd MIDET/visualization/
 Run show_simulation.m 
```

- Draw the spetral density lines

```shell
cd MIDET/visualization/
Run show_line.m
```

### 4. Real Experiment

#### 4.1　Training
```
cd MIDET/real/train_code/ 
python train.py   --outf ./exp/MIDET_2stg/ --method midet_2  
```
#### 4.2　Testing
```
cd MIDET/real/test_code/
python test.py   --outf ./exp/MIDET_2stg/ --method midet_2    --pretrained_model_path ./checkpoints/midet_2stg.pth
```
#### 4.3	Visualization

- Put the reconstruted HSI in `MIDET/visualization/`
- Generate the RGB images of the reconstructed HSI

```shell
cd MIDET/visualization/
Run show_real.m
```

###  5. Acknowledgements 

This code repository's implementation is based on  [MST](https://github.com/caiyuanhao1998/MST)、[PADUT](https://github.com/MyuLi/PADUT) and [RDLUF](https://github.com/ShawnDong98/RDLUF_MixS2) . We thank them for their generous open-source contributions.

## Citation
```shell
@article{MIDET,
  title={Multi-Feature Interaction and Degradation Estimation Transformer for Spectral Compressive Imaging},
  author={Li, Jiaojiao and Zhu, Ding and Song, Rui and Xu, Haitao and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}

```