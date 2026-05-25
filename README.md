# lowlight-enhance

本仓库是数字图像处理论文对应的低照度图像增强实验代码仓库，统一整理了 `Zero-DCE`、`SCI` 和 `Retinexformer` 三种方法的推理代码、预训练权重与数据准备工具。仓库直接支持 `HEIC`、`JPG`、`JPEG`、`DNG` 和 `PNG` 输入，并提供 `2560` 长边评测集生成脚本。

## 仓库结构

```text
lowlight-enhance/
├── repos/
│   ├── Zero-DCE/
│   ├── SCI/
│   └── Retinexformer/
├── datasets/
│   ├── nightdataset/
│   ├── nightdataset_resized_long2560/
│   ├── nightdataset_manifest.txt
│   └── README.md
├── tools/
│   ├── check_nightdataset_manifest.py
│   └── resize_long_edge.py
└── outputs/
```

## 运行环境

本仓库在以下环境下完成验证：

- `Python 3.10.20`
- `PyTorch 2.11.0+cu130`
- `torchvision 0.26.0+cu130`
- `Pillow 12.2.0`
- `pillow-heif 1.3.0`
- `rawpy 0.27.0`
- `OpenCV 4.13.0`
- `NumPy 2.2.6`

按下面命令创建环境：

```bash
conda create -n i2p python=3.10.20 -y
conda activate i2p

pip install torch==2.11.0 torchvision==0.26.0 torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install pillow==12.2.0 pillow-heif==1.3.0 rawpy==0.27.0 numpy==2.2.6 opencv-python scipy scikit-image tqdm natsort einops yacs lmdb addict future requests pyyaml joblib matplotlib tensorboard h5py

cd repos/Retinexformer
python setup.py develop --no_cuda_ext
cd ../..
```

## 数据准备

### 1. 获取原始测试集

原始测试图像不进入 Git 历史。复现实验时，请从https://pan.baidu.com/s/1dbESUHpGBVLUbcDIHGGyEg?pwd=phjr （提取码: phjr）下载 `nightdataset`数据集，将其中全部 `44` 张原图存储到：

```text
datasets/nightdataset/
```

文件名必须与 [datasets/nightdataset_manifest.txt](datasets/nightdataset_manifest.txt) 完全一致。

解压完成后，执行下面命令核对文件集合：

```bash
python tools/check_nightdataset_manifest.py \
  --input-dir datasets/nightdataset \
  --manifest datasets/nightdataset_manifest.txt
```

脚本输出 `Manifest check passed.` 后，再进入下一步。

### 2. 生成 2560 长边评测集

```bash
python tools/resize_long_edge.py \
  --input-dir datasets/nightdataset \
  --output-dir datasets/nightdataset_resized_long2560 \
  --long-edge 2560
```

该命令会把原始图像统一转换为 `PNG`，并将长边缩放到 `2560`。

## 预训练权重

本仓库已经包含实验所需预训练权重，无需额外下载：

- `repos/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth`
- `repos/SCI/model/demo.pt`
- `repos/Retinexformer/pretrain_model/LOL_v2_real.pth`
- `repos/Retinexformer/pretrain_model/LOL_v1.pth`

## 使用步骤

### Zero-DCE

原始分辨率测试：

```bash
python repos/Zero-DCE/Zero-DCE_code/batch_infer.py \
  --input-dir datasets/nightdataset \
  --output-dir outputs/zero_dce/nightdataset
```

`2560` 长边测试：

```bash
python repos/Zero-DCE/Zero-DCE_code/batch_infer.py \
  --input-dir datasets/nightdataset_resized_long2560 \
  --output-dir outputs/zero_dce/nightdataset_resized2560
```

### SCI

原始分辨率测试：

```bash
python repos/SCI/infer_nightdataset.py \
  --data_path datasets/nightdataset \
  --save_path outputs/sci/nightdataset
```

`2560` 长边测试：

```bash
python repos/SCI/infer_nightdataset.py \
  --data_path datasets/nightdataset_resized_long2560 \
  --save_path outputs/sci/nightdataset_resized2560
```

### Retinexformer

原始分辨率测试：

```bash
python repos/Retinexformer/Enhancement/test_nightdataset.py \
  --input_dir datasets/nightdataset \
  --output_dir outputs/retinexformer/nightdataset \
  --opt repos/Retinexformer/Options/RetinexFormer_LOL_v2_real.yml \
  --weights repos/Retinexformer/pretrain_model/LOL_v2_real.pth
```

`2560` 长边测试：

```bash
python repos/Retinexformer/Enhancement/test_nightdataset.py \
  --input_dir datasets/nightdataset_resized_long2560 \
  --output_dir outputs/retinexformer/nightdataset_resized2560 \
  --opt repos/Retinexformer/Options/RetinexFormer_LOL_v2_real.yml \
  --weights repos/Retinexformer/pretrain_model/LOL_v2_real.pth
```

## 输出目录

所有推理结果默认写入：

```text
outputs/
```

每种方法的输出目录互相独立，可以直接用于后续定量或主观对比实验。

## Citation

如果需要在其他研究中使用本仓库，请考虑引用本研究使用的三个低照度图像增强方法的原始论文：

### Zero-DCE

```bibtex
@inproceedings{Zero-DCE,
  author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
  title = {Zero-reference deep curve estimation for low-light image enhancement},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {1780--1789},
  month = {June},
  year = {2020}
}
```

### SCI

```bibtex
@inproceedings{ma2022toward,
  title = {Toward Fast, Flexible, and Robust Low-Light Image Enhancement},
  author = {Ma, Long and Ma, Tengyu and Liu, Risheng and Fan, Xin and Luo, Zhongxuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5637--5646},
  year = {2022}
}
```

### Retinexformer
```bibtex
@inproceedings{retinexformer,
  title = {Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author = {Cai, Yuanhao and Bian, Hao and Lin, Jing and Wang, Haoqian and Timofte, Radu and Zhang, Yulun},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2023}
}
```
