# Position-aware Interleaved Spatial-Spectral Network



Official Pytorch implementation of 

## Architecture


## Usage:
### Recommended environment:
```
Python 3.8
Pytorch 1.10.0+cu111
torchvision 0.10.0+cu111
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
- **Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases. move into './data/ACDC/' folder.

- **ACDC dataset:**
Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and move into './data/ACDC/' folder.


### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV?usp=sharing), and then put it in the './pretrained_pth' folder for initialization. 

### Training:

For Synapse Multi-organ training run ```train_Synapse.py```

For ACDC training run ```train_ACDC.py```

### Testing:

For Synapse Multi-organ testing run ```test_Synapse.py```

For ACDC testing run ```test_ACDC.py```


### Trained model
Our trained model can be finded by https://pan.baidu.com/s/170UZ5jq8DbStdBG81Q71zQ?pwd=12qw

## Acknowledgement
We are very grateful for these excellent works 
## Citations

```
