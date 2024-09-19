## 
**
The code,weights and data sets are all provide.

Official PyTorch implementation for AgCLIP. Details can be found in the paper.


### Preparation
#### 1. Download Dataset

In our project, the FSC-147 dataset is used.
Please visit following link to download this dataset.

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

We also use AC-46 provided in this link.

* [AC-46](https://)


#### 2. Set Up Anaconda Environment:

```
conda create --name countx-environ python=3.7
conda activate countx-environ
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
git clone git@github.com:niki-amini-naieni/CounTX.git
cd CounTX/open_clip
pip install .
```
* This repository uses [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. This fix can be implemented by replacing the file timm/models/layers/helpers.py in the timm codebase with the file [helpers.py](https://github.com/niki-amini-naieni/CounTX/blob/main/helpers.py) provided in this repository.

### AgCLIP Train

To train the model, run the following command after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). 

```
--output_dir
"./results"
--img_dir
"/home/test/countx/CounTX-main-arg-up/FSC147_384_V2/images_384_VarV2"
--gt_dir
"/home/test/countx/CounTX-main-arg-up/FSC147_384_V2/gt_density_map_adaptive_384_VarV2"
--class_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/ImageClasses_Arg.txt"
--FSC147_anno_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/annotation_FSC147_384.json"
--FSC147_D_anno_file
"/home/test/countx/CounTX-main-arg-up/Arg.json"
--data_split_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/Train_Test_Val_Arg.json"
```

### CounTX Inference
To test a model, run the following commands after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model you want to test. 

For the validation set:

```

```

For the test set:

```

```

### Pre-Trained Weights

For the validation set:

```

```

For the test set:

```
python test_reproduce_paper.py --data_split "test" --output_dir "./test" --resume "paper-model.pth" --img_dir "/scratch/local/hdd/nikian/images_384_VarV2" --FSC147_anno_file "/scratch/local/hdd/nikian/annotation_FSC147_384.json" --FSC147_D_anno_file "./FSC-147-D.json" --data_split_file "/scratch/local/hdd/nikian/Train_Test_Val_FSC_147.json"
```

* The reason that the code for testing the model from the paper is different from the main testing code is that since releasing the paper, the CounTX class definition has been refactored for readability. 



### Citation



### Acknowledgements

This repository is based on the [CounTR repository](https://github.com/Verg-Avesta/CounTR) and uses code from the [OpenCLIP repository](https://github.com/mlfoundations/open_clip). 



