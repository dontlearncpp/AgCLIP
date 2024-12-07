## 
**
The code,weights and datasets are all provide.

Official PyTorch implementation for AgCLIP. Details can be found in the paper.
**
The frame of AgCLIP is
![image](https://github.com/user-attachments/assets/beacc0cf-09ed-4009-8942-fff5d25d7777)




### Preparation
#### 1. Download Dataset

In our project, the FSC-147 dataset is used.
Please visit following link to download this dataset.

* [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

We also use AC-46 provided in this link.
AC-46 was collected from large public datasets COCO, FSC-147, and 22 types open-source agriculture target detection datasets.

* [AC-46](https://drive.google.com/file/d/1yp3yoD_GRCMTF1lQFjNARkvnMjazY4kH/view?usp=drive_link)
![image](https://github.com/user-attachments/assets/f9e3b6f2-638f-4243-97bf-6b57d841c9cb)


#### 2. Set Up Anaconda Environment:

```
conda create --name Agclip python=3.7
conda activate Agclip
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install scipy
pip install imgaug
git clone git@github.com:niki-amini-naieni/CounTX.git
cd CounTX/open_clip
pip install .
```
* This repository uses [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. This fix can be implemented by replacing the file timm/models/layers/helpers.py in the timm codebase with the file [helpers.py](https://github.com/niki-amini-naieni/CounTX/blob/main/helpers.py) provided in this repository.

##### Open set prompt learning (OPL):

The related code (/home/test/.conda/envs/Agclip/lib/python3.8/site-packages/open_clip/model.py) Line 216 in OpenCLIP was modeified:
```
    def encode_text(self, text,prompts, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)+prompts

        # x = self.positional_embedding.to(cast_dtype)+prompts

        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x=x+prompts
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
```


OPL was refered to the ['cocoop'](https://github.com/KaiyangZhou/CoOp), and [OpenCLIP repository](https://github.com/mlfoundations/open_clip). 

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
"/home/test/countx/CounTX-main-arg-up/Arg46.json"
--data_split_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/Train_Test_Val_Arg46.json"
```

### AgCLIP Inference
To test a model, run the following commands after activating the Anaconda environment set up in step 2 of [Preparation](#preparation). Make sure to change the directory and file names to the ones you set up in step 1 of [Preparation](#preparation). Make sure that the model file name refers to the model you want to test. 

For the validation set:

```
--data_split
"val"
--output_dir
"./test"
--resume
"results/checkpoint-200ok.pth"
--img_dir
"/home/test/countx/CounTX-main-arg-up/FSC147_384_V2/images_384_VarV2"
--FSC147_anno_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/annotation_FSC147_384.json"
--FSC147_D_anno_file
"/home/test/countx/CounTX-main-arg-up/Arg46.json"
--data_split_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/Train_Test_Val_Arg46.json"
```

For the test set:

```
--data_split
"test"
--output_dir
"./test"
--resume
"results/checkpoint-200ok.pth"
--img_dir
"/home/test/countx/CounTX-main-arg-up/FSC147_384_V2/images_384_VarV2"
--FSC147_anno_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/annotation_FSC147_384.json"
--FSC147_D_anno_file
"/home/test/countx/CounTX-main-arg-up/Arg46.json"
--data_split_file
"/home/test/countx/CounTX-main-arg-up/LearningToCountEverything-master/data/Train_Test_Val_Arg46.json"
```

### Pre-Trained Weights
The weight of replicating the accuracy of the paper：
* [The weight of paper](https://drive.google.com/file/d/1bLYgfxeOvpHow99W3EvlzBVn80LkVuR5/view?usp=drive_link)





### Citation

The paper is still under reviewing.

### Acknowledgements

This repository is based on the [CounTX repository](https://github.com/niki-amini-naieni/CounTX) and uses code from the [OpenCLIP repository](https://github.com/mlfoundations/open_clip). 



