# nerf-lightning
### Unofficial lightning implementation version of nerf.

<p>
<img src="https://github.com/gentleman-zhong/nerf-lightning/assets/74064666/5721d644-464f-4c95-b5a6-8e6afd80668d" width="400" alt="Looping GIF" loop>
<img src="https://github.com/gentleman-zhong/nerf-lightning/assets/74064666/199a8054-5d84-474f-9fc4-c578e44aa804" width="400" alt="Looping GIF" loop>
</p>

# DataSet
[Sample Dataset Download](https://drive.google.com/file/d/1lD0OFIjZaGRGdmLnvj5zL8eUIzaza-2K/view?usp=drive_link)


# How to use
## Using the Conda environment
- `conda create -n nerf_lightning python=3.8` 
- `conda activate nerf_lightning`
- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Configuring packages
- `pip install -r requirements.txt` 

# Train 
Adjust the parameters in the opt.py file.Enter the pre-configured nerf_lightning virtual environment.Utilize the following command for training.
```bash
python train.py
```

# Test

Adjust the parameters in the eval.py file to load the corresponding model. You can download our pretrained models [psnr=29.52.ckpt](https://drive.google.com/file/d/1mCOnqpUOal8SL2wWVfrcT6Pow0_NA2AW/view?usp=drive_link). Use the following command for testing.

```bash
python eval.py
```
