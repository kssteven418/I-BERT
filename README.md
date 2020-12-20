<img width="900" alt="Screen Shot 2020-12-19 at 9 51 50 PM" src="https://user-images.githubusercontent.com/50283958/102689854-d5604e80-4244-11eb-83cd-5d75e76c8d04.png">

# I-BERT: Integer-only BERT Quantization


## Installation & Requirements
You can find more detailed installation guides from the Fairseq repo: https://github.com/pytorch/fairseq

**1. Fairseq Installation**

Reference: [Fairseq](https://github.com/pytorch/fairseq)
* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* Currently, I-BERT only supports training on GPU

```bash
git clone https://github.com/kssteven418/I-BERT.git
cd I-BERT
pip install --editable ./
```

**2. Download pre-trained RoBERTa models**

Reference: [Fairseq RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)

Download pretrained RoBERTa models from the links and unzip them.
* RoBERTa-Base: [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
* RoBERTa-Large: [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
```bash
# In I-BERT (root) directory
mkdir models && cd models
wget {link}
tar -xvf roberta.{base|large}.tar.gz
```


**3. Download GLUE datasets**

Reference: [Fairseq Finetuning on GLUE](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md)

First, download the data from the [GLUE website](https://gluebenchmark.com/tasks). Make sure to download the dataset in I-BERT (root) directory.
```bash
# In I-BERT (root) directory
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

Then, preprocess the data. 

```bash
# In I-BERT (root) directory
./examples/roberta/preprocess_GLUE_tasks.sh glue_data {task_name}
```
`task_name` can be one of the following: `{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}` .
`ALL` will preprocess all the tasks.
If the command is run propely, preprocessed datasets will be stored in `I-BERT/{task_name}-bin`

Now, you have the models and the datasets ready, so you are ready to run I-BERT!


## Task-specific Model Finetuning

Before quantizing the model, you first have to finetune the pre-trained models to a specific downstream task. 
Although you can finetune the model from the original Fairseq repo, we provide `ibert-base` branch where you can train non-quantized models without having to install the original Fairseq. 
This branch is identical to the master branch of the original Fairseq repo, except for some loggings and run scripts that are irrelevant to the functionality.
If you already have finetuned models, you can skip this part.

Run the following commands to fetch and move to the `ibert-base` branch:
```bash
# In I-BERT (root) directory
git fetch
git checkout -t origin/ibert-base
```

Then, run the script:
```bash
# In I-BERT (root) directory
# CUDA_VISIBLE_DEVICES={device} python run.py --arch {roberta_base|roberta_large} --task {task_name}
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task MRPC
```
Checkpoints and validation logs will be stored at `./outputs` directory. You can change this output location by adding the option `--output-dir OUTPUT_DIR`. The exact output location will look something like: `./outputs/none/MRPC-base/wd0.1_ad0.1_d0.1_lr2e-5/1219-101427_ckpt/checkpoint_best.pt`.
By default, models are trained according to the task-specific hyperparameters specified in [Fairseq Finetuning on GLUE](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md). However, you can also specify the hyperparameters with the options (use the option `-h` for more details). 


## Quantiation & Quantization-Aware-Finetuning

Now, we come back to `ibert` branch for quantization. 
```bash
git checkout ibert
```

And then run the script. This will first quantize the model and do quantization-aware-finetuning with the learning rate that you specify with the option `--lr {lr}`.
```bash
# In I-BERT (root) directory
# CUDA_VISIBLE_DEVICES={device} python run.py --arch {roberta_base|roberta_large} --task {task_name} \
# --restore-file {ckpt_path} --lr {lr}
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task MRPC --restore-file ckpt-best.pt --lr 1e-6
```

**NOTE:** Our work is still on progress. Currently, all integer operations are executed with floating point.
