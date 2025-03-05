# MNER Vietnamese

## Guide
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Training](#training)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.6+. To run our code, install:

```
pip install -r requirements.txt
```


## Datasets
vlsp [here](https://vlsp.org.vn/vlsp2021/eval/ner)

### Data Format
The dataset is structured in a specific format, and you can find sample data in the sample_data folder for reference. Ensure your model or processing pipeline is compatible with this format.

## Training

Run:

```bash
task_name="sonba"
alpha=0.5
beta=0.5
theta=0.05
sigma=0.005
lr_pixelcnn=0.001
weight_decay_pixelcnn=0.00005
learning_rate=3e-5
num_train_epochs=10
train_batch_size=32
path_image="/home/rad/bags/vlsp_all/origin+image/VLSP2016/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="/home/rad/bags/vlsp_all/origin+image/VLSP2016"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train_umt_pixelcnn_fixedlr.py \
    --do_train \
    --do_eval \
    --output_dir $_beta${beta}_theta${theta}_sigma${sigma}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
    --warmup_proportion 0.4 \
    --gradient_accumulation_steps 8 \
    --weight_decay_pixelcnn ${weight_decay_pixelcnn} \
    --lr_pixelcnn ${lr_pixelcnn} \
    --learning_rate ${learning_rate} \
    --data_dir "${data_dir}" \
    --num_train_epochs ${num_train_epochs} \
    --train_batch_size ${train_batch_size} \
    --path_image "${path_image}" \
    --task_name "${task_name}" \
    --resnet_root "${resnet_root}" \
    --cache_dir "${cache_dir}" \
    --max_seq_length ${max_seq_length}
```

