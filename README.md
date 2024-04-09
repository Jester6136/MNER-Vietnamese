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
theta=0.07
sigma=0.007
lr_pixelcnn=0.0008
weight_decay_pixelcnn=0.000001
learning_rate=3e-5
num_train_epochs=10
train_batch_size=64
path_image="vlsp_2016/ner_image"
bert_model="vinai/phobert-base-v2"
data_dir="vlsp_2016"
resnet_root="modules/resnet"
cache_dir="cache"
max_seq_length=256

python train_umt_pixelcnn_fixedlr.py \
    --do_train \
    --do_eval \
    --output_dir ./output_result_${data_dir}_epoch${num_train_epochs}_${train_batch_size}_alpha${alpha}_beta${beta}_theta${theta}_sigma${sigma}_weight_decay_pixelcnn${weight_decay_pixelcnn}_lr_pixelcnn${lr_pixelcnn}_lr${learning_rate} \
    --bert_model "${bert_model}" \
    --alpha ${alpha} \
    --beta ${beta} \
    --sigma ${sigma} \
    --theta ${theta} \
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

