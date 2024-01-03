# Multimodal BioNER Vietnamese

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

## Training

Run:

```bash
python train2.py --do_train --do_eval --output_dir ./output_result --bert_model "vinai/phobert-base-v2" --data_dir vlsp --num_train_epochs 30 --train_batch_size 128 --path_image vlsp/ner_image --task_name sonba --resnet_root "modules/resnet" --cache_dir "cache" --max_seq_length 256
```

