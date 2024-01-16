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

### Data Format
The dataset is structured in a specific format, and you can find sample data in the sample_data folder for reference. Ensure your model or processing pipeline is compatible with this format.

## Training

Run:

```bash
python train2.py \
--do_train \
--do_eval \
--output_dir ./output_result \
--bert_model "vinai/phobert-base-v2" \
--lamb 0.62 \
--temp 0.179 \
--temp_lamb 0.7 \
--negative_rate 16 \
--learning_rate 3e-5 \
--data_dir vlsp \
--num_train_epochs 50 \
--train_batch_size 128 \
--path_image vlsp/ner_image \
--task_name sonba \
--resnet_root "modules/resnet" \
--cache_dir "cache" \
--max_seq_length 256
```

