
# 1st place solution for *[Visual Product Recognition Challenge 2023](https://www.aicrowd.com/challenges/visual-product-recognition-challenge-2023)*


## Setting up the environment

Change /docker_env/start.sh for your workspace
Run container 
```bash
./start.sh
```
and configure accelerate lib for train used mixed_precision and multi-gpu
```bash
accelerate config
```


## Download dataset

Download and unzip raw dataset from https://storage.yandexcloud.net/mcs2023/wb_products.zip


## Clean dataset

### Train model on product10k

```bash
accelerate launch main.py --cfg ./config/pre_product10k.yml
```
### Convert best model to torchscript
```bash
python convert_to_torchscript.py --checkpoint-path <best_checkpoint_path> --output-path model_for_cleaned.pth
```

### Generate embeddings for images
```bash
cd dataset_prepare/embedding_generation

python main.py --model-path model_for_cleaned.pth --dataset-path <path_to_unpacking_dataset>
```

### Generate annotations file
```bash
cd dataset_prepare/annotation_generation

python main.py --annotation_path all_annotations.csv --dataset-path <path_to_unpacking_dataset>
```

### Filter annotations file
```bash
cd dataset_prepare/annotation_generation

python filter_empty_query_or_gallery.py --csv-path all_annotations.csv --output-path clean_all_annotations1.csv

python delete_duplicate_in_one_class.py --csv-path clean_all_annotations1.csv --dataset-path <path_to_unpacking_dataset> --output-path clean_all_annotations2.csv

python delete_duplicate_classes.py --csv-path clean_all_annotations2.csv --dataset-path <path_to_unpacking_dataset> --output-path clean_all_annotations3.csv
```

### Generate bounding boxes for query images

```bash
cd dataset_prepare/find_bboxes

python main.py --model-path model_for_cleaned.pth --csv-path clean_all_annotations3.csv --dataset-path <path_to_unpacking_dataset>
```

## Train model
### Train model on our dataset

```bash
accelerate launch main.py --cfg ./config/wb.yml
```


### Fine tune model on Products10k

```bash
accelerate launch main.py --cfg ./config/ft_product10k.yml --checkpoint-path <best checkpoint>
```

### Averaging checkpoints (soup)

```bash
python model_averaging.py
```

## Inference
### Convert best soup to torchscript

```bash
python convert_to_torchscript.py --checkpoint-path <best_soup_path> --output-path best_model.pth
```

### Run local evalute

```bash
python local_evaluation.py
```