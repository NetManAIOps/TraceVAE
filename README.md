# TraceVAE
This is the source code for "Unsupervised Anomaly Detection on Microservice Traces through Graph VAE".

## Usage
1. `pip3 install -r requirements.txt`.
2. Convert the dataset with `python3 -m tracegnn.cli.data_process preprocess -i [input_path] -o [dataset_path]`. The sample dataset is under `sample_dataset`. (Note: This sample dataset only shows data format and usage, and cannot be used to evaluate model performance. Please replace it with your dataset.)
sample:
```
python3 -m tracegnn.cli.data_process preprocess -i sample_dataset -o sample_dataset
```
3. Train the model with `bash train.sh [dataset_path]`:
```
bash train.sh sample_dataset
```
4. Evaluate the model with `bash teset.sh [model_path] [dataset_path]`. The default model path is under `results/train/models/final.pt`:
```
bash test.sh results/train/models/final.pt sample_dataset
```
