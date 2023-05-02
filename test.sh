echo "Usage: bash test.sh [model_path] [dataset_path]"
echo "MODEL: $1"
echo "DATASET: $2"
python3 -m tracegnn.models.trace_vae.test evaluate-nll -M "$1" --use-train-val -D "$2" --device cpu --use-std-limit --std-limit-global
