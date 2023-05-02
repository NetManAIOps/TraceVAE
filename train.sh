echo "Usage: bash train.sh [dataset_path]"
echo "DATASET: $1"
rm -r results
python3 -m tracegnn.models.trace_vae.train --device=cpu --dataset.root_dir="$1" --seed=1234 --model.struct.z_dim=10 --model.struct.decoder.use_prior_flow=true --train.z_unit_ball_reg=1 --model.latency.z2_dim=10 --model.latency.decoder.condition_on_z=true