TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
    python3 ../scripts/train.py \
    @../params/10_all.txt \
    --dset_dir ~/data/s2cnn2/m10 \
    --logdir ./10_log \
    --run_id 10
