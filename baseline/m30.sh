TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=3 \
    python3 ../scripts/train.py \
    @../params/m30_all.txt \
    --dset_dir ~/data/s2cnn2/m30 \
    --logdir ./30_log \
    --run_id 30
