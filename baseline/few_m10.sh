for i in `seq 0 9`
do
TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=2 \
    python3 ../scripts/train.py \
    @../params/few_all.txt \
    --dset_dir ~/data/s2cnn2/few_m10/$i \
    --logdir ./few_log/$i \
    --run_id few_$i
done
