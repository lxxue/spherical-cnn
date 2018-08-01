for i in `seq 0 9`
do
    TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=1 \
                   python3 ../scripts/train.py \
                           @../params/model-64.txt \
                           @../params/m40-64.txt \
                           @../params/training.txt \
                           --dset_dir ~/data/few_m10_$i \
                           --logdir ./few_m10_log/$i \
                           --run_id few_m10_$i
done