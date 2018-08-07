for i in `seq 0 9`
do
    TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
                   python3 ../scripts/train.py \
                           @../params/model-64.txt \
                           @../params/m40-64.txt \
                           @../params/training.txt \
                           --dset_dir ~/data/few_m102/few_m10_$i \
                           --logdir ./few_m102_log/$i \
                           --run_id few_m102_$i
done
