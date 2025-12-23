CUDA_VISIBLE_DEVICES=0 python search.py  \
    --world-size 1 --rank 0 \
    --latency-gpu cpu \
    --gpu 0 \
    --search-without-acc \
    --max-epochs 1 --population-num 32 --m-prob 0.2 --crossover-num 32 --mutation-num 32 \
    --ops-min 80 --ops-max 90 --step 1 --max-train-iters 10 --train-batch-size 16 --test-batch-size 16 \
    --dataset imagenet -a superbnn ./work_dirs/nasbnn_exp/checkpoint.pth.tar \
    data/ImageNet_split ./work_dirs/nasbnn_exp/search