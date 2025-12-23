path=/mnt/wr/my_project_nas/NAS-BNN/work_dirs/nasbnn_exp/search/

CUDA_VISIBLE_DEVICES=0 python print_arch.py 2>&1 | tee ${path}print_arch.log