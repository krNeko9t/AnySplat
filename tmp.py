from lpips import LPIPS
LPIPS(net="vgg")

python iggt_idmap.py \
    --image_dir 3dovs/bench/images \
    --model_path /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/iggt_checkpoint.pth \
    --output_dir output_idmap \
    --image_size 504 336 \
    --batch_size 4 \
    --knn_k 20 \
    --eps 0.06 \
    --min_samples 100 \
    --min_cluster_size 500

python iggt_idmap.py \
    --image_dir 3dovs/bench/images \
    --model_path /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/iggt_checkpoint.pth \
    --output_dir output_idmap_bench_sw0_kk60 \
    --image_size 504,336 \
    --batch_size 48 \
    --eps 0.06 \
    --hdbscan_min_samples 100 \
    --hdbscan_min_cluster_size 500 \
    --knn_k 60 \
    --spatial_weight 0.0 \
    --max_cluster_points 600000

python iggt_idmap.py \
    --image_dir infinigen/16255241/frames/Image/camera_0 \
    --model_path /mnt/shared-storage-gpfs2/solution-gpfs02/liaoyuanjun/iggt_checkpoint.pth \
    --output_dir output_idmap2 \
    --image_size 504,336 \
    --batch_size 48 \
    --eps 0.06 \
    --hdbscan_min_samples 100 \
    --hdbscan_min_cluster_size 500