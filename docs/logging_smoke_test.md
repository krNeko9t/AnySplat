# Logger smoke test

After changing the logging backend (wandb / tensorboard / local), run these checks.

## Single-GPU

- **TensorBoard**: `logger=tensorboard python src/main.py -m experiment=instseg_custom trainer.max_steps=100`  
  Then `tensorboard --logdir output/exp_instseg_custom/<run_dir>` and confirm scalars and images.
- **WandB**: `logger=wandb python src/main.py -m experiment=instseg_custom trainer.max_steps=100`  
  Confirm run appears in WandB and val images/videos log.
- **Local**: `logger=local python src/main.py -m experiment=instseg_custom trainer.max_steps=100`  
  Confirm `outputs/local/` gets comparison images and videos.

## Multi-GPU (e.g. 8 cards)

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py -m experiment=instseg_inscene_infinigen_mv`  
  With `instseg_inscene_infinigen_mv` already setting `logger: tensorboard`, confirm a single TensorBoard log dir under the run output and no NCCL timeouts.
