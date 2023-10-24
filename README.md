# Instruction for training DeeperSense models on HPC

### Please run everything in a terminal multiplexer session such as screen or tmux, so that access to same seesion is available even after ssh session ends.

1. login to `hpc3-login.dfki.uni-bremen.de`
2. goto `SCRATCH` dir
   ```
   cd SCRATCH
   ```
3. clone `git@git.hb.dfki.de:deepersense/deepersense-core/sonavision.git`. For this you may have to add your ssh key to hpc and start the ssh-agent and add key to the running instance of `ssh-agent` at. If this doesn't work, easiest option is clone using HTTPS protocol.
   ```
   git clone git@git.hb.dfki.de:deepersense/deepersense-core/sonavision.git
   cd sonavision/sonavision
   ```
4. Copy datasets from `~/scratch/uc1/hpc3_bkp/dataset` at `dprsnssrv.hb.dfki.de` to `sonavision/sonavision`. Need to start ssh-agent and add keys as mentioned above.
   ```
   rsync -anv <username>:~/scratch/uc1/hpc3_bkp/dataset .
   ```
   *once you are sure change `anv` to `azP`*
5. Start a docker `nsshah/deepersense:nvcr21.12-tf2-py3` container on HPC
   ```
   srun --account=deepl --nodelist=hpc-dnode4 --job-name=nish02nightvision3 --time=30-00:00:00 --pty --mem-per-cpu=16g --cpus-per-task=16 --gres=gpu:01 --partition=gpu_ampere shifter --image=nsshah/deepersense:nvcr21.12-tf2-py3 bash
   ```

6. run `train.py`  
   ```
   python train.py --name <expt-name> --dataset_dir <path/to/dataset> --image_format <jpg|png> --input_shape 512 1024 --output_shape 512 1024 --batch_size 1 --train_epoch 100 --arch_type with-camera-late-fusion
   ```

## `train.py` 

### usage
```
train.py [-h] --name NAME --dataset_dir DATASET_DIR [--train_subdir TRAIN_SUBDIR] [--val_subdir VAL_SUBDIR] [--test_subdir TEST_SUBDIR] [--batch_size BATCH_SIZE]
[--test_batch_size TEST_BATCH_SIZE] [--input_shape INPUT_SHAPE INPUT_SHAPE] [--output_shape OUTPUT_SHAPE OUTPUT_SHAPE] [--crop_shape CROP_SHAPE CROP_SHAPE] [--resize RESIZE RESIZE][--fliplr] [--train_epoch TRAIN_EPOCH] [--lrD LRD] [--lrG LRG] [--ngf NGF] [--ndf NDF] [--lambda_l1 LAMBDA_L1] [--beta1 BETA1] [--beta2 BETA2] [--results_dir RESULTS_DIR][--logs_dir LOGS_DIR] [--checkpoints_dir CHECKPOINTS_DIR] [--model_save_dir MODEL_SAVE_DIR] [--image_format IMAGE_FORMAT] [--num_images_per_image NUM_IMAGES_PER_IMAGE] --arch_type {with-camera-early-fusion,with-camera-late-fusion,with-out-camera}
```
### Arguments Description
```
optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the experiment. It decides where to store samples and models
  --dataset_dir DATASET_DIR
                        path to dataset
  --train_subdir TRAIN_SUBDIR
                        train subfolder in dataset
  --val_subdir VAL_SUBDIR
                        validation subfolder in dataset
  --test_subdir TEST_SUBDIR
                        test subfolder in dataset
  --batch_size BATCH_SIZE
                        train batch size
  --test_batch_size TEST_BATCH_SIZE
                        test batch size
  --input_shape INPUT_SHAPE INPUT_SHAPE
                        input shape, format: height width
  --output_shape OUTPUT_SHAPE OUTPUT_SHAPE
                        output shape, format: height width
  --crop_shape CROP_SHAPE CROP_SHAPE
                        crop shape (0 0 is false) format: height width
  --resize RESIZE RESIZE
                        resize scale (0 is false) format: height width
  --fliplr              random fliplr True or False
  --train_epoch TRAIN_EPOCH
                        number of train epochs
  --lrD LRD             learning rate, default=0.0002
  --lrG LRG             learning rate, default=0.0002
  --ngf NGF             base filters for generator
  --ndf NDF             base filters for discriminator
  --lambda_l1 LAMBDA_L1 
                        lambda for L1 loss
  --beta1 BETA1         beta1 for Adam optimizer
  --beta2 BETA2         beta2 for Adam optimizer
  --results_dir RESULTS_DIR
                        results save path
  --logs_dir LOGS_DIR   logs dir
  --checkpoints_dir CHECKPOINTS_DIR
                        checkpoints save path
  --model_save_dir MODEL_SAVE_DIR
                        models save path
  --image_format IMAGE_FORMAT
                        image format for input
  --num_images_per_image NUM_IMAGES_PER_IMAGE
                        number of images per sample
  --arch_type {with-camera-early-fusion,with-camera-late-fusion,with-out-camera}
                        architecture type

```