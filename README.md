# Fastorch
A pytorch project for fast runing deep learning and iterating version.

# Files

```text
run.py # main function for runing
trainer.py # trainer, train and test model
config.yaml # global config file, loaded in run.py
dataset.py # define personal dataset, used in run.py with dataloader
loss.py # define personal loss, used in train.py
net.py # define personal network, used in run.py
utils.py # define other functions
```

# Usage

```bas
# define your dataset, loss, net, trainer and config file
ssh run.sh
```

# Change Log
### 2022-11-09
+ add `load_in_memory` for dataset for speed training
+ remove some above functions (Doesn't work well in the actual version iteration):
  + save version project file for each version
    + type of saved files is defined by `save_version_file_patterns` in config file
    + if `load_epoch` in config file is set false, save files in `runs/latest_project` and `runs/version/project`
    + ~~if `load_epoch` in config file is set epoch name in saved model file~~
      + ~~save latest files in `runs/latest_project`.~~
      + ~~load files from `runs/version/project` for testing~~
      + ~~restore latest files from `runs/latest_project`.~~ 

### 2022-07-05
+ save version project file for each version
  + type of saved files is defined by `save_version_file_patterns` in config file
  + if `load_epoch` in config file is set false, save files in `runs/latest_project` and `runs/version/project`
  + if `load_epoch` in config file is set epoch name in saved model file
    + save latest files in `runs/latest_project`
    + load files from `runs/version/project` for testing
    + restore latest files from `runs/latest_project`
+ save tensorboard file, running log, config file, model state dict for each version
  + use `tensorboard --logdir=runs` for visualization
  + load model parameters from saved model files
  + read runing log for training output
+ easily change train parameters
  + `random_seed`: set random seed for each training
  + `epochs`: set total training epochs
  + `batch_size`: set batch size
  + `num_workers`: set number of processes
  + `lr`: set learning rate
  + `device_ids`: gpu device ids, use cpu if none; if its number greater than 1, use data parallel
  + `valid_every_epochs`: epochs of validing model
  + `early_stop_epochs`: epochs of early stop, set negative number for not using
  + `start_save_model_epochs`: greater than it will save model
  + `save_model_interval_epochs`: interval epochs for saving model
+ add parameters in config file to args, so it can use `args.xxx` to use these parameters easily
