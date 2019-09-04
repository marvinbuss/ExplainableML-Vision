# ExplainableAI-Vision

This repository introduces different Explainable AI approaches and demonstrates how they can be implemented with PyTorch and torchvision. Approaches used in this Jupyter notebook are:
1. [Class Activation Mappings (CAMs)](https://github.com/metalbubble/CAM)
2. [LIME](https://github.com/marcotcr/lime)
3. [SHapley Additive exPlanations (SHAP)](https://github.com/slundberg/shap) - **WIP**

## Dataset

The notebook uses the PyTorch hymenoptera dataset, which can be downloaded [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip). The dataset consists of 397 images of bees and ants which were split into a train, test and validation set as follows:

| Class | Train | Test  | Validation | Total |
| ----- |:-----:| -----:| ----------:| -----:|
| Bees  | 130   | 41    | 33         | 204   |
| Ants  | 124   | 38    | 31         | 193   |

## Models for transfer learning

The notebook allows to perform transfer learning using various pretrained models. The list includes:

- Densenet-161
- ResNet-152
- ResNet-101
- Inception v3
- ResNeXt-50-32x4d
- ResNeXt-101-32x8d

The user is also able to adjust the input size of the model in the parameters section (see parameters section below).

## Parameters

All relevant settings can be adjusted in the third cell of the notebook. These parameters adjust the following:

- `DATA_DIR`: Directory of the image dataset. The image dataset is expected to have the same folder structure as the hymenoptera dataset in this repository.
- `MODEL_NAME`: Name of the pretrained model that should be used in the notebook. Possible values are densenet161, resnet152, resnet101, inception, resnext50, resnext101
- `NUM_CLASSES`: Number of classes for the image classification task. For the hymenoptera dataset the number of classes is 2.
- `FIXED_FEATURE_EXTRACTOR`: Set to true, if you want to fix the model weights and only retrain the final output layer.
- `INPUT_SIZE`: Input size of the images.
- `BATCH_SIZE`: Size of each batch during training. How many samples per batch to load.
- `SHUFFLE`: Set to true to have the data reshuffled at every epoch.
- `NUM_WORKERS`: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
- `LEARNING_RATE`: Initial learning rate that should be used for training.
- `USE_ADAM_OPTIM`: Set to true, if you want to use the Adam optimizer. Set to false, if you want to use the SGD optimizer. 
- `MOMENTUM`: Hyperparameter of the SGD optimizer.
- `BETA_1`, `BETA_2`, `EPSILON`, `WEIGHT_DECAY`: Hyperparameters of the Adam optimizer.
- `NUM_EPOCHS`: Number of epochs in the training process.
- `DECAY_STEP_SIZE`, `GAMMA`: Hyperparameters of the [StepLR learning rate scheduler](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR).
- `BASE_LR`, `MAX_LR`, `STEP_SIZE_UP`, `STEP_SIZE_DOWN`, `MODE_CYCLIC`: Hyperparameters of the [cyclic learning rate scheduler](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CyclicLR).
- `MODE_PLATEAU`, `FACTOR`, `PATIENCE`, `COOLDOWN`, `MIN_LR`: Hyperparameters of the [ReduceLROnPlateau learning rate scheduler](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau).
- `RANDOM`: Number for random seed.

## Results

### CAMs



### LIME



### SHAP

**WIP**

## TODO

- Improve SHAP results (not working yet with provided models)
- Adjust data augmentation
- Make use of BCELoss
- Adjust and improve learning rate
- Handle different class sizes
