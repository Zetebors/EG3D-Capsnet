[readme.txt](https://github.com/user-attachments/files/22057004/readme.txt)
The 新EG3D(1).ipynb is our model.
The .ipynb files are used by JupyterLab.
For multimodal data, please run multimodal.ipynb; for unimodal (MRI image) data, use non-multimodal.ipynb.

Place the model code into the empty cell labeled cell2.

Configuration parameter description:

CLASS_MAP: Category mapping. It is not recommended to change this.

TARGET_SHAPE: The final shape of the image after cropping and before input. Images smaller than this size will be automatically padded.

TARGET_SPACING: The resolution used for isotropic resampling.

AUG_FACTOR: The number of times data augmentation is applied during training.

start_fold: The fold at which to start training. Since training time is on the order of days, this parameter allows resuming from an interrupted point. Valid range: [0, fold-1].

results_file: The filename of the JSON file for saving results. Used to store the results of completed training folds.

CUSTOM_PREFIX: The prefix for saving .pth model files. The code will save models for the epoch where early stopping was triggered and the epoch with the best test performance in each fold. The .pth filenames are defined as:
best_model_path = f"{CUSTOM_PREFIX}_fold{fold}_best.pth"
final_model_path = f"{CUSTOM_PREFIX}_fold{fold}_last.pth"

kf: Stratified k-fold cross-validation.

The paths in all_cases are the data paths.

dataset.py is for multimodal and Fdataset.py is for non-multimodal.

The package installation command is as follows: pip install scipy torchio nibabel numpy torch scikit-learn tensorboard torchvision matplotlib scikit-image
