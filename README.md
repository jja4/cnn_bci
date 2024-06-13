## README.md

### Project Overview

This project is a Python-based implementation of a Brain-Computer Interface (BCI) system using deep learning techniques. The code is structured into two main folders: `code` and `results`.

#### Code Folder

The `code` folder contains multiple scripts that implement various deep learning models for BCI classification tasks. These scripts are similar but may have slight variations in hyperparameters and network architectures. The main script provided in the code snippet is an example of one such script.

#### Results Folder

The `results` folder is used to store the accuracy and loss values obtained during the training and evaluation of the deep learning models. These results are saved in a format that can be easily loaded and analyzed later.

### Code Overview

The provided code snippet is a Python script that performs the following tasks:

1. **Data Loading**: The script loads EEG data and corresponding labels from MATLAB files located in a specified directory (`default_path`).
2. **Preprocessing**: The loaded EEG data is preprocessed using various techniques, including lowpass filtering, highpass filtering, and exponential running standardization.
3. **Data Conversion**: The preprocessed data is converted to the format expected by the Braindecode library, which is used for training and evaluating the deep learning models.
4. **Model Training and Evaluation**: The script performs a cross-validation loop over multiple subjects and folds. For each subject and fold, the data is split into training and test sets, and a deep learning model (either ShallowFBCSPNet or Deep4Net) is trained and evaluated on the respective sets.
5. **Result Saving**: The accuracy and loss values obtained during training and evaluation are saved in numpy arrays (`Accuracies` and `Losses`), which are then stored in a PyTorch tensor file (`var_save_path`) for later use.

### Dependencies

The code relies on the following Python libraries:

- `scipy`
- `braindecode`
- `numpy`
- `torch`
- `os`
- `glob`

Make sure to install these libraries before running the code.

### Usage

1. Ensure that the required data files are present in the specified `default_path` directory.
2. Modify the `default_path`, `save_path`, and `var_save_path` variables to match the desired locations on your system.
3. Run the script, and it will perform the training and evaluation process for the specified subjects and folds.
4. The accuracy and loss values will be saved in the specified `var_save_path` file.

### Notes

- The code is designed to run on a GPU, as indicated by the `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` line. Modify this line if you want to run the code on a different GPU or on the CPU.
- The hyperparameters, such as the number of epochs (`num_epochs`) and the kernel sizes for the convolutional layers (`model.network.conv_time.kernel_size` and `model.network.conv_spat.kernel_size`), can be adjusted as needed.
- The code assumes the presence of a file `LR_subjs.tar` in the specified `subjs_path` directory, which contains a list of subjects with only left and right imagined movements.
