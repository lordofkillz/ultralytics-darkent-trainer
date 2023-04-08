# Python Darknet PyTorch Trainer

This is a training framework for 5 and 8 using ultralytics , and for yolo 12347 .weights using  Darknet framework.

![Traininer ](trainer.gif)

# Darknet-PyTorch Trainer

This Python trainer simplifies the process of training and fine-tuning Darknet models with various features and options, making it easier to achieve better results in less time. With a user-friendly interface, the trainer streamlines the workflow and reduces the manual work required to train models on custom datasets.

## Features

- **Efficient training with various flags**: Train Darknet models using different options to optimize the training process, such as enabling multi-GPU support and controlling batch sizes.

- **Automatic CFG file parsing**: Automatically updates max_batches and filters after calculating anchors. Just upload the anchors.txt file, and the trainer takes care of the rest.

- **Integration with Ultralytics**: Train models using Ultralytics' YAML files. Upload and modify the YAML files as needed to customize the training process.

- **Automatic creation of train/valid .txt files**: Generates obj.names, obj.data, obj.yaml, train.txt, and valid.txt files when creating train/valid .txt files, simplifying the dataset management process.

- **JSON to .txt conversion**: Easily convert JSON files to .txt files for compatibility with Darknet.

- **Batch creation of blank .txt files**: Create blank .txt files in your directory for efficient labeling.

- **Purge images without .txt and .txt files without images**: Automatically remove unnecessary files to keep your dataset clean and organized.

- **Combine .txt files**: Merge .txt files, so you don't have to move images manually.

- **Image augmentation with cropping and noise**: Crop images, place them on noisy backgrounds, and bring their labels with them for enhanced data augmentation.

- **Negative image generation**: Create negative images from your non-labeled data, ensuring that the trainer does not use any labeled data for negative samples.

- **Video support with Python weights**: Upload and play videos using Python weights for seamless integration and visualization.

## Getting Started

To get started with the Darknet-PyTorch Trainer, download the files  and follow the instructions in the README file to set up the required dependencies and environment.After setting up the environment, follow the README file's detailed instructions to train and fine-tune your Darknet models using the various features and options provided by the trainer. Happy training!
