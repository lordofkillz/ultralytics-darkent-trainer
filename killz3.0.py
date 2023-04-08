import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import random
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Type
import yaml
from noise import snoise2
from clearml import Task
from skimage.metrics import structural_similarity as compare_ssim
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem,QLineEdit,QPushButton

ui_file: Path = Path(__file__).resolve().parent / "killzv3.0.ui"
with open(ui_file, "r", encoding="utf-8") as file:
    Ui_MainWindow: Type[QtWidgets.QMainWindow]
    QtBaseClass: Type[object]
    Ui_MainWindow, QtBaseClass = uic.loadUiType(file)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.weights_files = []
        self.setWindowTitle("Ultimate KillZ Trainer 3000")
        self.cfg_dict = {}
        self.parsed_yaml = {}
        self.filename = ""
        self.file_type = ""
        self.import_stylesheet.clicked.connect(self.apply_stylesheet)
        self.import_button.clicked.connect(self.import_images)
        self.output_button.clicked.connect(self.output_paths)
        self.create_blanks_button.clicked.connect(self.select_images_path)
        self.browse_data.clicked.connect(self.browse_data_clicked)
        self.browse_weights.clicked.connect(self.browse_weights_clicked)
        self.train_button.clicked.connect(self.train_button_clicked)
        self.build.clicked.connect(self.run_commands)
        self.import_data_button.clicked.connect(self.import_data)
        self.calculate_anchors_button.clicked.connect(self.calculate_anchors)
        self.browse_cfg.clicked.connect(self.browse_cfg_clicked)
        self.data_input.clicked.connect(self.browse_yaml_clicked)
        self.model_input.clicked.connect(self.browse_pt_clicked)
        self.train2.clicked.connect(self.train2_button_clicked)
        self.img_video.clicked.connect(self.select_video_file)
        self.pytorch_version.clicked.connect(self.select_pytorch_file)
        self.auto_label.clicked.connect(self.auto_label_clicked)
        self.inputButton.clicked.connect(self.select_input_directory)
        self.outputButton.clicked.connect(self.select_output_directory)
        self.classes.clicked.connect(self.select_classes_file)
        self.selected_video_file = None
        self.selected_pytorch_file = None
        self.anchors_button.clicked.connect(self.import_anchors)
        self.btn_open_file.clicked.connect(self.cfg_open_clicked)
        self.btn_save_file.clicked.connect(self.cfg_save_clicked)
        self.purge.clicked.connect(self.purge_button_clicked)
        self.import_yaml_button.clicked.connect(self.import_yaml)
        self.cfg_table.cellChanged.connect(self.save_table_changes)
        self.hide_activation_checkbox.stateChanged.connect(
        self.toggle_activation_layers)
        self.file_paths = {"data": "", "cfg": "", "weights": []}
        self.parsed_yaml = None
        self.filename = None
        self.input_dir = None
        self.imported_anchors = None
        self.yaml_filename = ""
        self.inputLineEdit = QtWidgets.QLineEdit(self)
        self.inputLineEdit.hide()
        self.outputLineEdit = QtWidgets.QLineEdit(self)
        self.outputLineEdit.hide()
        self.clear_imports.clicked.connect(self.clear_stylesheet)
        self.import_chart.clicked.connect(self.load_chart)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.combine_txt_button.clicked.connect(self.on_combine_txt_clicked)
        self.combine_txt_flag = False
        self.process_image_button.clicked.connect(self.on_button_click)
        self.images_import = []
        self.label_files = []
        self.import_images_button.clicked.connect(self.import_images_triggered)
        self.crop_button.clicked.connect(self.process_images_triggered)
        self.show()

    def add_shadows(self, img):
        alpha = 0.3
        shadow_intensity = np.random.uniform(0.6, 0.8)
        shadow = np.zeros_like(img, dtype=np.float32)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                shadow_value = snoise2(x / 20, y / 20)
                shadow[y, x] = shadow_value * shadow_intensity
        img = cv2.addWeighted(img.astype(np.float32), alpha, shadow, 1 - alpha, 0)
        return img.astype(np.uint8)

    def blend_textures(self, img1, img2, alpha=0.5):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        return blended.astype(np.uint8)

    def generate_game_like_background(self, size):
        # Generate a base background using Perlin noise
        base_background = self.generate_perlin_noise_background(size, scale=10)

        # Generate a combined shapes background
        shapes_background = self.generate_combined_shapes_background(size)

        # Blend the two backgrounds
        blended_background = self.blend_textures(base_background, shapes_background, alpha=0.6)

        # Add shadows to the blended background
        game_like_background = self.add_shadows(blended_background)

        return game_like_background

    def generate_random_background(self, size):
        random_choice = np.random.randint(0, 5)
        if random_choice == 0:
            return np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        elif random_choice == 1:
            return self.generate_perlin_noise_background(size)
        elif random_choice == 2:
            return self.generate_combined_shapes_background(size)
        else:
            return self.generate_game_like_background(size)

    def generate_combined_shapes_background(self, size, num_shapes=100, num_lines=100, thickness=2, shape_color_range=(0, 255)):
        # Initialize the background with a sky-like color
        sky_color = (np.random.randint(100, 256), np.random.randint(100, 256), np.random.randint(200, 256))
        background = np.full((size[1], size[0], 3), sky_color, dtype=np.uint8)

        # Draw lines
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            x2, y2 = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            color = np.random.randint(shape_color_range[0], shape_color_range[1], 3)
            cv2.line(background, (x1, y1), (x2, y2), color.tolist(), thickness)

        # Draw other shapes
        for _ in range(num_shapes):
            shape_type = np.random.randint(0, 3)
            color = np.random.randint(shape_color_range[0], shape_color_range[1], 3)

            if shape_type == 1:  # Square
                x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
                side_length = np.random.randint(5, min(size) // 4)
                cv2.rectangle(background, (x, y), (x + side_length, y + side_length), color.tolist(), thickness)

            else:  # Rectangle
                x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
                width, height = np.random.randint(5, size[0] // 2), np.random.randint(5, size[1] // 2)
                cv2.rectangle(background, (x, y), (x + width, y + height), color.tolist(), thickness)

        return background



    def generate_perlin_noise_background(self, size, scale=10, intensity=255):
        background = np.zeros((size[1], size[0]), dtype=np.float32)
        for y in range(size[1]):
            for x in range(size[0]):
                noise_value = snoise2(x / scale, y / scale)
                background[y, x] = (noise_value + 1) * intensity / 2
        return cv2.cvtColor(np.uint8(background), cv2.COLOR_GRAY2BGR)

    def generate_random_background(self, size):
        random_choice = np.random.randint(0, 4)
        if random_choice == 0:
            return np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        elif random_choice == 2:
            return self.generate_perlin_noise_background(size)
        else:
            return self.generate_combined_shapes_background(size)

    def place_on_noisy_pad(self, image_path, label_path, output_path, target_size, output_format="jpg"):
        # Check the output format
        if output_format not in ["jpg", "png"]:
            raise ValueError("Invalid output format. Supported formats are 'jpg' and 'png'.")

        # Read YOLO label file
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Read the image
        image = cv2.imread(image_path)

        height, width, _ = image.shape

        for idx, label in enumerate(labels):
            class_id, x_center, y_center, width_bbox, height_bbox = map(float, label.split())

            # Calculate the bounding box coordinates
            x_min = int((x_center - width_bbox / 2) * width)
            y_min = int((y_center - height_bbox / 2) * height)
            x_max = int((x_center + width_bbox / 2) * width)
            y_max = int((y_center + height_bbox / 2) * height)

            # Get the labeled part of the image
            labeled = image[y_min:y_max, x_min:x_max]

            # Resize the labeled part while maintaining the aspect ratio
            labeled_height, labeled_width, _ = labeled.shape
            aspect_ratio = float(labeled_width) / float(labeled_height)

            if labeled_height >= target_size[1] or labeled_width >= target_size[0]:
                new_height = target_size[1] - 1
                new_width = int(aspect_ratio * new_height)
                if new_width >= target_size[0]:
                    new_width = target_size[0] - 1
                    new_height = int(new_width / aspect_ratio)
                labeled = cv2.resize(labeled, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                labeled_height, labeled_width, _ = labeled.shape

            # Place the labeled part on a noisy pad
            noisy_pad = self.generate_random_background(target_size)
            pad_height, pad_width, _ = noisy_pad.shape

            max_pad_top = pad_height - labeled_height
            max_pad_left = pad_width - labeled_width
            pad_top = np.random.randint(0, max_pad_top + 1)
            pad_left = np.random.randint(0, max_pad_left + 1)
            noisy_pad[pad_top:pad_top+labeled_height, pad_left:pad_left+labeled_width] = labeled

            # Save the image with the labeled part on the noisy pad
            output_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_padded_{idx}.{output_format}")
            cv2.imwrite(output_file, noisy_pad)

            # Adjust YOLO label coordinates for the new image
            new_x_center = (pad_left + labeled_width / 2) / pad_width
            new_y_center = (pad_top + labeled_height / 2) / pad_height
            new_width_bbox = labeled_width / pad_width
            new_height_bbox = labeled_height / pad_height
            updated_label = f"{int(class_id)} {new_x_center} {new_y_center} {new_width_bbox} {new_height_bbox}\n"

            # Save the updated YOLO label file
            output_label_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(label_path))[0]}_padded_{idx}.txt")
            with open(output_label_path, 'w') as f:
                f.write(updated_label)


    def process_images_triggered(self):
        if not self.height_input.text() or not self.width_input.text():
            QMessageBox.warning(self, "Error", "Please enter both height and width.")
            return

        target_height = int(self.height_input.text())
        target_width = int(self.width_input.text())
        target_size = (target_width, target_height)

        # Get the selected output format from the combo box
        output_format = self.jpg_png.currentText()

        if self.images_import and self.label_files:
            output_directory = os.path.dirname(self.images_import[0])
            output_path = os.path.join(output_directory, 'padded')
            os.makedirs(output_path, exist_ok=True)

            for image_path, label_path in zip(self.images_import, self.label_files):
                if os.path.basename(label_path) != "classes.txt":
                    self.place_on_noisy_pad(image_path, label_path, output_path, target_size, output_format=output_format)

            QMessageBox.information(self, "Success", "Images have been successfully processed.")


    def import_images_triggered(self):
        directory = QFileDialog.getExistingDirectory(None, 'Select Image Directory')
        if directory:
            self.images_import = glob.glob(os.path.join(directory, '*.png')) + \
                                glob.glob(os.path.join(directory, '*.jpg')) + \
                                glob.glob(os.path.join(directory, '*.jpeg'))
            self.label_files = glob.glob(os.path.join(directory, '*.txt'))



    def load_chart(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.chart_filename, _ = QFileDialog.getOpenFileName(
            self, "Open Chart", "", "PNG Files (*.png);;All Files (*)", options=options)

        if self.chart_filename:
            self.update_chart()
            # Start or restart the timer after successfully loading the chart
            self.timer.start(10000)

    def update_chart(self):
        if self.chart_filename:
            pixmap = QPixmap(self.chart_filename)
            self.chart_png.setPixmap(pixmap.scaled(
                self.chart_png.size(), aspectRatioMode=Qt.KeepAspectRatio))

    # import stylesheet
    def clear_stylesheet(self):
        self.setStyleSheet("")

    def apply_stylesheet(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Stylesheet File", "", "Stylesheet Files (*.qss *.css *.stylesheet);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r', encoding="utf-8") as f:
                stylesheet = f.read()
            self.setStyleSheet(stylesheet)
        else:
            # Reset to default UI if no file is selected
            self.setStyleSheet("")

     # convert json to yolo
    def select_input_directory(self):
        # Ask user to select input directory
        self.input_dir = str(QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select input directory"))
        self.inputLineEdit.setText(self.input_dir)

    def select_output_directory(self):
        # Ask user to select output directory
        output_dir = str(QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select output directory"))
        self.outputLineEdit.setText(output_dir)

        # Check if input_dir is defined before calling convert_to_yolo
        if self.input_dir is not None:
            # Loop over JSON files in input directory and write YOLO files to output directory
            self.convert_to_yolo(self.input_dir, output_dir)
        else:
            QMessageBox.warning(
                self, 'Warning', 'Please select an input directory.')

    def select_classes_file(self):
        # Ask user to select a classes file
        classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Classes File', '', 'Classes Files (*.txt)')
        if classes_file:
            # Load class names from selected file
            with open(classes_file, 'r', encoding="utf-8") as file:
                self.class_names = [line.strip() for line in file]
            self.num_classes = len(self.class_names)

    def convert_to_yolo(self, input_dir, output_dir):
        # Loop over JSON files in input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                # Load JSON data from file
                with open(os.path.join(input_dir, filename), 'r', encoding="utf-8") as f:
                    data = json.load(f)
                # Extract image dimensions
                width = data['image']['width']
                height = data['image']['height']

                # Loop over marks in JSON data
                for mark in data['mark']:
                    # Extract class name and rectangle coordinates
                    class_name = mark['name']
                    x_min = int(mark['rect']['int_x'])
                    y_min = int(mark['rect']['int_y'])
                    x_max = x_min + int(mark['rect']['int_w'])
                    y_max = y_min + int(mark['rect']['int_h'])

                    # Calculate YOLO parameters
                    x_center = (x_min + x_max) / (2.0 * width)
                    y_center = (y_min + y_max) / (2.0 * height)
                    w = int(mark['rect']['int_w']) / width
                    h = int(mark['rect']['int_h']) / height

                    # Write YOLO line to output file
                    class_idx = self.class_names.index(
                        class_name) if class_name in self.class_names else -1
                    if class_idx != -1:
                        line = f"{class_idx} {x_center:.10f} {y_center:.10f} {w:.10f} {h:.10f}"
                        output_filename = os.path.splitext(filename)[
                            0] + ".txt"
                        output_path = os.path.join(output_dir, output_filename)
                        with open(output_path, 'a', encoding="utf-8") as out_file:
                            out_file.write(line + '\n')

        QMessageBox.information(self, 'Information',
                                ' .txt files created successfully!')

    # ultralytics label predicttion
    def select_file(self, file_filter, dialog_title):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_file, _ = QFileDialog.getOpenFileName(
            None, dialog_title, "", file_filter, options=options)
        return selected_file

    def select_video_file(self):
        file_filter = "All Supported Files (*.mp4 *.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif);;Video Files (*.mp4);;Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif);;All Files (*)"
        selected_file = self.select_file(file_filter, "Select Image or Video File")
        if selected_file:
            self.selected_video_file = selected_file
            self.img_video_path_label.setText(selected_file)
            self.auto_label.setEnabled(True)

    def select_pytorch_file(self):
        file_filter = ".pt Files (*.pt);;All Files (*)"
        file_name = self.select_file(file_filter, "Select .pt File")
        if file_name:
            self.selected_pytorch_file = file_name
            self.pytorch_version_label.setText(file_name)

    def auto_label_clicked(self):
        if not self.selected_video_file:
            QMessageBox.warning(
                None, "Error", "Please select an image or video file.")
            return
        if not self.selected_pytorch_file:
            QMessageBox.warning(None, "Error", "Please select a PyTorch file.")
            return

        command = f"yolo task=detect mode=predict model='{self.selected_pytorch_file}' source='{self.selected_video_file}'"

        if self.video.isChecked():
            command += " show=true "

        print("Command to be executed: ", command)
        subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)

        self.img_video.setText("Selected File: " +
                               os.path.basename(self.selected_video_file))
        self.pytorch_version.setText(
            "Selected File: " + os.path.basename(self.selected_pytorch_file))

    # calculate anchors

    def import_data(self):
        self.data_path = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', 'c:\\', "Data files (*.data)")[0]
        self.calculate_anchors_button.setEnabled(True)

    def calculate_anchors(self):
        if self.data_path is None:
            # code to handle when data_path is not provided
            return
        command = "darknet detector calc_anchors {} -num_of_clusters {} -width {} -height {}".format(
            self.data_path,
            self.clusters_spinbox.value(),
            self.width_spinbox.value(),
            self.height_spinbox.value()
        )
        if self.show_checkbox.isChecked():
            command += " -show"
        print("Command to be executed: ", command)
        subprocess.Popen(command, shell=True)

        # def for ultralytics train

    def browse_yaml_clicked(self):
        file_name = self.open_file_dialog(
            "Select YAML File", "YAML Files (*.yaml);;All Files (*)")
        self.yaml_path = file_name
        self.yaml_label.setText(f"YAML file: {file_name}")

    def browse_pt_clicked(self):
        file_name = self.open_file_dialog(
            "Select PT File", "PT Files (*.pt);;All Files (*)")
        self.pt_path = file_name
        self.pt_label.setText(f"PT file: {file_name}")

    def train2_button_clicked(self):
        data_input = self.yaml_path
        model_input = self.pt_path
        imgsz_input = self.imgsz_input.text()
        epochs_input = self.epochs_input.text()
        batch_input = self.batch_input.text()
        task = self.task.currentText()  # get the selected task from the QComboBox
        command = "yolo task={} mode=train model={} data={} imgsz={} epochs={} batch={}".format(
            task, model_input, data_input, imgsz_input, epochs_input, batch_input)
        if self.workers.isChecked():
            workers_size = self.workers_input.text()
            if workers_size.isdigit():
                command += " workers=" + workers_size
            else:
                print("Invalid workers size. Please enter an integer.")
                return
            if self.store.isChecked():
                command += " cache=true"

        print("Command to be executed: ", command)
        process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
        process.stdin.write(os.linesep.encode())  # Send Enter key press
        process.stdin.flush()

        # DEF FOR TXT MAKER
    def import_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        directory = QFileDialog.getExistingDirectory(
            self, "Select Import Directory", options=options)
        if directory:
            self.images = []
            for file in os.listdir(directory):
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".gif"):
                    self.images.append(os.path.join(
                        directory, file).replace("\\", "/"))
                    self.image_label.setPixmap(
                        QPixmap(self.images[0]).scaled(96, 96))

    def output_paths(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file, _ = QFileDialog.getSaveFileName(self, "Select Output File", self.dropdown.currentText(
        ), "Text Files (*.txt);;All Files (*)", options=options)
        if file:
            with open(file, "w", encoding="utf-8") as f:
                for image in self.images:
                    f.write(image + "\n")

            # Get the directory of the output file
            output_dir = os.path.dirname(file)
            output_dir = output_dir.replace("\\", "/")

            # Create obj.names file if it doesn't exist
            obj_names_file = os.path.join(output_dir, "obj.names")
            if not os.path.exists(obj_names_file):
                with open(obj_names_file, "w", encoding="utf-8") as f:
                    f.write("person\n")

            class_numb = self.class_numb.value()
            class_name = self.class_name.text()
            use_valid_txt = self.dropdown.currentText() == "valid.txt"

            # Update obj.names file
            with open(obj_names_file, "w", encoding="utf-8") as f:
                f.write(class_name + "\n")

            # Create obj.data file if it doesn't exist
            data_file_path = os.path.join(output_dir, "obj.data")
            with open(data_file_path, "w") as f:
                f.write("classes = " + str(class_numb) + "\n")
                f.write("train  = " + os.path.join(output_dir, "train.txt").replace("/", "\\") + "\n")
                valid_path = "valid.txt" if use_valid_txt else "train.txt"
                f.write("valid  = " + os.path.join(output_dir, valid_path).replace("/", "\\") + "\n")
                f.write("names = " + os.path.join(output_dir, "obj.names").replace("/", "\\") + "\n")
                f.write("backup = " + os.path.join(output_dir, "backup").replace("/", "\\") + "\n")

            # Define obj.yaml file
            obj_yaml_file = os.path.join(output_dir, "obj.yaml")

            # Create or update obj.yaml file
            with open(obj_yaml_file, "w", encoding="utf-8") as f:
                f.write("# YOLOv8 configuration file\n\n")
                f.write("# Dataset path\n")
                f.write("path: " + output_dir + "\n\n")
                f.write("# Training and validation set paths\n")
                f.write("train: " + os.path.join(output_dir, "train.txt").replace("\\", "/") + "\n")
                valid_path = "valid.txt" if use_valid_txt else "train.txt"
                f.write("val: " + os.path.join(output_dir, valid_path).replace("\\", "/") + "\n\n")
                f.write("# Class names and number of classes\n")
                f.write("names:\n")

                # Write the class names
                class_names = class_name.split(',')
                for i, name in enumerate(class_names):
                    f.write("  {}: {}\n".format(i, name.strip()))

                f.write("nc: " + str(class_numb) + "\n")

            # Create the "backup" directory if it doesn't exist
            backup_dir = os.path.join(output_dir, "backup")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            QMessageBox.information(self, 'Information',
                                'obj.data,obj.yaml,obj.names,backup and train.txt has been created!')

            # def for build darkent

    def run_commands(self):
        try:
            # Check if Git is installed
            git_installed = subprocess.run(["cmd.exe", "/C", "git --version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if git_installed.returncode != 0:
                # Install Git if not installed
                subprocess.run(["powershell", "-Command", "Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.33.1.windows.1/Git-2.33.1-64-bit.exe -OutFile git-installer.exe"], check=True)
                subprocess.run(["cmd.exe", "/C", "git-installer.exe /SILENT"], check=True)
        except subprocess.CalledProcessError:
            # Show a message box if the installation fails
            QMessageBox.warning(self, "Warning", "Failed to install Git. Please downgrade Git or install it manually.")

        # Continue with the rest of the commands
        subprocess.run(["cmd.exe", "/C", "if not exist C:\\src mkdir C:\\src"], check=True)
        subprocess.run(["cmd.exe", "/C", "cd /d C:\\src & git clone https://github.com/AlexeyAB/darknet.git"], check=True)
        subprocess.run(["powershell", "-Command", "Set-ExecutionPolicy unrestricted -Scope CurrentUser -Force"], check=True)

        # Run the build command and print the output in real-time
        build_cmd = ["powershell", "-Command", "cd c:\\src\\darknet; .\\build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN"]
        with subprocess.Popen(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as build_process:
            while True:
                output = build_process.stdout.readline()
                if output == '' and build_process.poll() is not None:
                    break
                if output:
                    print(output.strip())

        build_process.wait()

        # def for train darknet

    def extract_value_from_cfg(self, cfg_lines, key, default_value):
        for line in cfg_lines:
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=")[1].strip()
        return default_value

    def extract_activation_layers(self, cfg_lines):
        activation_layers = []
        in_convolutional_section = False

        for line in cfg_lines:
            line = line.strip()

            if line == "[convolutional]":
                in_convolutional_section = True
            elif line.startswith("["):
                in_convolutional_section = False

            if in_convolutional_section and line.startswith("activation="):
                activation_layers.append(line.split("=")[1].strip())

        return ','.join(activation_layers)

    def extract_yolo_layers(self, cfg_lines):
        yolo_layers = []
        current_yolo_layer = {}
        in_yolo_section = False

        for line in cfg_lines:
            line = line.strip()

            if line == "[yolo]":
                in_yolo_section = True
                if current_yolo_layer:  # If a YOLO layer was parsed previously, add it to the list
                    yolo_layers.append(current_yolo_layer)
                current_yolo_layer = {}  # Reset the current YOLO layer dictionary
            elif line.startswith("["):
                in_yolo_section = False

            if in_yolo_section and "=" in line:
                key, value = line.split("=")
                current_yolo_layer[f"yolo_{len(yolo_layers)}_{key.strip()}"] = value.strip()

        if current_yolo_layer:  # Add the last YOLO layer to the list
            yolo_layers.append(current_yolo_layer)
        return yolo_layers

    def browse_data_clicked(self):
        file_name = self.open_file_dialog(
            "Select Data File", "Data Files (*.data);;All Files (*)")
        self.file_paths["data"] = file_name
        self.data_label.setText(f"Data file: {file_name}")

    def browse_cfg_clicked(self):
        file_name = self.open_file_dialog(
            "Select Config File", "Config Files (*.cfg);;All Files (*)")
        self.file_paths["cfg"] = file_name
        self.cfg_label.setText(f"Cfg file: {file_name}")

    def browse_weights_clicked(self):
        file_names = self.open_file_dialog(
            "Select Weights Files", "Weights Files (*.weights *.conv.*);;All Files (*)", multiple=True)
        self.file_paths["weights"] = [file for file in file_names if file.endswith(
            ('.weights', '.conv.')) or re.match('.*\.conv\.\d+', file)]
        self.weights_label.setText(
            "Weights files: " + ", ".join(self.file_paths["weights"]).rstrip())

    def train_button_clicked(self):
        # Initialize a new ClearML task
        task = Task.init(project_name="Darknet YOLO", task_name="yolo training")
        logger = task.get_logger()
        # Initialize a new ClearML task
        task = Task.init(project_name="Darknet YOLO", task_name="yolo training")
        parser = argparse.ArgumentParser()
        # Read the configuration file and extract the hyperparameters
        cfg_file = self.file_paths["cfg"]
        with open(cfg_file, "r") as f:
            cfg_lines = f.readlines()
        # Extract the values from the configuration file
        keys = ['batch', 'subdivisions', 'width', 'height', 'channels', 'momentum',
                'decay', 'angle', 'saturation', 'exposure', 'hue', 'learning_rate',
                'burn_in', 'max_batches', 'policy', 'steps', 'scales', 'cutmix', 'mosaic']
        defaults = [64, 16, 608, 608, 3, 0.949, 0.0005, 0, 1.5, 1.5, 0.1, 0.001, 1000,
                    500500, 'steps', '4000,4500', '.1,.1', 1, 1]
        for key, default in zip(keys, defaults):
            value = self.extract_value_from_cfg(cfg_lines, key, default)
            if isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            parser.add_argument(f'--{key}', type=type(default), default=value, help=f'{key} parameter')

        # Extract activation layers and YOLO layers
        activation_layers = self.extract_activation_layers(cfg_lines).split(',')
        parser.add_argument('--activation_layers', type=str, default=activation_layers, help='Activation layers in the model')

        yolo_layers = self.extract_yolo_layers(cfg_lines)
        activation_layers_dict = {}
        for index, activation_type in enumerate(activation_layers):
            activation_layers_dict[f'activation_{index}'] = activation_type

        yolo_layers = self.extract_yolo_layers(cfg_lines)
        for layer in yolo_layers:
            for key, value in layer.items():
                parser.add_argument(f'--{key}', type=str, default=value, help=f'{key} in the YOLO layer')

        # Add more arguments as necessary
        args = parser.parse_args()
        # Connect the arguments to the ClearML task
        task.connect(args)

        activation_layers_dict = {f'activation_{index}': activation_type for index, activation_type in enumerate(activation_layers)}
        task.connect(activation_layers_dict, name='activation_layers')

        data_file = self.file_paths["data"]
        cfg_file = self.file_paths["cfg"]
        weights_file = self.file_paths["weights"][-1] if self.file_paths["weights"] else ""
        command = f"darknet detector train {data_file} {cfg_file} {weights_file}"
        if self.cache.isChecked() and self.cache_input.text().isdigit():
            command += f" -cache {self.cache_input.text()}"
        command += " -dont_show" if self.dont_show_check.isChecked() else ""
        command += " -ext_output < data/train.txt > result.txt" if self.results.isChecked() else ""
        command += " -map" if self.map_check.isChecked() else  ""
        command += " -clear" if self.clear_check.isChecked() else  ""
        command += " -random" if self.random.isChecked() else ""
        command += " -gpus 0,1 " if self.gpu1.isChecked() else ""
        command += " -gpus 0,1,2 " if self.gpu2.isChecked() else ""

        print(f"Command to be executed: {command}")
        subprocess.Popen(command, shell=True)

    def open_file_dialog(self, caption, file_filter, multiple=False):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        if multiple:
            file_names, _ = QFileDialog.getOpenFileNames(
                self, caption, "", file_filter, options=options)
            return file_names
        else:
            file_name, _ = QFileDialog.getOpenFileName(
                self, caption, "", file_filter, options=options)
            return file_name

    # yaml parser

    def clear_table(self):
        self.cfg_table.setRowCount(0)

    # Inside import_yaml method
    def import_yaml(self):
        if self.file_type != "yaml":
            self.clear_table()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        initial_directory = self.default_yaml_path.text()  # Get the text from the default_yaml_path widget
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Select YAML File", initial_directory, "YAML Files (*.yaml *.yml);;All Files (*)", options=options)

        if file_name:
            self.hide_activation_checkbox.setChecked(False)  # Comment out or remove this line
            self.yaml_filename = file_name
            self.default_yaml_path.setText(file_name)  # Set the path to the YAML file in the default_yaml_path widget
            with open(file_name, 'r', encoding="utf-8") as f:
                self.parsed_yaml = yaml.safe_load(f)
            self.file_type = "yaml"
            self.cfg_table.setColumnCount(2)
            self.cfg_table.setHorizontalHeaderLabels(["Key", "Value"])
            self.cfg_table.setRowCount(len(self.parsed_yaml))
            for row, (key, value) in enumerate(self.parsed_yaml.items()):
                self.cfg_table.setItem(row, 0, QTableWidgetItem(str(key)))
                self.cfg_table.setItem(row, 1, QTableWidgetItem(str(value)))
            self.filename = None  # Clear the CFG file when a YAML file is opened
            self.cfg_open_label.setText("")  # Clear the CFG path from the cfg_open_label widget

    def save_table_changes(self, row, column):
        key_item = self.cfg_table.item(row, 0)
        value_item = self.cfg_table.item(row, 1)

        if key_item is not None and value_item is not None:
            key = key_item.text()
            value = value_item.text()

            if self.file_type == "yaml" and self.yaml_filename is not None:
                self.parsed_yaml[key] = self.parse_value(value)
                with open(self.yaml_filename, 'w', encoding="utf-8") as f:
                    yaml.dump(self.parsed_yaml, f)

    def parse_value(self, value):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        return value

        # def for cfg editor
    def import_anchors(self):
        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please open a .cfg file before importing anchors.")
            return

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Anchors File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, 'r', encoding="utf-8") as f:
                anchors = f.read().strip()

            self.update_cfg_anchors(anchors)

    # Inside cfg_open_clicked method
    def cfg_open_clicked(self):
        self.hide_activation_checkbox.setChecked(False)
        if self.file_type != "cfg":
            self.clear_table()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        initial_directory = "C:/EAL/Scripts/cfg"  # Set the initial directory
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Config File", initial_directory, "Config Files (*.cfg);;All Files (*)", options=options)
        if file_name:
            self.filename = file_name
            self.cfg_open_label.setText("Cfg: " + file_name)
            self.parse_cfg_file(file_name)
            self.yaml_filename = None  # Clear the YAML file when a CFG file is opened
            self.default_yaml_path.setText("")  # Clear the default_yaml_path widget


    def toggle_activation_layers(self):
        # Check if 'activation_row_count' attribute exists
        if not hasattr(self, 'activation_row_count'):
            return  # If not, simply return without performing any action

        for row in self.activation_row_count.values():
            if self.hide_activation_checkbox.isChecked():
                self.cfg_table.hideRow(row)
            else:
                self.cfg_table.showRow(row)

    def parse_cfg_file(self, file_name=None, anchors_list=None):
        if file_name is None:
            file_name = self.filename

        with open(file_name, 'r', encoding="utf-8") as f:
            config = f.read()

        # Initialize table and properties
        self.activation_row_count = {}
        activation_count = 0
        self.cfg_table.setRowCount(0)
        self.cfg_table.setColumnCount(2)
        self.cfg_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.cfg_table.setAlternatingRowColors(True)

        # Regular expressions to match the relevant sections of the config file.
        sections = re.findall(r"\[(.*?)\]\s*([^[]*)", config, re.DOTALL)

        # Replace the existing anchors with the imported ones and update the section content
        if self.imported_anchors is not None:
            for idx, (section_type, section_content) in enumerate(sections):
                if section_type == "yolo":
                    section_lines = section_content.strip().split("\n")
                    section_dict = {line.split("=")[0].strip(): line.split(
                        "=")[1].strip() for line in section_lines if "=" in line}

                    # Replace the existing anchors with the imported ones
                    section_dict["anchors"] = ', '.join(
                        [f"{x},{y}" for x, y in self.imported_anchors])

                    # Update the section_content
                    sections[idx] = (section_type, '\n'.join(
                        [f"{key} = {value}" for key, value in section_dict.items()]))

        for idx, (section_type, section_content) in enumerate(sections):
            section_lines = section_content.strip().split("\n")
            section_dict = {line.split("=")[0].strip(): line.split(
                "=")[1].strip() for line in section_lines if "=" in line}

            if section_type == "net":
                net_items = ["batch", "subdivisions", "width", "height", "saturation", "exposure", "hue", "max_batches", "flip",
                             "mosaic", "letter_box", "cumtmix", "mosaic_bound", "mosaic_scale", "mosaic_center", "mosaic_crop", "mosaic_flip"]

                for item in net_items:
                    if item in section_dict:
                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(row_count, 0, QtWidgets.QTableWidgetItem(
                            f"{item}_0"))  # Add the index to the parameter name
                        self.cfg_table.setItem(
                            row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

                self.net_dict = section_dict
            elif section_type == "convolutional":
                is_before_yolo = idx < len(
                    sections) - 1 and sections[idx + 1][0] == "yolo"
                conv_items = ["activation"]

                for item in conv_items:
                    if item in section_dict and (is_before_yolo or item != "filters"):
                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(
                            row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "activation":
                            activation_combo = QtWidgets.QComboBox()
                            activation_combo.addItems(
                                ["leaky", "mish", "swish", "linear"])
                            activation_combo.setCurrentText(section_dict[item])
                            self.cfg_table.setCellWidget(
                                row_count, 1, activation_combo)
                            self.activation_row_count[activation_count] = row_count
                            activation_count += 1
                        else:
                            self.cfg_table.setItem(
                                row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))
            elif section_type == "yolo":
                yolo_items = ["anchors", "classes", "ignore_thresh", "random"]

                for item in yolo_items:
                    if item in section_dict:
                        row_count = self.cfg_table.rowCount()
                        self.cfg_table.insertRow(row_count)
                        self.cfg_table.setItem(
                            row_count, 0, QtWidgets.QTableWidgetItem(f"{item}_{idx}"))

                        if item == "anchors" and self.imported_anchors is not None:
                            self.cfg_table.setItem(
                                row_count, 1, QtWidgets.QTableWidgetItem(self.imported_anchors))
                        else:
                            self.cfg_table.setItem(
                                row_count, 1, QtWidgets.QTableWidgetItem(section_dict[item]))

        # Resize table columns to fit content
        self.cfg_table.resizeColumnsToContents()
        self.cfg_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)

    def cfg_save_clicked(self):
        if self.filename:
            # Create a dictionary with the parameter values from the table widget
            table_data = {}
            for row in range(self.cfg_table.rowCount()):
                param = self.cfg_table.item(row, 0).text()

                if param.startswith("activation"):
                    activation_count = {v: k for k, v in self.activation_row_count.items()}[
                        row]
                    value = self.cfg_table.cellWidget(row, 1).currentText()
                    table_data[f"{param}"] = value
                else:
                    item = self.cfg_table.item(row, 1)
                    value = item.text() if item else ""
                    table_data[param] = value
            # Find the corresponding max_batches value and update steps
            if "max_batches_0" in table_data:
                max_batches = int(table_data["max_batches_0"])
                steps_70 = int(max_batches * 0.7)
                steps_80 = int(max_batches * 0.8)
                steps_90 = int(max_batches * 0.9)
                table_data["steps_0"] = f"{steps_80},{steps_90}"

            # Find the corresponding classes value for each YOLO layer
            yolo_classes = {}
            for key, value in table_data.items():
                if "classes" in key:
                    section_idx = int(key.split('_')[-1])
                    yolo_classes[section_idx] = int(value)

            # Read the original configuration file line by line and modify the relevant lines
            new_config = ""
            section_idx = -1
            current_section = ""
            yolo_layer_indices = []
            conv_before_yolo = False
            with open(self.filename, "r", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()

                    if stripped_line.startswith("["):
                        section_idx += 1

                        if conv_before_yolo:
                            # store the index of the convolutional layer before the YOLO layer
                            yolo_layer_indices.append(section_idx - 1)
                            conv_before_yolo = False

                        current_section = stripped_line.strip("[]")
                        new_config += line

                        if current_section == "convolutional":
                            conv_before_yolo = True
                        elif current_section == "yolo":
                            conv_before_yolo = False
                    elif "=" in stripped_line:
                        param, value = stripped_line.split("=")
                        param = param.strip()

                        if current_section == "net":
                            new_param = f"{param}_0"
                        else:
                            new_param = f"{param}_{section_idx}"

                        new_value = table_data.get(new_param, value.strip())

                        if param == "filters" and conv_before_yolo:
                            classes = yolo_classes.get(section_idx + 1)
                            if classes is not None:
                                new_value = (classes + 5) * 3
                                new_line = f"{param}={new_value}\n"
                                new_config += new_line
                                continue  # Skip the rest of the loop, so the old filter line is not added

                        new_line = f"{param}={new_value}\n"
                        new_config += new_line
                    elif stripped_line.startswith("#"):
                        new_config += line
                    else:
                        new_config += line

            # Add the save dialog code block here
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            save_file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Config File As", "", "Config Files (*.cfg);;All Files (*)", options=options)

            if save_file_name:
                # Update the filters in the convolutional layer before the YOLO layers based on the new classes

                # Add .cfg extension to the save_file_name if it doesn't have it
                if not save_file_name.endswith('.cfg'):
                    save_file_name += '.cfg'

                # Save the modified configuration to the selected file
                with open(save_file_name, 'w', encoding='utf-8') as f:
                    f.write(new_config)

                # Show a success message
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setWindowTitle("Success")
                msg.setText("Configuration file saved successfully.")
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()

    def update_cfg_anchors(self, anchors):
        for row in range(self.cfg_table.rowCount()):
            item_key = self.cfg_table.item(row, 0)
            if item_key and "anchors" in item_key.text():
                # Pass the string representation of the anchors
                new_var = self.new_method(str(anchors))
                self.cfg_table.setItem(row, 1, new_var)

    def new_method(self, text):
        # Return a QTableWidgetItem instance with the text
        return QtWidgets.QTableWidgetItem(text)

    # create blank txt files for images without labels in the same folder

    def select_images_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        images_path = QFileDialog.getExistingDirectory(
            self, 'Select Images Path', '', options=options)
        if images_path:
            self.create_blanks(images_path)

    def create_blanks(self, images_path):
        for img in [f for f in os.listdir(images_path) if f.endswith(('png', 'jpg'))]:
            txt_file_path = os.path.join(images_path, img[:-4]+'.txt')
            if not os.path.exists(txt_file_path):
                with open(txt_file_path, 'x', encoding='utf8'):
                    pass

        QMessageBox.information(self, 'Information',
                                'Blank files created successfully!')

    # purge images without labels png and jgp with txt and move them to a new folder called purged
    def purge_files(self, directory):
        files_to_move = []

        for filename in os.listdir(directory):
            if filename.endswith(".txt") and "classes.txt" not in filename:
                img = os.path.join(directory, filename.split('.')[0] + '.jpg')
                img2 = os.path.join(directory, filename.split('.')[0] + '.png')
                if not os.path.exists(img) and not os.path.exists(img2):
                    files_to_move.append(os.path.join(directory, filename))
            elif filename.endswith(('.png', '.jpg')):
                txt = os.path.join(directory, filename.split('.')[0] + '.txt')
                xml_file = os.path.join(
                    directory, filename.split('.')[0] + '.xml')
                if not os.path.exists(txt) and not os.path.exists(xml_file):
                    files_to_move.append(os.path.join(directory, filename))
            elif filename.endswith('.json'):
                files_to_move.append(os.path.join(directory, filename))

        return files_to_move

    def purge_button_clicked(self):
        input_directory = QFileDialog.getExistingDirectory(
            None, "Select Directory")
        if not input_directory:
            return
        files_to_move = self.purge_files(input_directory)
        output_directory = QFileDialog.getExistingDirectory(
            None, "Choose Output Directory")
        if not output_directory:
            return
        purged_directory = os.path.join(output_directory, 'purged')
        os.makedirs(purged_directory, exist_ok=True)
        for file in files_to_move:
            input_file_path = os.path.join(
                input_directory, os.path.basename(file))
            output_file_path = os.path.join(
                purged_directory, os.path.basename(file))
            os.rename(input_file_path, output_file_path)

        QMessageBox.information(self, 'Information',
                                'Purge completed successfully!')
     # cobnine txt files
    def on_combine_txt_clicked(self):
        if self.combine_txt_flag:
            print("Function is already running.")
            return
        self.combine_txt_flag = True
        print("Function called.")
        self.combine_txt_button.setEnabled(False)

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file1, _ = QFileDialog.getOpenFileName(
            self, "Select First File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file1:
            self.combine_txt_flag = False
            print("File 1 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        file2, _ = QFileDialog.getOpenFileName(
            self, "Select Second File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not file2:
            self.combine_txt_flag = False
            print("File 2 prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Save Combined File", "", "Text Files (*.txt);;All Files (*)", options=options
        )
        if not output_file:
            self.combine_txt_flag = False
            print("Output file prompt cancelled.")
            self.combine_txt_button.setEnabled(True)
            return

        try:
            unique_lines = set()
            with open(file1, "r") as f1, open(file2, "r") as f2:
                for line in f1:
                    unique_lines.add(line)
                for line in f2:
                    unique_lines.add(line)

            with open(output_file, "w") as output:
                for line in unique_lines:
                    output.write(line)

            QMessageBox.information(self, "Success", "Files have been combined and saved successfully!")
            self.combine_txt_flag = False
            print("Function finished successfully.")
            self.combine_txt_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while combining files: {e}")
            self.combine_txt_flag = False
            print("Function finished with error.")
            self.combine_txt_button.setEnabled(True)
        #img padding/tiling option.

    def on_button_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images and YOLO Annotation Files", "", options=options)
        if not folder_path:
            return

        save_folder_path = os.path.join(folder_path, "tiled")

        # Get the selected percentage from the combo box
        limit_percentage = int(self.limit_blanks.currentText().rstrip('%'))

        # Get the selected image format from the combo box
        image_format = self.image_format_combo.currentText().lower()

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]
        yolo_annotation_files = [os.path.splitext(image_file)[0] + '.txt' for image_file in image_files]

        for image_file, yolo_annotation_file in zip(image_files, yolo_annotation_files):
            self.process_image(image_file, yolo_annotation_file, save_folder_path, limit_percentage, image_format)

        QMessageBox.information(self, "Information", "Processing finished check the tiled folder")

        # Process the image

    def process_image(self, image_path, yolo_annotation_path, save_folder_path, limit_percentage, image_format):
        folder_path = os.path.dirname(image_path)

        img = cv2.imread(image_path)
        original_img = img.copy()
        with open(yolo_annotation_path, 'r') as f:
            yolo_data = f.readlines()

        img_height, img_width, _ = img.shape

        labeled_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        margin = int(0.005 * min(img_height, img_width))  #% margin in each direction

        for data in yolo_data:
            data = data.split()
            class_id = data[0]
            x_center, y_center, w, h = [float(x) for x in data[1:]]
            x_min = max(int((x_center - w/2) * img_width) - margin, 0)
            x_max = min(int((x_center + w/2) * img_width) + margin, img_width)
            y_min = max(int((y_center - h/2) * img_height) - margin, 0)
            y_max = min(int((y_center + h/2) * img_height) + margin, img_height)

            labeled_img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]
            img[y_min:y_max, x_min:x_max] = 0
            mask[y_min:y_max, x_min:x_max] = 255

        img = self.fill_cropped_area(original_img, labeled_img, img, mask, image_path, folder_path)

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save the negative image and its corresponding negative.txt file
        negative_filename = f'{base_filename}_no_labels_negative'
        negative_img = img.copy()
        self.save_image_with_limit(negative_img, save_folder_path, negative_filename, limit_percentage, image_format, labeled_img)


    def save_image_with_limit(self, img, save_folder_path, filename, limit, image_format, labeled_img):
        save_negative_image = random.choices([True, False], weights=[limit, 100 - limit], k=1)[0]

        if save_negative_image:
            negative_filename = f'{save_folder_path}/{filename}.{image_format}'
            cv2.imwrite(negative_filename, img)

            with open(f'{save_folder_path}/{filename}.txt', 'w') as f:
                pass
        # Fill the empty regions with random patches from the original image
    def fill_cropped_area(self, original_img, labeled_img, img, mask, image_path, folder_path):
        height, width, _ = original_img.shape

        # Define the maximum patch size
        max_patch_size = 50

        # Read all the images from the directory
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]
        images = [cv2.imread(image_file) for image_file in image_files]

        # Fill the empty regions with random patches from the original image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            max_attempts = 100
            found_patch = False
            for attempt in range(max_attempts):
                # Choose a random image from the list
                random_img = random.choice(images)

                random_img_height, random_img_width, _ = random_img.shape
                x_rand = random.randint(0, random_img_width - w)
                y_rand = random.randint(0, random_img_height - h)

                if np.sum(mask[y_rand:y_rand + h, x_rand:x_rand + w]) == 0:
                    img[y:y + h, x:x + w] = random_img[y_rand:y_rand + h, x_rand:x_rand + w]
                    found_patch = True
                    break

            if not found_patch:
                # Break the large regions into smaller patches
                for patch_y in range(y, y + h, max_patch_size):
                    for patch_x in range(x, x + w, max_patch_size):
                        patch_w = min(max_patch_size, x + w - patch_x)
                        patch_h = min(max_patch_size, y + h - patch_y)

                        for attempt in range(max_attempts):
                            # Choose a random image from the list
                            random_img = random.choice(images)

                            random_img_height, random_img_width, _ = random_img.shape
                            x_rand = random.randint(0, random_img_width - patch_w)
                            y_rand = random.randint(0, random_img_height - patch_h)

                            img[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w] = random_img[y_rand:y_rand + patch_h, x_rand:x_rand + patch_w]
                            break
                        if attempt == max_attempts - 1:
                            print(f"Warning: Could not find a suitable random patch for image {os.path.basename(image_path)} after {max_attempts} attempts.")

        return img


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    sys.exit(app.exec_())