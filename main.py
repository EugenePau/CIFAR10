from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import os
import re
import sys
import cv2
import torch
import torchvision
from torchsummary import summary
#from torchvision.transforms import v2
from torchvision import transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ignore warnings from PyTorch 
torchvision.disable_beta_transforms_warning()


class Global:
    folderpath = None 
    image_files = []
    aug_image_files = []
    image_inf = None
    qt_img_label = None
    qt_status_label = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    label = None




g = Global()


class VGG19:
    def __init__(self):
        pass
    
    def loadImage(self):
        img, _ = QtWidgets.QFileDialog.getOpenFileName()
        if img:
            g.image_inf = img
            pixmap = QtGui.QPixmap(img)
            pixmap = pixmap.scaled(200, 200)
            g.qt_img_label.setPixmap(pixmap)
        return 

    def loadMultiImage(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory()
        print(folderpath)
        if folderpath:
            g.folderpath = folderpath
            for img_title in os.listdir(folderpath):
                img = cv2.imread(os.path.join(folderpath, img_title))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                g.image_files.append((img_title, img))  ## format: [ ('title.png' , nparray ) , ... ]
            # self.plotImages(g.image_files)
        return

        
    def plotImages(self,image_files):
        plt.figure(figsize=(10, 10))
        for i, (title, img) in enumerate(image_files,start=1):
            plt.subplot(3, 3, i)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.title(re.sub(r'\.(.*)', '', title))
            plt.imshow(img)
        plt.show()
        return

        
    def showDataAugmentation(self):
        self.loadMultiImage()

        if g.image_files is None:
            print("Please load image first.")
            return

        for (title, img) in g.image_files:
            # Converts the OpenCV image to a PyTorch tensor [3, H, W]
            img_tensor = v2.ToImageTensor()(img)  # Use ToImageTensor instead of deprecated ToTensor

            # Transformations
            transforms = v2.Compose([
                v2.RandomRotation(32),
                v2.RandomHorizontalFlip(p=0.45),
                v2.RandomVerticalFlip(p=0.45),
                v2.ConvertImageDtype(torch.float32),  # Convert dtype to float32 (scaling handled here)
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = transforms(img_tensor)  # Apply transformations

            # Denormalize the image for visualization
            def denormalize(img_tensor, mean, std):
                mean = torch.tensor(mean).view(3, 1, 1)
                std = torch.tensor(std).view(3, 1, 1)
                img_tensor = img_tensor * std + mean  # Reverse normalization
                return img_tensor

            # Denormalize the transformed image
            img_denormalized = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Convert to NumPy and transpose dimensions for Matplotlib (H, W, C)
            img_np = img_denormalized.permute(1, 2, 0).numpy()

            # Clip values to the range [0, 1] (for valid visual representation)
            img_np = np.clip(img_np, 0, 1)

            # Append the augmented image into the global set
            g.aug_image_files.append((title, img_np))

            # Display the image
            #plt.imshow(img_np)
        self.plotImages(g.aug_image_files)
        return


    def showModelStructure(self):
        # Get the device (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the appropriate device
        model = torchvision.models.vgg19_bn(num_classes=10)
        model = model.to(device)

        # Ensure the summary call uses the same device
        summary(model, input_size=(3, 32, 32), device=str(device))

    def showTrainImages(self):
        (g.x_train, g.y_train), (g.x_test, g.y_test) = tf.keras.datasets.cifar10.load_data()
        if g.y_train is None or g.x_train is None:
            print("Please load image first")
            return


    def showAccAndLoss(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        image_path = os.path.join(script_dir, 'epoch_100.png')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.figure(figsize=(15, 10))  # Set the figure size
        plt.imshow(img)
        plt.axis('off')  # Turn off axis
        plt.title("Accuracy and Loss Visualization")  # Add a title
        plt.show()

    def inference(self):
        if g.image_inf is None:
            print("Please load image first.")
            return

        # Display the image using PyQt
        pixmap = QtGui.QPixmap(g.image_inf)
        pixmap = pixmap.scaled(200, 200)
        g.qt_img_label.setPixmap(pixmap)
        
        # Path containing trained model
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
        model_path = os.path.join(script_dir, 'model_epoch_40.pth')
        
        # check for GPU
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print("Using {} device".format(device))

        # Load the pre-trained model
        num_classes = 10
        model = torchvision.models.vgg19_bn()
        model.classifier[6] = nn.Linear(4096, num_classes)  # Modify final layer for 10 classes
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set model to evaluation mode

        # Load and preprocess the image
        image = Image.open(g.image_inf).convert('RGB')  # Open image and ensure it has 3 channels (RGB)
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # Normalize as per training and validation
        ])
        image_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension (shape: [1, 3, 32, 32])

        # Perform inference
        with torch.no_grad():  # Disable gradient computation
            output = model(image_tensor)  # Get logits
            print(f"Model output (logits): {output}")

        # Predicted class index (from your inference code)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()  # Convert tensor to a Python integer

        # Map the index to the class name
        predicted_class_name = g.classes[predicted_class_index]

        # Print the result
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Predicted class name: {predicted_class_name}")
        
        # Display the results in the GUI
        g.qt_status_label.setText(f"Predicted class = {predicted_class_name}")
        
        # Display the probability distribution in bar chart
        self.showProbBarChart(output)

        return 
        
    def showProbBarChart(self, logits):
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).squeeze()  # Shape: [10]

        # Convert probabilities tensor to numpy array for plotting
        probabilities = probabilities.numpy()

        # Plot the probabilities
        plt.figure(figsize=(10, 6))
        bars = plt.bar(g.classes, probabilities, color='blue')

        # Add numerical values on top of each bar
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{prob:.3f}', ha='center', fontsize=10)

        # Add labels and title
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Probability of Each Class', fontsize=14)
        plt.ylim(0, 1.1)  # Limit y-axis to fit bar labels
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.show()
        
        return




class Window:
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html

    def __init__(self):
        self.windowHeight = 720
        self.UnitWIDTH = 250
        self.UnitWIDTHWithSpace = self.UnitWIDTH+10
        self.vgg19 = VGG19()

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('2024 CvDl Hw2')
        self.window.resize(self.UnitWIDTHWithSpace*3, self.windowHeight)
        self.showImg = None
        self.boxA()
        self.boxB()

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self.window)
        btn1.setText('Load Image')
        btn1.clicked.connect(self.vgg19.loadImage)

        btn2 = QtWidgets.QPushButton(self.window)
        btn2.setText('1. Show Augmented Images')
        btn2.clicked.connect(self.vgg19.showDataAugmentation)

        btn3 = QtWidgets.QPushButton(self.window)
        btn3.setText('2. Show Model Structure')
        btn3.clicked.connect(self.vgg19.showModelStructure)

        btn4 = QtWidgets.QPushButton(self.window)
        btn4.setText('3. Show Accuracy and Loss')
        btn4.clicked.connect(self.vgg19.showAccAndLoss)

        btn5 = QtWidgets.QPushButton(self.window)
        btn5.setText('4. Inference')
        btn5.clicked.connect(self.vgg19.inference)

        box = QtWidgets.QGroupBox(title="VGG19", parent=self.window)
        box.setGeometry(0, 0, self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addWidget(btn4)
        layout.addWidget(btn5)

    def boxB(self):
        g.qt_img_label = QtWidgets.QLabel(self.window)
        g.qt_status_label = QtWidgets.QLabel(self.window)
        g.label = QtWidgets.QLabel(self.window)
        g.label.resize(100, 50)
        hbox = QtWidgets.QWidget(self.window)
        hbox.setGeometry(self.UnitWIDTHWithSpace, 0, 500, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(hbox)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(g.label)
        layout.addWidget(g.qt_img_label)
        layout.addWidget(g.qt_status_label)

    def render(self):
        self.window.show()
        sys.exit(self.app.exec())


window = Window()
window.render()
