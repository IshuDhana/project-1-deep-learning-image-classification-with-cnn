🧠 Deep Learning Project: Image Classification with CNN
Project I | Deep Learning | Image Classification using Convolutional Neural Networks (CNN)


📋 Project Overview
This project focuses on developing and deploying a Convolutional Neural Network (CNN) for image classification.
 The model is designed to classify images from two possible datasets — CIFAR-10 or a custom 10-class animal dataset — into predefined categories.
The primary objective is to:
Build and train a CNN from scratch using Keras/TensorFlow.


Apply data preprocessing, augmentation, and transfer learning.


Evaluate and visualize model performance.


Deploy the best-performing model via an interactive Streamlit app.



🧩 Repository Structure
project-1-deep-learning-image-classification-with-cnn/
│
├── G4_CNN_adam_model_keras.ipynb    # Main deep learning notebook (model training & evaluation)
├── app.py                            # Streamlit user interface for model deployment
├── requirements.txt                  # List of dependencies
├── model/                            # Saved model weights & architecture files
│   └── cnn_model.h5
├── data/                             # (optional) local dataset folder if not using CIFAR-10
├── assets/                           # Images, visualizations, or logos used in the README/app
├── README.md                         # Project documentation (this file)
└── utils/                            # Helper scripts (data loading, preprocessing, visualization)


🧠 Task Description
Students were tasked to:
Build a CNN-based image classifier.


Train and evaluate the model using CIFAR-10 or the 10-class animal dataset.


Apply transfer learning with pre-trained models (e.g., VGG16, Inception).


Deploy the final model in a user-friendly interface.



🧬 Datasets
You may use one of the following datasets:
Option 1: CIFAR-10
Images: 60,000 color images (32x32 pixels)


Classes: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)


Source: CIFAR-10 Dataset


Option 2: Animal Dataset
Images: ~28,000 medium-quality animal images


Classes: dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant


Source: Animal-10 Dataset



⚙️ Key Components
1. Data Preprocessing
Loading and splitting the dataset (train/test/validation)


Normalization and one-hot encoding of labels


Data augmentation using:


Rotation, flipping, zooming, shifting


Visualizing sample images and their corresponding labels


2. Model Architecture
Implemented CNN structure includes:
Multiple convolutional layers with ReLU activation


MaxPooling layers for spatial reduction


Dropout layers for regularization


Dense fully connected layers


Softmax output layer for classification


📄 The main model is implemented in:
 G4_CNN_adam_model_keras.ipynb
3. Model Training
Optimizer: Adam


Loss: Categorical Crossentropy


Metrics: Accuracy


Techniques used:


Early stopping


Validation split


Learning rate tuning


4. Model Evaluation
Metrics calculated:


Accuracy, Precision, Recall, F1-Score


Confusion matrix visualization


Classification report for class-wise analysis


Training history plots (accuracy vs loss)


5. Transfer Learning
Experimented with VGG16 and InceptionV3


Layers frozen selectively to preserve learned features


Compared results with base CNN


Reported performance improvements and trade-offs



🚀 Model Deployment
Framework: Streamlit
The Streamlit web app allows users to:
Upload one or multiple images


View predictions with class probabilities


See sample outputs and class visualizations


📁 App Entry Point:
 app.py
To run locally:
pip install -r requirements.txt
streamlit run app.py

The Gradio version was also implemented during testing but Streamlit was finalized for deployment due to its superior user experience and layout flexibility.

📊 Results & Insights
Model Type
Accuracy
Notes
Custom CNN (Adam)
~85%
Strong baseline
VGG16 (Transfer Learning)
~92%
Improved generalization
InceptionV3 (Transfer Learning)
~90%
Slightly lower due to overfitting

🧩 Best Model:
 VGG16 (Transfer Learning) — achieved highest validation accuracy and stable generalization.
Insights:
Data augmentation was crucial for avoiding overfitting.


Transfer learning provided significant performance gain.


Adam optimizer outperformed SGD for this dataset.


Streamlit provided a smooth model deployment experience.



🛠️ Tech Stack
Category
Technologies
Language
Python
Deep Learning
TensorFlow, Keras
Data Handling
NumPy, Pandas
Visualization
Matplotlib, Seaborn
Deployment
Streamlit
Transfer Learning
VGG16, InceptionV3


🧪 How to Run
1️⃣ Clone the repository
git clone https://github.com/IshuDhana/project-1-deep-learning-image-classification-with-cnn.git
cd project-1-deep-learning-image-classification-with-cnn

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional)
If you want to retrain:
jupyter notebook G4_CNN_adam_model_keras.ipynb

4️⃣ Run the Streamlit app
streamlit run app.py

5️⃣ Upload Images and Predict
Drag and drop images into the app.


Get predictions and confidence scores instantly.



📈 Future Work
Experiment with TensorFlow Serving for cloud-based deployment (+5 bonus points goal)


Optimize the model with pruning or quantization


Add real-time webcam classification feature


Deploy via Hugging Face Spaces or Streamlit Cloud



👨‍💻 Contributors
Name
Role
Ishu Dhana
Model development, UI deployment, documentation
Group 4 Team Members
Data preprocessing, evaluation, and transfer learning experiments


📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

🌐 Project Links
🔗 GitHub Repo: project-1-deep-learning-image-classification-with-cnn
 🎯 Demo (Streamlit App): [To be added if deployed]

