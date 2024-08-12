# HEA MicroAnalyzer

## Project Overview

The HEA MicroAnalyzer is a GUI application developed for analyzing High Entropy Alloys (HEA) using Streamlit, PyCaret, and PyTorch. The application provides an interactive interface for analyzing alloy properties and performing structural analysis. The project aims to facilitate the analysis of HEAs by combining machine learning models with a user-friendly GUI.

### Features

- **Image Processing**: Processes 2D and 3D images of alloys using OpenCV.
- **Machine Learning**: Utilizes a Convolutional Neural Network (CNN) model implemented in PyTorch for image analysis and prediction.
- **Thresholding**: Applies a thresholding function to highlight specific features in the images.
- **Interactive Visualization**: Provides interactive visualizations of the original, thresholded, and processed images.
- **Property Prediction**: Predicts properties of alloys using a trained model.
- **Structural Analysis**: Performs structural analysis of alloys with a trained model saved as an `.h5` file.

## Dataset

The dataset used in this project contains detailed information about various High Entropy Alloys (HEA). 

### Data File

- **`data.csv`**: Contains detailed alloy information with the following columns:
  - `Alloy ID`: Unique identifier for each alloy.
  - `Alloy`: Alloy composition name.
  - `Al`, `Co`, `Cr`, `Fe`, `Ni`, `Cu`, `Mn`, `Ti`, `V`, `Nb`, `Mo`, `Zr`, `Hf`, `Ta`, `W`, `C`, `Mg`, `Zn`, `Si`, `Re`, `N`, `Sc`, `Li`, `Sn`, `Be`: Elemental composition percentages.
  - `Num_of_Elem`: Number of elements in the alloy.
  - `Density_calc`: Calculated density of the alloy.
  - `dHmix`: Mixing enthalpy.
  - `dSmix`: Mixing entropy.
  - `dGmix`: Mixing Gibbs free energy.
  - `Tm`: Melting temperature.
  - `n.Para`: Number of parameters.
  - `Atom.Size.Diff`: Atomic size difference.
  - `Elect.Diff`: Electrical difference.
  - `VEC`: Valence electron concentration.
  - `Sythesis_Route`, `Hot-Cold_Working`, `Homogenization_Temp`, `Homogenization_Time`, `Annealing_Temp`, `Annealing_Time_(min)`, `Quenching`, `HPR`, `Microstructure_`, `Multiphase`, `IM_Structure`, `Microstructure`, `Phases`, `References`: Processing and structural details.

### Preprocessed Data

After preprocessing, the dataset is used for predicting alloy properties and structural analysis. The preprocessed data includes:

- **`Num_of_Elem`**: Number of elements in the alloy.
- **`Density_calc`**: Calculated density of the alloy.
- **`dHmix`**: Mixing enthalpy.
- **`dSmix`**: Mixing entropy.
- **`Atom.Size.Diff`**: Atomic size difference.
- **`Elect.Diff`**: Electrical difference.
- **`VEC`**: Valence electron concentration.
- **`Phases`**: Phases present in the alloy.

## Code Overview

The codebase includes various modules and scripts for data processing, model training, and visualization. The following libraries are used:

- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Matplotlib & Seaborn**: Data visualization.
- **Plotly**: Interactive plots.
- **Scikit-learn**: Machine learning metrics and preprocessing.
- **PyCaret**: Model setup and comparison.
- **Streamlit**: Creating interactive web applications.
- **PyTorch**: For deep learning model implementation.
- **OpenCV**: Image processing.

**Sample Code:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pycaret.classification import *
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Load dataset
data = pd.read_csv('data.csv')

# Preprocess data
data = preprocess_data(data)

# Model training and evaluation
model = train_model(data)

# Streamlit interface
st.title('HEA MicroAnalyzer')
st.write('Interactive Alloy Analysis Tool')
```

## Model Training

### Model Type
- **Convolutional Neural Network (CNN)**

### Framework
- **PyTorch**

### File Format
- `.h5` (for structural analysis)

### Model Training Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
# ...

# Save model
torch.save(model.state_dict(), 'model.h5')

```

## Installation

To run the application, ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `pycaret`
- `streamlit`
- `torch`
- `opencv-python`
- `Pillow`

Install them using pip:

```sh
pip install pandas numpy matplotlib seaborn plotly scikit-learn pycaret streamlit torch opencv-python Pillow
```

## Usage

1. Clone the repository.
2. Install the required packages.
3. Run the Streamlit application:

   ```sh
   streamlit run app.py
   ```
4. Open your web browser and navigate to http://localhost:8501 to access the application.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.


For more details, please refer to the documentation and the project repository.

