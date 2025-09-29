<div align="center">

# 🤖 Predictive Keyboard Model

[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange?logo=streamlit)](https://streamlit.io/)

**A deep learning-based predictive keyboard application with LSTM neural networks**

</div>

---

## 📋 Table of Contents
- [✨ Features](#-features)
- [🛠️ Requirements](#️-requirements)
- [🚀 Setup](#-setup)
- [💻 Usage](#-usage)
- [📚 Model Architecture](#-model-architecture)
- [📁 File Structure](#-file-structure)
- [📄 License](#-license)

---

## ✨ Features

- 🧠 **LSTM-based predictive text model** for accurate next-word predictions
- ⚡ **Real-time word prediction** with instant results
- 🎯 **Top-k sampling** for diverse and relevant predictions
- 🌙 **Dark mode UI** for comfortable viewing in any lighting
- 🖥️ **Interactive web interface** built with Streamlit
- 📊 **Confidence scores** for each prediction
- 📈 **Configurable prediction count** to control output

---

## 🛠️ Requirements

### Python Dependencies
- `Python 3.7+`
- `PyTorch >= 1.9.0`
- `Streamlit >= 1.28.0`
- `NumPy >= 1.21.0`

### System Requirements
- Modern CPU (GPU recommended for training)
- 2GB+ RAM
- 50MB+ disk space

---

## 🚀 Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd predictive-keyboard-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Model File
You need the model file `predictive_keyboard_model.pth` in the project directory. 
This file is typically too large for Git repositories. If it's not included, you'll need to train your own model 
using the provided notebook or download a pre-trained model file.
   
To train your own model, run the `predictive_keyboard_model.ipynb` notebook with your training data.

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 💻 Usage

1. **Input Text**: Enter your text in the input field (e.g., "the quick brown")
2. **Configure Predictions**: Adjust the number of predictions to show using the slider
3. **Get Predictions**: Click "Get Predictions" to see the forecasted words
4. **View Results**: The model displays the top-k most likely next words with confidence scores

### Example Usage
- Input: "the quick"
- Predictions: ["brown", "way", "and", "time", "man"]
- Confidence Scores: [0.45, 0.23, 0.15, 0.12, 0.05]

---

## 📚 Model Architecture

The predictive model utilizes a sophisticated LSTM-based architecture:

- **Embedding Layer**: Converts tokens to dense vector representations
- **LSTM Layers**: Two-layer LSTM for sequence processing and context understanding
- **Dropout Layer**: Prevents overfitting with 30% dropout rate
- **Fully Connected Layer**: Outputs probability distribution over vocabulary
- **Special Tokens**: Support for padding, unknown, start-of-sequence, and end-of-sequence tokens

### Neural Network Specifications
| Component | Details |
|----------|---------|
| Embedding Dimension | 128 |
| Hidden Dimension | 256 |
| Number of Layers | 2 |
| Dropout Rate | 0.3 |
| Vocabulary Size | 4500+ |

---

## 📁 File Structure

```
predictive-keyboard-model/
├── app.py                 # Main Streamlit application
├── predictive_keyboard_model.pth   # Pre-trained model weights
├── predictive_keyboard_model.ipynb # Training notebook
├── dataset.txt            # Training data
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── LICENSE              # License information
```

### Key Files Description
- **`app.py`**: Interactive web application with dark mode UI
- **`predictive_keyboard_model.pth`**: Trained model weights and vocabulary
- **`predictive_keyboard_model.ipynb`**: Training notebook with detailed implementation
- **`dataset.txt`**: Text corpus for training the model
- **`requirements.txt`**: List of required Python packages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ using PyTorch and Streamlit**

⭐ Star this repository if you found it helpful!

</div>