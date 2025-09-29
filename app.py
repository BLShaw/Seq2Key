import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import string
import re
from collections import Counter, OrderedDict
import random
import numpy as np
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vocabulary:
    """Class to handle vocabulary and tokenization"""

    def __init__(self):
        # Special tokens
        self.PAD = 0  # Padding token
        self.UNK = 1  # Unknown token
        self.SOS = 2  # Start of sequence
        self.EOS = 3  # End of sequence

        # Initialize vocabulary with special tokens
        self.token2idx = {
            '<PAD>': self.PAD,
            '<UNK>': self.UNK,
            '<SOS>': self.SOS,
            '<EOS>': self.EOS
        }

        self.idx2token = {
            self.PAD: '<PAD>',
            self.UNK: '<UNK>',
            self.SOS: '<SOS>',
            self.EOS: '<EOS>'
        }

        self.token_counts = Counter()

    def build_vocab(self, text_data, min_freq=2):
        """Build vocabulary from text data"""
        # Tokenize the text
        tokens = self.tokenize(text_data)

        # Count tokens
        self.token_counts.update(tokens)

        # Add tokens that appear more than min_freq to vocabulary
        for token, count in self.token_counts.items():
            if count >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def tokenize(self, text):
        """Simple tokenization function"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower()
        # Remove punctuation and replace with space (but keep apostrophes)
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Split into tokens
        tokens = text.strip().split()
        return tokens

    def encode(self, text):
        """Convert text to indices"""
        tokens = self.tokenize(text)
        # Add SOS and EOS tokens
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        encoded = [self.token2idx.get(token, self.UNK) for token in tokens]
        return encoded

    def decode(self, indices):
        """Convert indices back to text"""
        tokens = [self.idx2token.get(idx, '<UNK>') for idx in indices]
        # Remove special tokens for readability
        tokens = [token for token in tokens if token not in ['<SOS>', '<EOS>', '<PAD>']]
        return ' '.join(tokens)

    @property
    def size(self):
        return len(self.token2idx)


class LSTMModel(nn.Module):
    """LSTM model for next word prediction"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Reshape for linear layer: (batch_size * seq_len, hidden_dim)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Output: (batch_size * seq_len, vocab_size)
        output = self.fc(lstm_out)

        # Reshape back to (batch_size, seq_len, vocab_size)
        output = output.view(x.size(0), x.size(1), -1)

        return output, hidden


def load_model(model_path, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
    """Load a trained model"""
    # Initialize vocabulary
    vocab = Vocabulary()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Restore vocabulary
    vocab.token2idx = checkpoint['vocab_token2idx']
    vocab.idx2token = {}
    for key, value in checkpoint['vocab_idx2token'].items():
        try:
            vocab.idx2token[int(key)] = value
        except ValueError:
            # Handle cases where key isn't numeric
            continue
    vocab.token_counts = checkpoint['vocab_token_counts']

    # Initialize model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, vocab


def predict_next_words(model, vocab, input_text, top_k=5, max_length=20):
    """Generate predictions for next words using top-k sampling"""
    model.eval()

    # Tokenize input
    tokens = vocab.tokenize(input_text.lower())
    token_indices = [vocab.token2idx.get(token, vocab.UNK) for token in tokens]

    if not token_indices:
        # If no valid tokens, return some common words
        common_words = ['the', 'and', 'to', 'a', 'in']
        return [(word, 0.2) for word in common_words]

    # Convert to tensor
    input_tensor = torch.LongTensor([token_indices]).to(device)

    model.eval()
    with torch.no_grad():
        # Get LSTM output for the input sequence
        output, hidden = model(input_tensor)

        # Get the output for the last token in the sequence
        last_output = output[0, -1, :]  # Shape: (vocab_size,)

        # Get top-k predictions
        top_k_values, top_k_indices = torch.topk(last_output, k=top_k)

        # Convert to probabilities using softmax
        probs = torch.softmax(top_k_values, dim=0)

        # Get predicted words
        predicted_words = []
        for i in range(min(top_k, len(top_k_indices))):  # Prevent index out of bounds
            word_idx = top_k_indices[i].item()
            word = vocab.idx2token.get(word_idx, '<UNK>')
            # Filter out special tokens and ensure word is not empty
            if word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] and word.strip() != '' and word != '<PAD>':
                prob = probs[i].item()
                predicted_words.append((word, prob))
        
        # If all top predictions were filtered out, return some common words
        if not predicted_words:
            common_words = ['the', 'and', 'to', 'a', 'in']
            return [(word, 0.2) for word in common_words]
        
        # Limit to top_k after filtering
        predicted_words = predicted_words[:top_k]

    return predicted_words


# Initialize the app
st.set_page_config(
    page_title="Predictive Keyboard",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #0e1117;
        color: white;
    }
    [data-testid=stSidebar] h1, [data-testid=stSidebar] h2, [data-testid=stSidebar] h3,
    [data-testid=stSidebar] h4, [data-testid=stSidebar] h5, [data-testid=stSidebar] h6 {
        color: white;
    }
    [data-testid=stSidebar] p, [data-testid=stSidebar] div, [data-testid=stSidebar] span {
        color: #dcdcdc;
    }
    [data-testid=stHeader] {
        background: #0e1117;
    }
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj:hover {
        background-color: #1a1d23;
        color: white;
    }
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .st-emotion-cache-1nz34qd {
        background-color: #1e2127;
        border: 1px solid #34373d;
    }
    .st-emotion-cache-1nz34qd:hover {
        border-color: #4a4d52;
    }
    .st-emotion-cache-1nz34qd:focus {
        border-color: #8b8c8c;
    }
    .st-ejykbx1 {
        color: white;
    }
    .st-bw {
        color: white;
    }
    .st-emotion-cache-16idsys p {
        color: white;
    }
    .css-1d391kg {
        background-color: #0e1117;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("⌨️ Predictive Keyboard Model")
st.markdown("Predict the next word based on your input using a trained LSTM model.")

# Load model and vocabulary
@st.cache_resource
def load_model_and_vocab():
    try:
        # Load the trained model
        model, vocab = load_model('predictive_keyboard_model.pth', 4500)
        return model, vocab
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vocab = load_model_and_vocab()

if model is None or vocab is None:
    st.stop()

# User input
st.header("Input Text")
user_input = st.text_input("Enter your text (e.g., 'the quick'):", value="the quick brown")

# Prediction parameters
st.header("Prediction Settings")
top_k = st.slider("Number of predictions to show", min_value=1, max_value=10, value=5, step=1)

# Predict button
if st.button("Get Predictions"):
    if user_input.strip():
        with st.spinner("Predicting next words..."):
            predictions = predict_next_words(model, vocab, user_input, top_k=top_k)
        
        st.header("Predicted Words")
        cols = st.columns(len(predictions))
        
        for i, (word, prob) in enumerate(predictions):
            with cols[i]:
                # Custom card-like display for dark mode
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1a1d23; 
                        padding: 15px; 
                        border-radius: 8px; 
                        border: 1px solid #34373d;
                        text-align: center;
                        height: 100%;
                    ">
                        <h4 style="color: #8b8c8c; margin-bottom: 10px;">Word {i+1}</h4>
                        <p style="font-size: 24px; color: white; font-weight: bold; margin: 10px 0;">{word}</p>
                        <p style="color: #a0a0a0; margin-top: 10px;">Confidence: {prob:.3f}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Display predictions in a table format
        st.subheader("Detailed Predictions")
        pred_data = {"Rank": [], "Word": [], "Confidence": []}
        for i, (word, prob) in enumerate(predictions):
            pred_data["Rank"].append(i+1)
            pred_data["Word"].append(word)
            pred_data["Confidence"].append(f"{prob:.3f}")
        
        # Create a more styled table
        table_html = "<div style='background-color: #1a1d23; padding: 15px; border-radius: 8px;'><table style='width:100%; border-collapse: collapse;'>"
        table_html += "<tr style='background-color: #2d3035; color: white;'><th style='padding: 10px; text-align: left; border: 1px solid #34373d;'>Rank</th><th style='padding: 10px; text-align: left; border: 1px solid #34373d;'>Word</th><th style='padding: 10px; text-align: left; border: 1px solid #34373d;'>Confidence</th></tr>"
        
        for i, (word, prob) in enumerate(predictions):
            bg_color = "#26272b" if i % 2 == 0 else "#1e1f23"
            table_html += f"<tr style='background-color: {bg_color}; color: white;'><td style='padding: 10px; border: 1px solid #34373d;'>{i+1}</td><td style='padding: 10px; border: 1px solid #34373d;'>{word}</td><td style='padding: 10px; border: 1px solid #34373d;'>{prob:.3f}</td></tr>"
        
        table_html += "</table></div>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text for prediction.")

st.markdown("---")
st.header("About This Model")
st.markdown("""
This predictive keyboard model uses an LSTM neural network trained on text data. 
The model predicts the next most likely words based on the input text using top-k sampling.
""")

st.sidebar.header("About")
st.sidebar.info("""
This is a predictive keyboard model using LSTM networks.
It can predict the next word based on input text.
""")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Enter text in the input field (e.g., "the quick brown")
2. Adjust the number of predictions to show
3. Click "Get Predictions" to see the forecasted words
""")