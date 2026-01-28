import json
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
import numpy as np

# Make sure we have the necessary NLTK data
# If it crashes here, you might need to run nltk.download() manually
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK files...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Config
INTENTS_FILE = 'intents.json'
MODEL_FILE = 'model_data.pth'

class NeuralNet(nn.Module):
    """
    A simple 3-layer neural network.
    Input -> Hidden (128) -> Hidden (64) -> Output
    """
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        # Dropout helps prevent overfitting on small datasets
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        return out

class ChatBot:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.model = None
        self.all_words = []
        self.tags = []
        
    def tokenize(self, sentence):
        # Split sentence into array of words/tokens
        return nltk.word_tokenize(sentence)

    def stem(self, word):
        return self.lemmatizer.lemmatize(word.lower())

    def bag_of_words(self, tokenized_sentence):
        # Create the input vector (1s and 0s)
        sentence_words = [self.stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(self.all_words), dtype=np.float32)
        
        for idx, w in enumerate(self.all_words):
            if w in sentence_words:
                bag[idx] = 1.0
        return bag

    def train(self):
        print("Loading intents...")
        with open(INTENTS_FILE, 'r') as f:
            intents = json.load(f)

        xy = [] # Holds patterns and tags
        self.all_words = []
        self.tags = []

        # Loop through each sentence in our intents patterns
        for intent in intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            for pattern in intent['patterns']:
                w = self.tokenize(pattern)
                self.all_words.extend(w)
                xy.append((w, tag))

        # Stem and lower each word
        self.all_words = [self.stem(w) for w in self.all_words if w not in ['?', '.', '!']]
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))

        # Create training data
        x_train = []
        y_train = []

        for (pattern_sentence, tag) in xy:
            bag = self.bag_of_words(pattern_sentence)
            label = self.tags.index(tag)
            x_train.append(bag)
            y_train.append(label)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # PyTorch Dataset
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long())
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Model Setup
        input_size = len(x_train[0])
        output_size = len(self.tags)
        self.model = NeuralNet(input_size, output_size)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print("Training model...")
        for epoch in range(1000):
            for (words, labels) in loader:
                outputs = self.model(words)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

        # Save everything needed to run later
        data = {
            "model_state": self.model.state_dict(),
            "input_size": input_size,
            "output_size": output_size,
            "all_words": self.all_words,
            "tags": self.tags
        }
        torch.save(data, MODEL_FILE)
        print(f"Training complete. File saved to {MODEL_FILE}")

    def load(self):
        if not os.path.exists(MODEL_FILE):
            return False

        print("Loading saved model...")
        data = torch.load(MODEL_FILE)
        
        self.input_size = data["input_size"]
        self.output_size = data["output_size"]
        self.all_words = data["all_words"]
        self.tags = data["tags"]
        
        self.model = NeuralNet(self.input_size, self.output_size)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        return True

    def get_response(self, msg):
        sentence = self.tokenize(msg)
        X = self.bag_of_words(sentence)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        # Check confidence
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() < 0.75:
            return "I do not understand..."

        # Handle "Special" functions (like stocks)
        if tag == "stocks":
            return self.check_stocks()

        # Otherwise just return a random response from JSON
        with open(INTENTS_FILE, 'r') as f:
            intents = json.load(f)
            
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    
    def check_stocks(self):
        # Fake function to simulate checking an API
        my_portfolio = ['AAPL', 'MSFT', 'TSLA']
        return f"Fetching data... You own: {', '.join(my_portfolio)}"

if __name__ == "__main__":
    bot = ChatBot()
    
    # Check if we need to train
    if not bot.load():
        print("No model found. Training now...")
        bot.train()
        
    print("\nLet's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        
        resp = bot.get_response(sentence)
        print(f"Bot: {resp}")