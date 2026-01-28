PyTorch Chatbot

I built this simple chatbot to practice Deep Learning and NLP. Instead of just matching keywords, it uses a simple Feed-Forward Neural Network to guess the "intent" of your sentence.

How it works:
- It reads patterns from 'intents.json'.
- Converts text to a bag-of-words vector.
- Predicts the category using the trained model in 'main.py'.

How to run:
1. Install dependencies: pip install torch nltk numpy
2. Run the script: python main.py

It will automatically train the model on the first run (takes a few seconds) and save it. If you want to change what the bot says, just edit the JSON file.
