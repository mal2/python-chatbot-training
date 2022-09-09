# things we need for NLP
import nltk
from nltk.stem.snowball import GermanStemmer

import numpy as np
import json
import torch
import torch.nn as nn
import unittest

from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        self.out = self.l1(x)
        self.out = self.relu(self.out)
        self.out = self.l2(self.out)
        self.out = self.relu(self.out)
        self.out = self.l3(self.out)
        # no activation and no softmax at the end
        return self.out

class ChatDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class Chatbot:
    def __init__(self, path_to_intents: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stemmer = GermanStemmer()
        self.path_to_intents = path_to_intents
        with open(self.path_to_intents, encoding='utf-8') as json_data:
            self.intents = json.load(json_data)
        self.generate_train_data()
    
    def tokenize(self, sentence):
        """
        split sentence into array of words/tokens
        a token can be a word or punctuation character, or number
        """
        return nltk.word_tokenize(sentence, language='german')


    def stem(self, word):
        """
        stemming = find the root form of the word
        examples:
        words = ["organize", "organizes", "organizing"]
        words = [stem(w) for w in words]
        -> ["organ", "organ", "organ"]
        """
        return self.stemmer.stem(word.lower())


    def bag_of_words(self, tokenized_sentence, words):
        """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
        # stem each word
        self.sentence_words = [self.stem(word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        self.bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in self.sentence_words: 
                self.bag[idx] = 1

        return self.bag

    def generate_train_data(self):
        print(f'Generated train data from {self.path_to_intents}')
        self.all_words = []
        self.tags = []
        self.actions = []
        self.xy = []

        # loop through each sentence in our intents patterns
        for intent in self.intents['intents']:
            self.tag = intent['tag']
            try:
                self.actions.append(intent['action'])
            except:
                pass
            # add to tag list
            self.tags.append(self.tag)
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                self.w = self.tokenize(pattern)
                # add to our words list
                self.all_words.extend(self.w)
                # add to xy pair
                self.xy.append((self.w, self.tag))

        # stem and lower each word
        self.ignore_words = ['?', '.', '!']
        self.all_words = [self.stem(w) for w in self.all_words if w not in self.ignore_words]
        # remove duplicates and sort
        self.all_words = sorted(set(self.all_words))
        self.tags = sorted(set(self.tags))
        self.actions = list(self.actions)

        # create training data
        self.X_train = []
        self.y_train = []
        for (pattern_sentence, tag) in self.xy:
            # X: bag of words for each pattern_sentence
            self.bag = self.bag_of_words(pattern_sentence, self.all_words)
            self.X_train.append(self.bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            self.label = self.tags.index(self.tag)
            self.y_train.append(self.label)

        self.X_train = torch.from_numpy(np.array(self.X_train))
        self.y_train = torch.from_numpy(np.array(self.y_train)).type(torch.LongTensor)
    
    def get_train_data():
        return self.X_train, self.y_train
    
    def train(self, num_epochs = 500, batch_size = 16, learning_rate = 0.001, hidden_size = 512):
        # Hyper-parameters 
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.input_size = len(self.X_train[0])
        self.output_size = len(self.tags)

        self.dataset = ChatDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=2)

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("Start Training")
        # Train the model
        for epoch in range(self.num_epochs):
            for (words, labels) in self.train_loader:
                self.words = words.to(self.device)
                self.labels = labels.to(self.device)
                
                # Forward pass
                self.outputs = self.model(self.words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                self.loss = self.criterion(self.outputs, self.labels)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
            if (epoch+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {self.loss.item():.6f}')
        print(f'Training complete. Final loss: {self.loss.item():.6f}')

    def save_model(self, model_save_path: str):
        self.data = {
            #"model": model,
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "all_words": self.all_words,
            "tags": self.tags,
            "action": self.actions,
        }

        torch.save(self.data, model_save_path)
        print(f'File saved to {model_save_path}')

class ChatbotTestCase(unittest.TestCase):

    def setUp(self):
        self.chatbot = Chatbot("./intents.json")

    def test_trickster_greetings(self):
        self.assertTrue(self.trickster.greeting() in self.trickster.intents['intents'][0]['responses'])
        self.assertFalse(self.trickster.greeting() in self.trickster.intents['intents'][1]['responses'])

    def test_trickster_response(self):
        self.assertTrue(self.trickster.response("Hey") in self.trickster.intents['intents'][0]['responses'])
        self.assertTrue(self.trickster.response("Tschüss") in self.trickster.intents['intents'][1]['responses'])
        self.assertTrue(self.trickster.response("Wie lange ist heute geöffnet") in self.trickster.intents['intents'][4]['responses'])
        self.assertIn(self.trickster.response("quit"), "Danke, bis zum nächsten mal.")
        self.assertIn(self.trickster.response("foo bar"), "Das habe ich nicht verststanden...")


if __name__ == '__main__':
    unittest.main()