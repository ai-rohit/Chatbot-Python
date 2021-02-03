import random
import json
import pyttsx3
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from colorama import Fore, Back, Style


users = [{"accountNum":"A", "name":"Rohit Shrestha", "amount": "Rs 30000"}, {"accountNum":"B", "name":"Rohit Shrestha", "amount": "Rs 35000"}]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
engine = pyttsx3.init()
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot = "Your Banking Bot"
print("|-----------------------------------------------------------------------------------------------|")
print("|               Welcome! Your queries will be answered here by your Banking Bot.                |")
print("|-----------------------------------------------------------------------------------------------|")

while True:
    sentence = input(Fore.RED+"|\tYou=> ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "user":
                    print(f"|\t{bot}: {random.choice(intent['responses'])}")
                    while True:
                        account = input("You: ")
                        if account == "quit":
                            print(f"|\t{bot}: Now ask other queries if you have any sir!")
                            break

                        for user in users:

                            if user["accountNum"]==account:
                                print(f"|\t{bot}: Account Number: {user['accountNum']} \n Name: {user['name']} \n Amount: {user['amount']}")
                                print("\n Type 'quit' to start a conversation with me again or Type your "
                                      "account number again to know details")
                                break
                else:
                    print(Fore.BLUE+f"|\t{bot}: {random.choice(intent['responses'])}")
                #engine.say(f"{random.choice(intent['responses'])}")
                #engine.runAndWait();
    else:
        print(f"{bot}: I do not understand...")