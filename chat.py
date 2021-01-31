import random
import json
import pyttsx3
import torch
from tkinter import *
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

users = [{"accountNum":"A", "name":"Rohit Shrestha", "amount": "Rs 30000"}, {"accountNum":"B", "name":"Rohit Shrestha", "amount": "Rs 35000"}]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('queries.json', 'r') as json_data:
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


bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

#root = Tk()
#txt=Text(root)
#txt.grid(row=0, column=0, columnspan=2)
#txt.insert(END,"\n"+"Your Banking Chatbot: Welcome, I am here to help You")
#e=Entry(root, width=100).grid(row=1,column=0)
#send=Button(root,text="Send").grid(row=1,column=1)
#root.mainloop()

while True:
    #sentence = ""
    #reply = ""
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
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
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    while True:
                        account = input("You: ")
                        if account == "quit":
                            break

                        for user in users:
                            if user["accountNum"]==account:
                                print(f"{bot_name}: Account Number: {user['accountNum']} \n Name: {user['name']} \n Amount: {user['amount']}")
                                break
                else:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                #engine.say(f"{random.choice(intent['responses'])}")
                #engine.runAndWait();
    else:
        print(f"{bot_name}: I do not understand...")