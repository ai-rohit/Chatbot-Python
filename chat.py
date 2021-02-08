import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from colorama import Fore

# user details stored in list nested with dictionary
users = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading the data set present in json file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# storing data set in torch file in torch_file
torch_file = "data.pth"
data = torch.load(torch_file)

# providing hyper parameters for neural net
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# assigning name for bot and making UI screen presentable in console
bot = "Your Banking Bot"
print(
    "|-----------------------------------------------------------------------------------------------------------------------------|")
print(
    "|                            Welcome! Your queries will be answered here by your Banking Bot.                                 |")
print(
    "|-----------------------------------------------------------------------------------------------------------------------------|")
print(Fore.MAGENTA + "|  Ask any queries related to bank. Type 'quit' to stop the conversation")

while True:
    sentence = input(Fore.RED + "|\tYou=> ")
    if sentence == "quit":
        # on quit
        print(Fore.BLUE + f"|\t{bot}=> Thank you for interacting with me. Have a good day!")
        break

    # pre processing the input sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # predicting tag using the torch data set
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # getting output/responses according to predicted intent of the input sentence
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "user":
                    print(Fore.BLUE + f"|\t{bot}=> {random.choice(intent['responses'])}")
                    while True:
                        accountNum = []
                        account = input(Fore.RED + "|\tYou=> ")
                        if account == "quit":
                            print(Fore.BLUE + f"|\t{bot}=> Now ask other queries if you have any sir!")
                            break

                        for user in users:
                            accountNum.append(user["accountNum"])

                        for user in users:
                            if account not in accountNum:
                                print(Fore.BLUE + f"|\t{bot}=> Hmm, I can't find your account number. You can enter account number again correctly or enter quit for other queries.")
                                break

                            if user["accountNum"] == account:
                                print(
                                    Fore.BLUE + f"|\t{bot}=> Account Number: {user['accountNum']} \n|\tName: {user['name']} \n|\tAmount: {user['amount']}\n")

                                print(
                                    "|----------------------------------------------------------------------------------------------------------------------|")
                                print("|\t\t\tType 'quit' to start a conversation with me again or Type your "
                                      "account number again to know details\t\t   |")
                                print(
                                    "|----------------------------------------------------------------------------------------------------------------------|")
                                break
                else:
                    print(Fore.BLUE + f"|\t{bot}=> {random.choice(intent['responses'])}")
    else:
        print(Fore.BLUE + f"|\t{bot}=> Sorry sir, I can't answer your query!")
