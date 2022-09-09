from chatbot import Chatbot

if __name__ == '__main__':
    chatbot = Chatbot("./intents.json")
    chatbot.train()
    chatbot.save_model("./data.pth")