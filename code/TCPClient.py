# from socket import *
# serverName = input("input server IP:")
# str(serverName)
# serverPort = 12000
# clientSocket = socket(AF_INET,SOCK_STREAM)
# clientSocket.connect((serverName,serverPort))

# while True:
#     sentence = input("input sentence:")
#     clientSocket.send(sentence.encode())

#     modifiedSentence = clientSocket.recv(1024)
#     print("From Server: ",modifiedSentence.decode())

#     modifiedSentence = clientSocket.recv(1024)
#     print("From Server: ",modifiedSentence.decode())

#     flag = input("input continue of end")
#     clientSocket.send(flag.encode())
#     if flag != "continue":
#         break

# clientSocket.close()
from socket import *

serverName = input("Input server IP:")
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((serverName, serverPort))

while True:
    sentence = input("Input sentence (type 'quit' to close): ")
    clientSocket.send(sentence.encode())

    response = clientSocket.recv(1024).decode()
    print("From Server:", response)

    if sentence.lower() == "quit" or response.lower() == "quit":
        print("Closing connection.")
        break

clientSocket.close()
