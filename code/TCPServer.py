# from socket import *
# serverPort = 12000
# serverSocket = socket(AF_INET,SOCK_STREAM)
# serverSocket.bind(('',serverPort))
# serverSocket.listen(1)
# print("The server is ready to receive")
# connectionSocket,addr = serverSocket.accept()

# while True:
#     sentence = connectionSocket.recv(1024).decode()
#     print("client message:",sentence)

#     message = input("input send client message:")
#     connectionSocket.send(message.encode())

#     flag_str = "continue of end"
#     connectionSocket.send(flag_str.encode())
#     flag = connectionSocket.recv(1024).decode()
#     print(flag)

#     if flag != "continue":
#         break
    
# connectionSocket.close()
from socket import *

serverPort = 12000
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)
print("The server is ready to receive")
connectionSocket, addr = serverSocket.accept()

while True:
    sentence = connectionSocket.recv(1024).decode()
    print("Client message:", sentence)

    if sentence.lower() == "quit":
        print("Received 'quit' from client. Closing connection.")
        connectionSocket.send("quit".encode())
        break

    message = input("Input send client message: ")
    connectionSocket.send(message.encode())

    response = connectionSocket.recv(1024).decode()
    print("From Client:", response)

    if response.lower() == "quit":
        print("Received 'quit' from client. Closing connection.")
        connectionSocket.send("quit".encode())
        break

connectionSocket.close()
serverSocket.close()
