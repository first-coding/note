from socket import *
serverPort = 12000
serverSocket = socket(AF_INET,SOCK_DGRAM)
serverSocket.bind(('', serverPort)) #bind用于端口号与服务器的套接字绑定
print("The server is ready to receive")
while True:
    message,clientAddress = serverSocket.recvfrom(2048)
    print(message,clientAddress)
    send_client = input("please input send client message")
    str(send_client)
    serverSocket.sendto(send_client.encode(),clientAddress)