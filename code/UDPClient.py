from socket import *
ip = '192.168.0.102'
port = 12000
clientSocket = socket(AF_INET,SOCK_DGRAM) #AF_INET使用IPV4，SOCK_DFRAM表示是一个UDP套接字
message = input('input message') 
clientSocket.sendto(message.encode(),(ip,port))
modifiedMessage,serverAddress = clientSocket.recvfrom(2048)
print(modifiedMessage.decode())
clientSocket.close()