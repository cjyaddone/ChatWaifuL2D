import socket
ip_port = ('127.0.0.1', 9000)
 
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
client.connect(ip_port)
while True:
    print("data to send")
    data = input().encode()
    client.send(data)
