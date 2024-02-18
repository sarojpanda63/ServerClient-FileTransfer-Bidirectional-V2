import socket
import SocketUtils
import os

c_path = os.getcwd()
os.chdir(c_path+'/data/client1_space')

def initialize_client1():
	host = '127.0.0.1'
	port = 8080
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	while True:
		try:
			sock.connect((host, port))
			s = str(sock)
			if "laddr = ('0.0.0.0'" not in s:
				break
		except ConnectionRefusedError as e:
			continue
	return sock

if __name__ == '__main__':
	sock = initialize_client1()
	SocketUtils.receiving_from_server(sock)
