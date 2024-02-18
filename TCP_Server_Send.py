import socket
import sys
import SocketUtils
import os

c_path = os.getcwd()
os.chdir(c_path+'/data/server_space')
def initialize_server():
	host = '127.0.0.1'
	port = 8080
	totalclient = 2
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.bind((host, port))
	sock.listen(totalclient)
	# Establishing Connections
	connections = []
	print('Initiating clients')
	for i in range(totalclient):
		conn = sock.accept()
		connections.append(conn)
		print('Connected with client', i + 1)
	return connections
def server_send(filename):
	connections = initialize_server()
	SocketUtils.sending_to_nodes(connections, filename)

if __name__ == '__main__':
	filename = str(sys.argv[1])
	server_send(filename)