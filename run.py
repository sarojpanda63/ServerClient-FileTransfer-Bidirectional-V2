
import subprocess
import time


def clients_to_server():
    process1 = subprocess.Popen(["python", "TCP_Server_Receive.py"])
    process2 = subprocess.Popen(["python", "TCP_Client1_Send.py","client1_to_server.txt"])
    process3 = subprocess.Popen(["python", "TCP_Client2_Send.py","client2_to_server.txt"])
    process1.wait()
    process2.wait()
    process3.wait()

def server_to_clients(filename):
    process1 = subprocess.Popen(["python", "TCP_Server_Send.py",filename])
    process2 = subprocess.Popen(["python", "TCP_Client1_Receive.py"])
    process3 = subprocess.Popen(["python", "TCP_Client2_Receive.py"])
    process1.wait()
    process2.wait()
    process3.wait()


if __name__ == '__main__':
    clients_to_server()
    time.sleep(10)
    server_to_clients("server_to_client.txt")