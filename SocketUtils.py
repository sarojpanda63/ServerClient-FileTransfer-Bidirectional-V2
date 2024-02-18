def receiving_from_nodes(connections):
    for conn in connections:
        while True:
            filename = conn[0].recv(1024).decode()
            if ".txt" in filename:
                conn[0].send("received".encode())
                print("Filename received: ", filename)
                break
        data = conn[0].recv(1024).decode()
        while not data:
            data = conn[0].recv(1024).decode()
        fo = open(filename, "w")
        while data:
            fo.write(data)
            print('Receiving file from client')
            data = conn[0].recv(1024).decode()
        fo.close()
        conn[0].close()
        print()
        print('Received successfully! New filename is:', filename)

def send_to_server(sock,filename):
    sock.send(str(filename).encode())
    data = sock.recv(1024).decode()
    while data!="received":
        data = sock.recv(1024).decode()
    print("Filename sent")

    while True:
        try:
            # Reading file and sending data to server
            fi = open(filename, "r")
            data = fi.read()
            if data:
                while data:
                    sock.send(str(data).encode())
                    data = fi.read()
            # File is closed after data is sent
            fi.close()
            break
        except IOError:
            print('invalid filename or filepath')
    sock.close()

def receiving_from_server(sock):
    filename = sock.recv(1024).decode()
    while True:
        if ".txt" in filename:
            sock.send("received".encode())
            break
    data = sock.recv(1024).decode()
    while not data:
        data = sock.recv(1024).decode()
    fo = open(filename, "w")
    while data:
        fo.write(data)
        print()
        print('Receiving file from client')
        data = sock.recv(1024).decode()
    fo.close()
    sock.close()
    print('Received successfully! New filename is:', filename)


def sending_to_nodes(connections, filename):
    for conn in connections:
        conn[0].send(str(filename).encode())
        data = conn[0].recv(1024).decode()
        while data != "received":
            data = conn[0].recv(1024).decode()

        while True:
            try:
                # Reading file and sending data to server
                fi = open(filename, "r")
                data = fi.read()
                if data:
                    while data:
                        conn[0].send(str(data).encode())
                        data = fi.read()
                # File is closed after data is sent
                fi.close()
                break
            except IOError:
                print('invalid filename or filepath')
        conn[0].close()