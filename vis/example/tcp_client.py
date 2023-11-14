import socket
import errno
import sys

from time import sleep

HOST, PORT = "localhost", 9998
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
for i in 'abcde':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Connect to server and send data
        sock.connect((HOST, PORT))

        msg = data + "-" + i
        sock.sendall(bytes(msg + "\n", "utf-8"))
        # Receive data from the server and shut down
        received = str(sock.recv(1024), "utf-8")

        print("Sent:     {}".format(msg))
        print("Received: {}".format(received))


