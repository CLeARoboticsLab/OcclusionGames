#!/usr/bin/env python3

import socket
import time

HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 12345      # Port to listen on

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)  # Receive up to 1024 bytes
            if not data:
                break
            print(f"Received: {data.decode('utf-8')}")
            # Optional: Send a response back
            rec_time = time.time()
            st = "Message received at time " + str(rec_time)
            conn.sendall(st.encode())
