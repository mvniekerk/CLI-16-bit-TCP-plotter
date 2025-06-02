import matplotlib.pyplot as plt
import numpy as np
import socket
import struct

import threading

lock = threading.Lock()

#
# const SAMPLE_SIZE: usize = 128;
#// const FFT_OUTPUT: usize = SAMPLE_SIZE / 2;
#// const FFT_OUTPUT: usize = SAMPLE_SIZE / 2;
#// const FFT_OUTPUT: usize = 7;
# const FFT_OUTPUT: usize = SAMPLE_SIZE /2;

SAMPLE_SIZE = 128
FFT_OUTPUT = SAMPLE_SIZE // 2

y = [0.0 for i in range(FFT_OUTPUT)]
x = [*range(1,FFT_OUTPUT + 1)]

plt.ion()  # turning interactive mode on

# plotting the first frame
graph = plt.plot(x,y)[0]
plt.ylim(0,10)
plt.pause(1)

def receive_float32_values(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        values = []
        s.connect((host, port))
        data = s.recv(FFT_OUTPUT)  # Adjust buffer size as per your requirement
        index = 0
        while index + 4 <= len(data):
            value = struct.unpack('<f', data[index:index+4])[0]
            values.append(value)
            index += 4

        y = np.array(values, dtype=np.float32)
        # removing the older graph
#         graph.remove()

        # plotting newer graph
        graph = plt.plot(x,y,color = 'g')[0]
#         plt.plot(frame)
#         plt.xlabel('Index')
#         plt.ylabel('Float32 Value')
#         plt.title('Visualization of Float32 Values from TCP Connection')
#         plt.grid(True)

#         fig, ax = plt.subplots()
#         plt.show()
        while True:
            # Receive data from the TCP connection
            data = s.recv(FFT_OUTPUT)  # Adjust buffer size as per your requirement

            # Assuming each value is a little-endian float32
            # Unpack binary data to float32 values
            values.clear()
            index = 0
            while index + 4 <= len(data):
                value = struct.unpack('<f', data[index:index+4])[0]
                values.append(value)
                index += 4

            y = np.array(values, dtype=np.float32)
            # removing the older graph
            graph.remove()

            # plotting newer graph
            graph = plt.plot(x,y,color = 'g')[0]
#             frame = np.array(values, dtype=np.float32)
#             line.set_ydata(frame)
#             ax.relim()
#             ax.autoscale/_view(True, True, True)
#             plt.plot(frame)
#             plt.draw()
            print("Received data: ", frame)


if __name__ == "__main__":
    # Define the TCP server's host and port
#     host = '192.168.0.106'  # Change to the server's IP address if not local
    host = '192.168.0.106'  # Change to the server's IP address if not local
    port = 1234  # Change to the server's port number

    receive_float32_values(host, port)

