import sys
import socket
import torch
import torch.nn as nn
from simpleNN import SimpleNN
from serialization import serialize_model_with_pickle, deserialize_model_with_pickle

HOST = "127.0.0.1"  # Localhost
PORT = 65432  # Port to listen on


def run_receiver():
    """Run as the receiver process (server) to accept and deserialize model weights."""
    input_size = 3
    output_size = 2
    model = SimpleNN(input_size, output_size)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("Receiver is waiting for connection...")
        conn, addr = s.accept()
        with conn:
            print("Connected by", addr)
            data = b""
            while True:
                packet = conn.recv(4096)  # Read in chunks
                if not packet:
                    break
                data += packet

            # Deserialize the received model weights and load them into the model
            deserialize_model_with_pickle(model, serialized_model=data)

            # Print the model weights for verification
            for param in model.parameters():
                print(param)

    return model


def run_sender():
    """Run as the sender process (client) to serialize and send model weights."""
    input_size = 3
    output_size = 2
    model = SimpleNN(input_size, output_size)

    # Train the model (dummy training for demonstration)
    X_train = torch.rand((100, input_size))
    y_train = torch.rand((100, output_size))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(5):  # Train for 5 epochs
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    for param in model.parameters():
        print(param)

    # Serialize the model using pickle and Protobuf
    serialized_model = serialize_model_with_pickle(model, "Pickled PyTorch Model")

    # Send the serialized model over the socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(serialized_model)
        print("Model sent successfully!")

    return model


def main():
    if len(sys.argv) != 2:
        print("Usage: python ipc.py [receiver|sender]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "receiver":
        model = run_receiver()
    elif mode == "sender":
        model = run_sender()
    else:
        print("Unknown mode:", mode)
        sys.exit(1)

    sample_input = torch.Tensor([10, 20, 30])
    out = model(sample_input)
    print("Model output:", out)

    return


if __name__ == "__main__":
    main()
