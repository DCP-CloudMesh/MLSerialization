import zmq
import sys
import torch
from torch import nn
from models import SimpleNN, SimpleResNet
from serialization import serialize_model_with_pickle, deserialize_model_with_pickle


def main():
    context = zmq.Context()
    if len(sys.argv) != 2 or sys.argv[1] not in ["sender", "receiver"]:
        print("Usage: python zeromq_ipc.py [sender|receiver]")
        sys.exit(1)

    role = sys.argv[1]
    print(f"Running as {role}...")

    socket = None
    if role == "sender":
        socket = context.socket(zmq.REQ)
        socket.connect("ipc://localhost:5555")
    else:
        socket = context.socket(zmq.REP)
        socket.bind("ipc://localhost:5555")
    print("Socket created and bound successfully.")

    num_classes = 10
    model = SimpleResNet(num_classes)
    if role == "sender":
        model = model.to("mps")
        X_train = torch.rand((100, 3, 224, 224)).to("mps")
        y_train = torch.randint(0, num_classes, (100,)).to("mps")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        print("Training the model...")
        for _ in range(5):  # Train for 5 epochs
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        print("Training complete!")

        model = model.to("cpu")
        smodel = serialize_model_with_pickle(model, "Pickled PyTorch Model")
        socket.send(smodel)

        print("Model sent successfully!")
        message = socket.recv()
        print(f"Received reply: {len(message)} bytes")

    else:
        message = socket.recv()
        print(f"Received request: {len(message)} bytes")

        model = deserialize_model_with_pickle(model, serialized_model=message)
        print("Model weights loaded successfully.")

        socket.send(b"Recieved model")

    socket.close()

    for param in model.parameters():
        print(param)

    sample_input = torch.Tensor(size=(10, 3, 224, 224))
    out = model(sample_input)
    print("Model output:", out)

    return


if __name__ == "__main__":
    main()
