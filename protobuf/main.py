import torch
import torch.nn as nn
import pickle
import model_pb2
from serialization import serialize_model_with_pickle, deserialize_model_with_pickle
from simpleNN import SimpleNN


def main():
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

    # Serialize the model using pickle and Protobuf
    serialize_model_with_pickle(model, "Pickled PyTorch Model", "bin/pickled_model.bin")

    # Create a new instance of the model
    new_model = SimpleNN(input_size, output_size)

    # Deserialize the model and load the weights
    deserialize_model_with_pickle(model=new_model, filename="bin/pickled_model.bin")

    # Compare the model's weights
    for param1, param2 in zip(model.parameters(), new_model.parameters()):
        print(param1, param2)
        assert torch.allclose(param1, param2), "Models have different weights!"

    print("Models have the same weights!")
    return


if __name__ == "__main__":
    main()
