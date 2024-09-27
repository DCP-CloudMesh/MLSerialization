import pickle
import model_pb2


def serialize_model_with_pickle(model, model_name, filename=None):
    # Step 1: Pickle the model's state_dict (weights)
    pickled_weights = pickle.dumps(model.state_dict())

    # Step 2: Create a Model protobuf message
    model_pb = model_pb2.Model()
    model_pb.name = model_name
    model_pb.pickled_weights = pickled_weights

    # Step 3: Serialize the protobuf message to a binary file
    serialized_model = model_pb.SerializeToString()
    if filename:
        with open(filename, "wb") as f:
            f.write(serialized_model)

    return serialized_model


def deserialize_model_with_pickle(model, serialized_model=None, filename=None):
    # Step 1: Read the serialized protobuf data from the file
    if (not serialized_model and not filename) or (serialized_model and filename):
        raise ValueError("Either serialized_model or filename must be provided.")

    if filename:
        with open(filename, "rb") as f:
            serialized_model = f.read()

    # Step 2: Parse the protobuf message
    model_pb = model_pb2.Model()
    model_pb.ParseFromString(serialized_model)

    print(f"Model Name: {model_pb.name}")

    # Step 3: Unpickle the weights and load them into the PyTorch model
    state_dict = pickle.loads(model_pb.pickled_weights)
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")

    return model
