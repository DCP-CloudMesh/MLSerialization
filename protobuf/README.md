# Protobuf Serialization Techniques for ML Models

### Generating Protobuf File

The following command should generate the model_pb2.py protobuf file.

```bash
protoc --python_out=. model.proto
```

We have some demo serialized protobuf input put into a bin file to check how large
it can get. Currently the simple NN model is about 4Kb in size. The model is simply

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        return out
```

### Testing Protobuf Serialization

```bash
python main.py
```

### Testing Protobuf IPC

```bash
python ipc.py receiver
python ipc.py sender
```
