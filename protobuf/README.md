# Protobuf Serialization Techniques for ML Models

### Generating Protobuf File

The following command should generate the model_pb2.py protobuf file.

```bash
protoc --python_out=. model.proto
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
