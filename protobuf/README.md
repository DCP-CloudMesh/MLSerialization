# Protobuf Serialization Techniques for ML Models

### Generating Protobuf File

The following command should generate the model_pb2.py protobuf file.

```bash
protoc --python_out=. model.proto
```

### Testing Protobuf

```bash
python main.py
```
