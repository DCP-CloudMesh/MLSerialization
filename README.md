# ML Serialization

A Repository Exploring Serialization Techniques for ML Models

### Protobuf

The first technique for serialization that we attempted is Protobuf serialization.

We create some protobuf that stores the pickled weights and the name of the model.
In the future, this protobuf definition will likely become expanded to include more
model metadata for training information. The pickled file is the best way to compress
this data.
