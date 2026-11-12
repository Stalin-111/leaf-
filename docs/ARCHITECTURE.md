# DeepLeaf AI Architecture

User -> Frontend -> Flask REST API -> preprocessing -> CNN -> prediction -> SQLite history.

Training pipeline: raw class folders -> reproducible split -> TensorFlow input pipeline -> augmentation -> CNN -> checkpoint -> evaluation reports.

The inference API loads the trained Keras model and class map. The frontend never contains model logic; it communicates through `/api/predict` and `/api/history`.
