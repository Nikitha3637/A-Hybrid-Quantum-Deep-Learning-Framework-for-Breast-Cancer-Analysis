import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
dataset_path = r"C:\Users\nikit\Downloads\Dataset"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.15,
    brightness_range=[0.8,1.2],
    rotation_range=5
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)
val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1]),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(4, activation="tanh")(x)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
def quantum_conv(params):
    qml.CNOT(wires=[0,1])
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)

    qml.CNOT(wires=[2,3])
    qml.RY(params[2], wires=2)
    qml.RY(params[3], wires=3)
def quantum_pool(params):
    qml.CNOT(wires=[1,0])
    qml.RZ(params[0], wires=0)

    qml.CNOT(wires=[3,2])
    qml.RZ(params[1], wires=2)
@qml.qnode(dev, interface="tf")
def qcnn(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    quantum_conv(weights[0])
    quantum_pool(weights[1])
    quantum_conv(weights[2])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
weight_shapes = {"weights": (3,4)}
quantum_layer = qml.qnn.KerasLayer(qcnn, weight_shapes, output_dim=4)
q_out = quantum_layer(x)
output = Dense(1, activation="sigmoid")(q_out)
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3
    )
]
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)
loss, acc = model.evaluate(val_data)
acc_percent = acc * 100
print("\nValidation Accuracy: %.2f%%" % acc_percent)
val_data.reset()
y_true = val_data.classes
y_pred_prob = model.predict(val_data)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", auc)
cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1],["Benign","Malignant"])
plt.yticks([0,1],["Benign","Malignant"])
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center")
plt.show()
plt.figure()
plt.plot(history.history['accuracy'],label="Train Accuracy")
plt.plot(history.history['val_accuracy'],label="Validation Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure()
plt.plot(history.history['loss'],label="Train Loss")
plt.plot(history.history['val_loss'],label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
fpr, tpr, _ = roc_curve(y_true,y_pred_prob)
plt.figure()
plt.plot(fpr,tpr,label="ROC (AUC=%.3f)"%auc)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()