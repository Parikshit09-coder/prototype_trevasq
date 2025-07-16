import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn import preprocessing
import pennylane as qml


def define_qaum_model(depth=2):
    dev = qml.device("default.qubit", wires=1)


    def variational_circ(i, w):
        qml.RZ(w[i][0], wires=0)
        qml.RX(w[i][1], wires=0)
        qml.RY(w[i][2], wires=0)

    def quantum_neural_network(x, w, depth=depth):
        qml.Hadamard(wires=0)
        variational_circ(0, w)
        for i in range(0, depth):
            for j in range(8):
                qml.RZ(x[j], wires=0)
                variational_circ(j + 8 * i, w)

    @qml.qnode(dev, diff_method='backprop')
    def get_output(x, w):
        quantum_neural_network(x, w, depth)
        return qml.expval(qml.PauliZ(wires=0))

    def get_parity_prediction(x, w):
        np_measurements = (get_output(x, w) + 1.) / 2.
        return np.array([1. - np_measurements, np_measurements])

    def categorise(x, w):
        out = get_parity_prediction(x, w)
        return np.argmax(out)

    def accuracy(data, w):
        correct = 0
        for x, y in data:
            if int(categorise(x, w)) == int(y):
                correct += 1
        return correct / len(data) * 100

    return {
        'get_output': get_output,
        'get_parity_prediction': get_parity_prediction,
        'categorise': categorise,
        'accuracy': accuracy
    }


def define_qaoa_model(depth=3):
    dev = qml.device("default.qubit.autograd", wires=9)

    def quantum_neural_network(x, w, depth=depth):
        qml.templates.embeddings.QAOAEmbedding(features=x, weights=w, local_field='Y', wires=range(9))

    @qml.qnode(dev, diff_method='backprop')
    def get_output(x, w):
        quantum_neural_network(x, w, depth)
        return qml.expval(qml.PauliZ(wires=0))

    def get_parity_prediction(x, w):
        np_measurements = (get_output(x, w) + 1.) / 2.
        return np.array([1. - np_measurements, np_measurements])

    def categorise(x, w):
        out = get_parity_prediction(x, w)
        return np.argmax(out)

    def accuracy(data, w):
        correct = 0
        for x, y in data:
            if int(categorise(x, w)) == int(y):
                correct += 1
        return correct / len(data) * 100

    return {
        'get_output': get_output,
        'get_parity_prediction': get_parity_prediction,
        'categorise': categorise,
        'accuracy': accuracy
    }


def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def evaluate_model(test_df, model, model_type="QAUM"):
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X_test = min_max_scaler.fit_transform(X_test)

    weights = model['weights']
    depth = model['depth']

    if model_type == "QAUM":
        model_functions = define_qaum_model(depth)
    else:
        model_functions = define_qaoa_model(depth)

    categorise = model_functions['categorise']
    accuracy = model_functions['accuracy']

    test_data = list(zip(X_test, y_test))
    test_accuracy = accuracy(test_data, weights)

    y_pred = [categorise(x, weights) for x in X_test]

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)

    metrics = {
        'accuracy': test_accuracy / 100,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1
    }

    return test_accuracy, conf_matrix, class_report, y_pred, metrics, y_test