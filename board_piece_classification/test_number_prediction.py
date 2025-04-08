from hexagon_prediction import predict_number
import pickle
import tensorflow as tf
from visualization.plot_tile_detector_results import plot_roc
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    auc,
)
import json
from keras import utils


def test_number_prediction(x_test, y_test, first_n, num_classes, label_encoder):

    y_test_np = y_test[:first_n]

    y_test_np = utils.to_categorical(y_test_np, num_classes=num_classes).numpy()
    y_pred = []

    for i in range(first_n):
        sample = x_test[i]
        sample_img = utils.array_to_img(sample.numpy())
        y_pred.append(predict_number(sample_img))

    y_pred_np = label_encoder.transform(y_pred)

    y_pred_np = utils.to_categorical(y_pred_np, num_classes=num_classes)

    acc = accuracy_score(y_test_np, y_pred_np)
    auc_score = roc_auc_score(y_test_np, y_pred_np, average="macro", multi_class="ovr")
    map_score = average_precision_score(y_test_np, y_pred_np, average="macro")

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(y_test_np.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_np[:, i], y_pred_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return acc, auc_score, map_score, fpr, tpr, roc_auc


if __name__ == "__main__":
    dataset_path = "data/output/compiled_dataset/synthetic_dataset_numbers.pkl"
    image_folder = "data/output/generated_synthetic_tiles_expanded"
    tesseract_result_save_path = "data/output/number_detector_test_results.txt"
    label_encoder_path = (
        "data/output/compiled_dataset/label_encoder/label_encoder_numbers.pkl"
    )
    roc_curve_save_path = "data/output/number_detector_roc_curve.png"
    no_samples_to_predict = 1000
    num_classes = 11

    print(f"Compiling dataset at path: {dataset_path}")

    with open(dataset_path, "rb") as f:
        (X, y) = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    acc, auc_score, map_score, fpr, tpr, roc_auc = test_number_prediction(
        X, y, no_samples_to_predict, num_classes, label_encoder
    )

    result_dict = {"test_acc": acc, "test_auc": auc_score, "test_map": map_score}

    with open(tesseract_result_save_path, "w") as file:
        json.dump(result_dict, file)

    plot_roc(11, fpr, tpr, roc_auc, label_encoder, roc_curve_save_path)
