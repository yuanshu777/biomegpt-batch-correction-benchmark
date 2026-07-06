from src.evaluation.metrics import balanced_accuracy, binary_auroc, macro_f1


def test_basic_classification_metrics():
    y_true = ["CRC", "CRC", "control", "control"]
    y_pred = ["CRC", "control", "control", "control"]
    assert balanced_accuracy(y_true, y_pred) == 0.75
    assert round(macro_f1(y_true, y_pred), 3) == 0.733
    assert binary_auroc(y_true, [0.9, 0.7, 0.2, 0.1], positive_label="CRC") == 1.0

