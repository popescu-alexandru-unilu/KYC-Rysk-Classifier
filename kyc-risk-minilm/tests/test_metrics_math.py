from src.eval_valid import confusion_matrix


def test_confusion_and_prf_simple():
    # 3 classes, tiny example
    y_true = [0,0,1,1,2,2]
    y_pred = [0,1,1,1,2,0]
    cm = confusion_matrix(y_true, y_pred, 3)
    # expected matrix rows=true, cols=pred
    # y=0 -> pred: 0,1
    # y=1 -> pred: 1,1
    # y=2 -> pred: 2,0
    assert cm == [
        [1,1,0],
        [0,2,0],
        [1,0,1]
    ]

