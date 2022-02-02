from pathlib import Path
import pickle

from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback


class EvaluateCallback(Callback):
    def __init__(self, val_data, out_dir, batch_size):
        super(EvaluateCallback, self).__init__()
        self.val_data = val_data
        self.batch_size = batch_size
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
        self.datasets = ('train', 'test', 'val')

    def _perform_evaluation(self):
        result = {}
        for i in range(len(self.val_data)):
            data_x, data_y = self.val_data[i]
            data_y = data_y.argmax(axis=1)
            name = self.datasets[i]
            preds = self.model.predict(
                data_x, batch_size=self.batch_size
            ).argmax(axis=1)
            result[name] = classification_report(data_y, preds,
                                                 output_dict=True)
        return result

    def _save_results(self):
        with open(self.out_dir / 'callback_results.pickle', 'wb') as f:
            pickle.dump(self.results, f)

    def on_train_begin(self, logs=None):
        assert len(self.results) == 0, 'There should be no previous results'
        result = self._perform_evaluation()
        self.results.append(result)
        self._save_results()

    def on_epoch_end(self, epoch, logs=None):
        assert len(self.results) >= 1, 'There should be at least ' \
                                       '1 previous result'
        result = self._perform_evaluation()
        self.results.append(result)
        self._save_results()
