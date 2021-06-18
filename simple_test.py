from sklearn.svm import SVC
from sklearn.metrics import classification_report

from load_partitions import load_partitions
from constants import dataset_names

data_idx = -1
part = 1
mask_value = 0
scale = True

dataset_name = dataset_names[data_idx]
train_x, train_y, _, _, test_x, test_y, _, _ = load_partitions(
    dataset_name, part, mask_value, scale)
model = SVC()
model.fit(train_x, train_y)
pred = model.predict(test_x)
print(classification_report(test_y, pred))
