import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_batch(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    data = data_dict[b'data']
    labels = data_dict[b'labels']

    return data, labels

X_train, y_train = load_batch("cifar-10-python/cifar-10-batches-py/data_batch_1")
X_test, y_test = load_batch("cifar-10-python/cifar-10-batches-py/test_batch")

print("Train veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

metric = input("Mesafe türü seç (L1 / L2): ")
k = int(input("k değerini gir: "))


test_index = 0
test_image = X_test[test_index]
true_label = y_test[test_index]

distances = []

for i in range(len(X_train)):

    train_image = X_train[i]

    l1_dist = np.sum(np.abs(train_image - test_image))
    l2_dist = np.sqrt(np.sum((train_image - test_image) ** 2))

    if i < 5:
        print(f"Görüntü {i} -> L1: {l1_dist:.2f} | L2: {l2_dist:.2f}")

    if metric == "L1":
        dist = l1_dist
    else:
        dist = l2_dist

    distances.append((dist, y_train[i]))


distances.sort(key=lambda x: x[0])
neighbors = distances[:k]


labels = []

for d, label in neighbors:
    labels.append(label)

prediction = max(set(labels), key=labels.count)


classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]


print("\nTest görüntüsü index:", test_index)
print("Tahmin edilen sınıf:", classes[prediction])
print("Gerçek sınıf:", classes[true_label])

img = test_image.reshape(3, 32, 32)
img = img.transpose(1, 2, 0)

plt.figure(figsize=(4,4))
plt.imshow(img)
plt.title(f"Tahmin: {classes[prediction]} | Gerçek: {classes[true_label]}")
plt.axis("off")
plt.show()
