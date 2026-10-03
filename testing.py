import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("models/trash_classifier.keras")

#shuffle off so the batch order (and the graphs) are the same every run
test_ds = tf.keras.utils.image_dataset_from_directory(r"C:\Users\anshi\Desktop\garbage sorting\data\Raw Data\Testing Dataset", image_size=(224, 224), batch_size = 32, shuffle=False)
class_names = test_ds.class_names

test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

x, y = model.evaluate(test_ds) #evalutates function
print ("Test Loss: ", x)
print ("Test Accuracy: ", y)


#goes through the test set again and records loss and accuracy for every image
sample_losses = []
sample_correct = []
sample_labels = []
sample_preds = []
batch_sizes = []
for images, labels in test_ds:
    preds = model(images, training=False)
    pred_classes = np.argmax(preds, axis=1)
    sample_losses.append(tf.keras.losses.sparse_categorical_crossentropy(labels, preds).numpy())
    sample_correct.append((pred_classes == labels.numpy()).astype(float))
    sample_labels.append(labels.numpy())
    sample_preds.append(pred_classes)
    batch_sizes.append(len(labels))

batch_loss = np.array([l.mean() for l in sample_losses])
batch_acc = np.array([c.mean() for c in sample_correct])
sample_losses = np.concatenate(sample_losses)
sample_correct = np.concatenate(sample_correct)
sample_labels = np.concatenate(sample_labels)
sample_preds = np.concatenate(sample_preds)

#running average weighted by batch size, last value equals the overall result
batch_sizes = np.array(batch_sizes)
running_loss = np.cumsum(batch_loss * batch_sizes) / np.cumsum(batch_sizes)
running_acc = np.cumsum(batch_acc * batch_sizes) / np.cumsum(batch_sizes)

#loss and accuracy for each class
class_counts = np.array([np.sum(sample_labels == i) for i in range(len(class_names))])
class_acc = np.array([sample_correct[sample_labels == i].mean() if class_counts[i] else 0 for i in range(len(class_names))])
class_loss = np.array([sample_losses[sample_labels == i].mean() if class_counts[i] else 0 for i in range(len(class_names))])

overall_loss = sample_losses.mean()
overall_acc = sample_correct.mean()


#graph 1: loss and accuracy across the test batches
batches = np.arange(1, len(batch_loss) + 1)
fig1, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

ax_loss.plot(batches, batch_loss, color="tab:red", alpha=0.35, marker="o", markersize=3, label="Per batch")
ax_loss.plot(batches, running_loss, color="tab:red", linewidth=2, label="Running average")
ax_loss.set_title(f"Test Loss by Batch (overall {overall_loss:.4f})")
ax_loss.set_xlabel("Batch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(alpha=0.3)
ax_loss.legend()

ax_acc.plot(batches, batch_acc, color="tab:blue", alpha=0.35, marker="o", markersize=3, label="Per batch")
ax_acc.plot(batches, running_acc, color="tab:blue", linewidth=2, label="Running average")
ax_acc.set_title(f"Test Accuracy by Batch (overall {overall_acc:.2%})")
ax_acc.set_xlabel("Batch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1.05)
ax_acc.grid(alpha=0.3)
ax_acc.legend()
fig1.tight_layout()


#graph 2: accuracy and loss for each class
fig2, (ax_cacc, ax_closs) = plt.subplots(1, 2, figsize=(13, 5))

bars = ax_cacc.bar(class_names, class_acc, color="tab:blue")
ax_cacc.axhline(overall_acc, color="black", linestyle="--", linewidth=1, label=f"Overall {overall_acc:.2%}")
for bar, acc, n in zip(bars, class_acc, class_counts):
    ax_cacc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{acc:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9)
ax_cacc.set_title("Test Accuracy by Class")
ax_cacc.set_ylabel("Accuracy")
ax_cacc.set_ylim(0, 1.25)
ax_cacc.legend(loc="upper center")

bars = ax_closs.bar(class_names, class_loss, color="tab:red")
ax_closs.axhline(overall_loss, color="black", linestyle="--", linewidth=1, label=f"Overall {overall_loss:.4f}")
for bar, loss in zip(bars, class_loss):
    ax_closs.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.3f}", ha="center", va="bottom", fontsize=9)
ax_closs.set_title("Test Loss by Class")
ax_closs.set_ylabel("Loss")
ax_closs.set_ylim(0, max(class_loss.max(), overall_loss) * 1.2 + 1e-6)
ax_closs.legend(loc="upper right")
fig2.tight_layout()


#graph 3: confusion matrix, rows are the real class and columns are what the model guessed
num_classes = len(class_names)
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(sample_labels, sample_preds):
    conf_matrix[true, pred] += 1
row_totals = conf_matrix.sum(axis=1, keepdims=True)
conf_percent = np.divide(conf_matrix, row_totals, out=np.zeros(conf_matrix.shape), where=row_totals > 0)

fig3, ax_cm = plt.subplots(figsize=(7, 6))
im = ax_cm.imshow(conf_percent, cmap="Blues", vmin=0, vmax=1)
fig3.colorbar(im, ax=ax_cm, label="Share of true class")
for i in range(num_classes):
    for j in range(num_classes):
        text_color = "white" if conf_percent[i, j] > 0.5 else "black"
        ax_cm.text(j, i, f"{conf_matrix[i, j]}\n{conf_percent[i, j]:.1%}", ha="center", va="center", color=text_color, fontsize=10)
ax_cm.set_xticks(range(num_classes))
ax_cm.set_yticks(range(num_classes))
ax_cm.set_xticklabels(class_names)
ax_cm.set_yticklabels(class_names)
ax_cm.set_xlabel("Predicted class")
ax_cm.set_ylabel("True class")
ax_cm.set_title(f"Test Confusion Matrix (accuracy {overall_acc:.2%})")
fig3.tight_layout()


#saves the graphs with a timestamp so older runs aren't overwritten
graph_dir = r"C:\Users\anshi\Desktop\garbage sorting\testgraphs"
os.makedirs(graph_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
batch_path = os.path.join(graph_dir, f"test_loss_accuracy_by_batch_{stamp}.png")
class_path = os.path.join(graph_dir, f"test_per_class_{stamp}.png")
fig1.savefig(batch_path, dpi=150, bbox_inches="tight")
cm_path = os.path.join(graph_dir, f"test_confusion_matrix_{stamp}.png")
fig2.savefig(class_path, dpi=150, bbox_inches="tight")
fig3.savefig(cm_path, dpi=150, bbox_inches="tight")
print ("Saved graph: ", batch_path)
print ("Saved graph: ", class_path)
print ("Saved graph: ", cm_path)

plt.show()
