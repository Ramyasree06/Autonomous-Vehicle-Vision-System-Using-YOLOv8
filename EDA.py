# EDA for YOLOv8 Custom Dataset
import os
import cv2
import glob
import yaml
import matplotlib.pyplot as plt

# Load YAML config
yaml_path = "data/data.yaml"

with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

train_path = os.path.join("data", "train", "images")
valid_path = os.path.join("data", "valid", "images")

print("YAML details:")
print(data_config)
print("\n")

# Count images
def count_images(path):
    exts = ['.jpg', '.jpeg', '.png']
    return len([f for f in os.listdir(path) if os.path.splitext(f)[-1].lower() in exts])

train_imgs = count_images(train_path)
valid_imgs = count_images(valid_path)

print(f"Train Images: {train_imgs}")
print(f"Valid Images: {valid_imgs}")

# Sample image display
sample_imgs = glob.glob(os.path.join(train_path, '*'))[:4]
plt.figure(figsize=(10, 10))
for i, img_path in enumerate(sample_imgs):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Example {i+1}")
plt.tight_layout()
plt.show()

# Count annotation files
train_labels = os.path.join("data", "train", "labels")
valid_labels = os.path.join("data", "valid", "labels")

def get_class_distribution(label_folder, names):
    class_counts = {i: 0 for i in range(len(names))}

    for label_file in glob.glob(os.path.join(label_folder, '*.txt')):
        with open(label_file, 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
    return class_counts

names = data_config['names']
train_distribution = get_class_distribution(train_labels, names)
valid_distribution = get_class_distribution(valid_labels, names)

# Plot class distribution
plt.figure(figsize=(8, 5))
plt.bar(names, [train_distribution[i] for i in range(len(names))], color='blue', alpha=0.6, label="Train")
plt.bar(names, [valid_distribution[i] for i in range(len(names))], color='orange', alpha=0.6, label="Valid")
plt.title("Class Distribution")
plt.ylabel("Object Count")
plt.legend()
plt.show()

# Checking image-label mismatch
def check_missing_pairs(img_folder, label_folder):
    missing = []
    imgs = [os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]
    labels = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith('.txt')]
    for img in imgs:
        if img not in labels:
            missing.append(img)
    return missing

missed = check_missing_pairs(train_path, train_labels)
if missed:
    print("Warning: Images without labels:", missed)
else:
    print("All training images have labels ✅")





