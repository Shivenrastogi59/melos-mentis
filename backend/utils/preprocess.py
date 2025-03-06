import os
import cv2
import numpy as np

def load_images_from_folder(folder, label):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return images, labels

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename).replace("\\", "/")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, (48, 48))  # Resize to 48x48
            images.append(img)
            labels.append(label)

    return images, labels

def preprocess_data(dataset_path):
    train_folder = os.path.join(dataset_path, "train").replace("\\", "/")
    test_folder = os.path.join(dataset_path, "test").replace("\\", "/")

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for label in os.listdir(train_folder):
        label_path = os.path.join(train_folder, label)
        imgs, lbls = load_images_from_folder(label_path, label)
        train_images.extend(imgs)
        train_labels.extend(lbls)

    for label in os.listdir(test_folder):
        label_path = os.path.join(test_folder, label)
        imgs, lbls = load_images_from_folder(label_path, label)
        test_images.extend(imgs)
        test_labels.extend(lbls)

    # Convert lists to numpy arrays
    train_images = np.array(train_images).astype("float32") / 255.0
    test_images = np.array(test_images).astype("float32") / 255.0

    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    dataset_path = ("./dataset/emotions").replace("\\", "/")
    print(f"Dataset path: {dataset_path}")
    (train_images, train_labels), (test_images, test_labels) = preprocess_data(dataset_path)
    print("Preprocessing complete!")
    print(f"Train set size: {len(train_images)}, Test set size: {len(test_images)}")
