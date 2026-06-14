import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(base_path="plant_data", img_size=128):
    data = []
    labels = []
    categories = os.listdir(base_path)

    print("Loading images from directories...")
    for category in categories:
        path = os.path.join(base_path, category)
        if not os.path.isdir(path):
            continue
            
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Read and resize
                    img = cv2.resize(img, (img_size, img_size))
                    data.append(img)
                    labels.append(category)
            except Exception as e:
                pass

    # Convert to numpy arrays to save RAM
    data = np.array(data, dtype=np.uint8)
    labels = np.array(labels)

    # Encode text labels to integers
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    
    # Convert labels to One-Hot
    labels_encoded = to_categorical(labels_encoded)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels_encoded, test_size=0.2, random_state=42
    )

    # Free up RAM before normalization
    del data
    del labels

    print("Normalizing data...")
    # Convert to 32-bit floats and normalize to 0-1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    print(f"Data ready! Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, encoder