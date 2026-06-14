import pickle
from preprocess import load_and_preprocess_data
from model import build_model

def train():
    # 1. Prepare Data
    # Note: Ensure you have a folder named 'plant_data' in the same directory!
    X_train, X_test, y_train, y_test, encoder = load_and_preprocess_data(base_path="plant_data", img_size=128)

    # Save the encoder so app.py knows the disease names later
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("Label encoder saved as 'label_encoder.pkl'")

    # 2. Build Model
    num_classes = y_train.shape[1]
    model = build_model(input_shape=(128, 128, 3), num_classes=num_classes)
    model.summary()

    # 3. Train Model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=32
    )

    # 4. Save Model
    model.save('plant_disease_model.keras')
    print("Model trained and saved as 'plant_disease_model.keras'")

if __name__ == "__main__":
    train()