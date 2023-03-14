import tensorflow as tf
import sys
import yfinance as yf
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Use like this: python training.py [file location to store model]")
        return   

    symbol = '^GSPC'
    data = yf.Ticker(symbol).history(period="max")
    data['Tomorrow'] = data['Close'].shift(-1)
    data['DirectionUp'] = (data['Tomorrow']>data['Open']).astype(int)
    data['DirectionDown'] = (data['Tomorrow']<data['Open']).astype(int)
    data_train = data.loc['1990-01-01':'2023-01-01'].copy()
    data_test = data.loc['2023-01-01':].copy()

    # Set target data
    target_names = ['DirectionUp', 'DirectionDown']
    target_train = data_train[target_names]
    target_test = data_test[target_names]
    target_train_arr = np.array(target_train)
    target_test_arr = np.array(target_test)

    # Set input data
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    features_train = data_train[feature_names]
    features_train_arr = np.array(features_train)
    features_test = data_test[feature_names]
    features_test_arr = np.array(features_test)

    model = build_model()

    # Train model
    model.fit(features_train_arr, target_train_arr, epochs=10)

    # Evaluate model
    model.evaluate(features_test_arr, target_test_arr, verbose=2)

    # Save to file
    file = sys.argv[1]
    model.save(file)
    print("Model saved.")


def build_model():
    model = tf.keras.models.Sequential([
        # Hidden layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),

        # Dropout
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    main()