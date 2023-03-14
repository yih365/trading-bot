import tensorflow as tf
import sys
import yfinance as yf
import numpy as np


def main():
    symbol = '^GSPC'
    data = yf.Ticker(symbol).history(period="max")
    # Show only data for most recent day
    data = data.iloc[-1:].copy()

    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    features_input = np.array(data[feature_names])

    model = tf.keras.models.load_model(sys.argv[1])
    prediction = model.predict(features_input)[0]
    print(str(prediction[0]) + f'% chance of increasing in price')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Use like this: python predict.py [model file]')
    main()