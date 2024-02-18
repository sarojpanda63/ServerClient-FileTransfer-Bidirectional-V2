import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.optimizers import Adam
from keras.losses import Loss, Huber
from sklearn.preprocessing import StandardScaler


# tf.random.set_seed(3)
def create_model_01():
    ts_model = Sequential()
    # Add LSTM
    ts_model.add(LSTM(5, input_shape=(1, 70)))
    ts_model.add(Dense(1))
    opt = Adam(learning_rate=0.00001)
    # Compile with Adam Optimizer. Optimize for minimum mean square error
    ts_model.compile(optimizer=opt, loss=Huber(), metrics=["mse"])
    return ts_model


def train_model(ts_model, epochs, batch_size,train_x, train_y,test_x, test_y,cp_callback):
    history = ts_model.fit(train_x, train_y,
                           epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(test_x, test_y),
                           shuffle=False, callbacks=[cp_callback])
    return history


def save_loss_val_loss(trainHistoryDict, history):
    with open(trainHistoryDict + '_loss', 'wb') as file_pi:
        pickle.dump(history.history['loss'], file_pi)
    with open(trainHistoryDict + '_val_loss', 'wb') as file_pi:
        pickle.dump(history.history['val_loss'], file_pi)


def update_loss_val_loss(trainHistoryDict, history):
    loss, val_loss = load_loss_val_loss(trainHistoryDict)
    with open(trainHistoryDict + '_loss', 'wb') as file_pi:
        pickle.dump(loss + history.history['loss'], file_pi)
    with open(trainHistoryDict + '_val_loss', 'wb') as file_pi:
        pickle.dump(val_loss + history.history['val_loss'], file_pi)


def load_loss_val_loss(trainHistoryDict):
    with open(trainHistoryDict + '_loss', "rb") as file_pi:
        loss = pickle.load(file_pi)
    with open(trainHistoryDict + '_val_loss', "rb") as file_pi:
        val_loss = pickle.load(file_pi)
    return loss, val_loss


def view_summary(model):
    model.summary()


def plot_train_and_test_loss_wrt_epoch(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Learning Curve(LOSAng)')
    plt.ylabel('Huber Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.savefig("LOSAng.jpg")
    plt.show()


def plot(history_path):
    m, v = load_loss_val_loss(history_path)
    plot_train_and_test_loss_wrt_epoch(m, v)


def load_model(model, checkpoint_path):
    model.load_weights(checkpoint_path)
    return model


def clean_path(path):
    os.rmdir(path)


def initialization(batch_size,epochs,train_x, train_y,test_x, test_y,history_path):
    # Initialization(run once). Clean tarining_1 folder
    model = create_model_01()
    history = train_model(model, epochs, batch_size,train_x, train_y,test_x, test_y)
    save_loss_val_loss(history_path, history)


def run_additional_epochs_and_plot_validation_curve(epoch_,batch_size,train_x, train_y,test_x, test_y,history_path,checkpoint_path):
    ts_model = create_model_01()
    ts_model = load_model(ts_model, checkpoint_path)
    history = train_model(ts_model, epoch_, batch_size,train_x, train_y,test_x, test_y)
    update_loss_val_loss(history_path, history)
    plot()


def plot_data(df,data_point):
    # plt.figure(figsize=(20,5)).suptitle("Total data", fontsize=20)
    plt.plot(df.head(data_point))
    plt.show()


class QuantileLoss(Loss):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        quantile_loss = []
        for q in self.quantiles:
            e = (y_true - y_pred)
            quantile_loss.append(tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), axis=-1))
        return tf.reduce_mean(tf.reduce_sum(quantile_loss, axis=0))


def create_rnn_dataset(data, lookback=1):
    data_x, data_y = [], []
    for i in range(len(data) - lookback - 1):
        a = data[i:(i + lookback), 0]
        data_x.append(a)
        # The next point
        data_y.append(data[i + lookback, 0])
    return np.array(data_x), np.array(data_y)

def load_data_01(total_size):
    np.random.seed(1)
    df = pd.read_csv("output_abilene_5_v1.csv", index_col='source')
    df=df[:total_size]
    df = df[["LOSAng"]]
    df.sort_index(inplace=True)
    print(df.shape)
    plt.plot(list(df["LOSAng"]))
    plt.title('data set(LOSAng)')
    plt.ylabel('Traffic')
    plt.xlabel('Time (with interval of 5min)')
    plt.legend(['LOSAng'], loc='upper right')
    plt.savefig("LOSAng-dataset.jpg")
    return df

def lstm_run_client_1():
    pass


def make_rnn_data(df,train_size,lookback):
    print("Traffic Range before scaling : " , min(df.LOSAng), max(df.LOSAng))
    scaler = StandardScaler()
    scaled_df=scaler.fit_transform(df)
    print("Traffic Range after scaling : " , min(scaled_df), max(scaled_df))
    train_size = int(train_size)
    train_df = scaled_df[0:train_size,:]
    test_df = scaled_df[train_size-lookback:,:]
    print("\nShaped of Train, Test : ", train_df.shape, test_df.shape)
    train_x, train_y = create_rnn_dataset(train_df,lookback)
    train_x = np.reshape(train_x, (train_x.shape[0],1, train_x.shape[1]))
    print("Shapes of X, Y(train): ",train_x.shape, train_y.shape)
    test_x, test_y = create_rnn_dataset(test_df,lookback)
    test_x = np.reshape(test_x, (test_x.shape[0],1, test_x.shape[1]))
    print("Shapes of X, Y(test): ",test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y