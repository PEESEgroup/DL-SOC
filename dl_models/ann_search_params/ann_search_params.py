import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import numpy as np
import time
import pickle


def create_model(trial):
    num_layers = trial.suggest_int("n_layers", 3, 5)
    model = tf.keras.Sequential()
    for i in range(num_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 16, 256, log=True)
        model.add(tf.keras.layers.Dense(num_hidden, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def load_array(feature_arrays, label_arrays, batch_size, is_train=True, buffer_size=10000):
    feature_dataset = tf.data.Dataset.from_tensor_slices(feature_arrays)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_arrays)
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
    if is_train:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def objective(trial):
    X_train = np.load('../../data/point_data/rf_train_1.npy')
    X_val = np.load('../../data/point_data/rf_val_1.npy')
    y_train = np.load('../../data/y/y_train_1.npy')
    y_val = np.load('../../data/y_val_1.npy')
    
    y_train, y_val = np.log(y_train), np.log(y_val)
    
    train_ds = load_array(X_train, y_train, batch_size=128, buffer_size=X_train.shape[0])
    val_ds = load_array(X_val, y_val, batch_size=128, is_train=False, buffer_size=None)
    
    model = create_model(trial)
    
    monitor = 'val_loss'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15),
        TFKerasPruningCallback(trial, monitor),
    ]
    
    history = model.fit(
        x=train_ds,
        epochs=300,
        steps_per_epoch=int(np.round(X_train.shape[0]/128)),
        validation_data=val_ds,
        validation_steps=int(np.round(X_val.shape[0]/128)),
        callbacks=callbacks,
        verbose=0,
    )
        
    return history.history[monitor][-15]


# perform optimal hyperparameter search
start = time.time()
study_name = 'ann_param_study'
study = optuna.create_study(study_name=study_name, direction="minimize", storage='sqlite:///ann_param_study.db')
study.optimize(objective, n_trials=200)

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
# print("Time elapsed: ", time.time() - start)

fig = optuna.visualization.matplotlib.plot_optimization_history(study)

with open("sampler.pkl", "wb") as fout:
    pickle.dump(study.sampler, fout)