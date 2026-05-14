import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import pickle
import copy
from sklearn.preprocessing import StandardScaler

class SimpleProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        mae = logs.get("mae", 0)
        print(f"loss: {loss:.4f}, mae: {mae:.4f}")

class TransferLearningModel:
    def __init__(self, input_dim=7, seq_length=30, hidden_layers=(64, 32), learning_rate=0.001, dropout_rate=0.2):
        """
        Initializes the Deep Neural Network using TensorFlow/Keras.
        
        Args:
            input_dim (int): Number of input features (default 7: Return, RSI, MA, etc.)
            seq_length (int): Number of time steps in the input sequence.
            hidden_layers (tuple): Structure of hidden layers (e.g., (64, 32) neurons).
            learning_rate (float): How fast the model updates its weights.
            dropout_rate (float): Fraction of the input units to drop to prevent overfitting.
        """
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self):
        """
        Constructs the neural network architecture.
        """
        # Sequential: A linear stack of layers
        model = tf.keras.Sequential()
        
        # 1. Input Layer
        # Set the shape to natively accept 3D sequential data
        model.add(tf.keras.layers.InputLayer(shape=(self.seq_length, self.input_dim)))
        
        # 2. Hidden Layers
        for i, units in enumerate(self.hidden_layers):
            # LSTM: Long Short-Term Memory layer for sequential patterns.
            # return_sequences must be True for all LSTM layers except the last one.
            return_seq = i < len(self.hidden_layers) - 1
            model.add(tf.keras.layers.LSTM(units, return_sequences=return_seq, name=f'lstm_layer_{i+1}'))
            
            if self.dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(self.dropout_rate, name=f'dropout_layer_{i+1}'))
            
        # 3. Output Layer
        # 1 unit: We want a single prediction (Percentage Rise).
        # Activation 'linear': Allows outputting any real number (positive or negative).
        model.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))
        
        # 4. Compile
        # Optimizer: Adam is the standard algorithm for adjusting weights.
        # Loss: Mean Squared Error (MSE) is standard for regression.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])
        
        return model

    def _scale_data(self, X, fit=False):
        """Helper to properly scale 3D data by temporarily flattening to 2D."""
        if X.ndim == 3:
            samples, timesteps, features = X.shape
            X_2d = X.reshape(-1, features)
            if fit:
                X_scaled = self.scaler.fit_transform(X_2d)
            else:
                X_scaled = self.scaler.transform(X_2d)
            return X_scaled.reshape(samples, timesteps, features)
        else:
            if fit:
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)

    def update_scaler(self, X):
        """Incrementally updates the scaler's mean and std using partial_fit."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 3:
            X_2d = X.reshape(-1, X.shape[2])
            self.scaler.partial_fit(X_2d)
        else:
            self.scaler.partial_fit(X)

    def train(self, X, y, epochs=50, batch_size=32, fit_scaler=False, patience=5, verbose=1):
        """
        Trains the model on the provided data.
        
        Transfer Learning Note:
        In TensorFlow, calling .fit() repeatedly on the same model instance
        does NOT reset the weights. It continues training from where it left off.
        This is exactly what we need for Stage 1 -> Stage 2 -> Stage 3.
        """
        # Ensure inputs are numpy arrays of float32 (standard for TF)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        X = self._scale_data(X, fit=fit_scaler)
        
        callbacks = []
        if verbose > 0:
            callbacks.append(SimpleProgress())
            
        if patience > 0:
            # Stop training if 'loss' doesn't improve for 'patience' epochs
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)

        # Set verbose=0 to suppress Keras default progress bar
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)
        return history

    def predict(self, X):
        """
        Returns binary class predictions (0 or 1).
        """
        X = np.array(X, dtype=np.float32)
        X = self._scale_data(X, fit=False)
        # Return raw continuous values (predicted percentage return)
        return self.model.predict(X, verbose=0)

    def evaluate(self, X, y):
        """
        Returns the Mean Absolute Error (MAE) of the model on the given data.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X = self._scale_data(X, fit=False)
        loss, mae = self.model.evaluate(X, y, verbose=0)
        return mae

    def save(self, filepath):
        """
        Saves the entire model (architecture, weights, and optimizer state) to a file.
        """
        filepath = str(filepath)  # Ensure filepath is a string for Windows compatibility
        self.model.save(filepath)
        
        # Save the scaler alongside the model
        scaler_path = filepath + ".scaler"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, filepath):
        """
        Loads a saved model from a file.
        """
        filepath = str(filepath)  # Ensure filepath is a string
        self.model = tf.keras.models.load_model(filepath)
        
        # After loading, update the wrapper's attributes to match the model's architecture
        try:
            # input_shape is typically (None, seq_length, input_dim) for LSTM
            input_shape = self.model.input_shape
            if len(input_shape) == 3:
                self.seq_length = input_shape[1]
                self.input_dim = input_shape[2]
            elif len(input_shape) == 2: # Fallback for non-sequential models
                self.input_dim = input_shape[1]
        except Exception as e:
            print(f"Warning: Could not auto-determine input shape from loaded model: {e}")

        # Load the scaler
        scaler_path = filepath + ".scaler"
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            print(f"Warning: Scaler file not found at {scaler_path}. Model will use uninitialized scaler!")

    def copy(self):
        """
        Creates a deep copy of the model in memory.
        Useful for Transfer Learning where we want to branch off a base model.
        """
        new_instance = TransferLearningModel(
            input_dim=self.input_dim, 
            seq_length=self.seq_length,
            hidden_layers=self.hidden_layers, 
            learning_rate=self.learning_rate,
            dropout_rate=self.dropout_rate
        )
        new_instance.model.set_weights(self.model.get_weights())
        new_instance.scaler = copy.deepcopy(self.scaler)
        return new_instance

    @staticmethod
    def exists(filepath):
        """
        Checks if the model and its scaler exist at the given filepath.
        """
        filepath = str(filepath)
        scaler_path = filepath + ".scaler"
        return os.path.exists(filepath) and os.path.exists(scaler_path)
