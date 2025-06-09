# !/usr/bin/env python
# Sample 2D-CNN from scratch to classify canadian crop land 
# Note: No data augmentation is used to facilitate comparison with QCNN
# 
#
# Script info
# -----------
# __author__ = 'Ryan Godin'
# __copyright__ = '© His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food Canada,' \
#                 '2025-'
# __credits__ = 'Ryan Godin, Etienne Lord'
# __email__ = ''
# __license__ = 'Open Government Licence - Canada'
# __maintainer__ = 'Ryan Godin, Etienne Lord'
# __status__ = 'Development'
# __version__ = '1.0.1'

import os
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import keras
import argparse

def setup_seed(seed=None):
    """
    Set random seeds for reproducibility
    Args:
        seed: Integer seed value. If None, generates random 7-digit seed
    Returns:
        seed: The seed value used
    """
    if seed is None:
        seed = int(''.join(str(np.random.randint(0, 9)) for _ in range(7)))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed

def setup_save_dir(savedir):
    """
    Create directory for saving results if it doesn't exist
    Args:
        savedir: Path to directory where results will be saved
    """
    Path(savedir).mkdir(parents=True, exist_ok=True)    
    if not os.path.isdir(savedir):
        print(f'Error: The output directory {savedir} could not be created!')
        exit(-1)

def load_data(file_path):
    """
    Load and split dataset into train, validation and test sets
    Args:
        file_path: Path to NPZ file containing the dataset
    Returns:
        Tuple containing split datasets and metadata
    """
    loaded_data = np.load(file_path)
    data_vectors = loaded_data['data']
    encoded_labels = loaded_data['labels'] #0:'BARLEY', 1:CORN', 2:'OAT', 3:'ORCHARD', 4:'SOYBEAN
    labels = loaded_data['crops'] 
    channels = loaded_data['channels']
    
    # Split data: 70% train, 15% validation, 15% test
    x_train, X_temp, y_train, y_temp = train_test_split(data_vectors, encoded_labels, test_size=0.30)
    x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50)
    return x_train, y_train, x_val, y_val, x_test, y_test, labels, channels

def create_2d_cnn_model(input_shape=(16, 16, 106), num_classes=5):
    """
    Create a 2D CNN model architecture
    Note: The 2D CNN architecture was created using a keras tuner optimization and represent the best 
          architecture found after 100 trials.
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=input_shape),
        
        # First convolutional block
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional block
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional block
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Classification head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def train_model(model, X_train, y_train, X_val, y_val, checkpoint_path, epochs=50):
    """
    Train the model with specified callbacks and parameters
    Note: 50 epochs was found to be a sweet spot
    Args:
        model: Keras model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        checkpoint_path: Path to save model checkpoints
        epochs: Number of training epochs
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, 
                         min_lr=0.000001, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',
                       save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, 
                     restore_best_weights=True, verbose=1)
    ]
    
    # Compile model
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', 
                 optimizer=optimizer, 
                 metrics=['accuracy'])
    
    # Train model
    return model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
    Returns:
        test_accuracy: Overall test accuracy
        classification_rep: Detailed classification report
        conf_matrix: Confusion matrix
    """
    # Generate predictions
    y_pred = model.predict(X_test, batch_size=1024)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate metrics
    classification_rep = classification_report(y_test, y_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    
    return test_accuracy, classification_rep, conf_matrix

def save_results(save_dir, seed, test_accuracy, classification_rep, conf_matrix, 
                model_history, epochs, date, model):
    """
    Save training results, metrics and plots
    Args:
        save_dir: Directory to save results
        seed: Random seed used
        test_accuracy: Model accuracy on test set
        classification_rep: Classification report
        conf_matrix: Confusion matrix
        model_history: Training history
        epochs: Number of epochs
        date: Current date
        model: Model name/type
    """
    # Save metrics to text file
    metrics_file = os.path.join(save_dir, f'metrics_{date}_{seed}.txt')
    actual_epochs = len(model_history.history['loss'])
    
    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Date: {date}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Epochs: {actual_epochs}\n")
        f.write(f"Classification Report:\n\n{classification_rep}\n")
        f.write(f"Confusion Matrix:\n\n{conf_matrix}\n")
    
    # Extract training history
    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    train_acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    
    # Create and save training plots
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), train_acc, 'b--o', label='Training accuracy')
    plt.plot(range(1, actual_epochs + 1), val_acc, 'r--s', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), train_loss, 'b--o', label='Training loss')
    plt.plot(range(1, actual_epochs + 1), val_loss, 'r--s', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt_file = os.path.join(save_dir, f"history_{date}_{seed}.svg")
    plt.savefig(plt_file)
    plt.close()

def main(args):
    """
    Main function to run the training pipeline
    Args:
        args: Command line arguments
    """
    # Clear any existing Keras sessions
    tf.keras.backend.clear_session()
    
    # Setup output directory
    setup_save_dir(args.save_dir)
    date = datetime.now()
    
    # Train and evaluate for each seed
    for seed in args.seed:
        # Set random seed
        setup_seed(seed)
        
        # Load and prepare data
        x_train, y_train, x_val, y_val, x_test, y_test, labels, channels = load_data(args.data)
        input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        n_classes = len(labels)
        
        # Create and train model
        model = create_2d_cnn_model(input_shape, n_classes)
        checkpoint_path = os.path.join(args.save_dir, f"model_checkpoint_{date}_{seed}.keras")
        model_history = train_model(model, x_train, to_categorical(y_train), 
                                  x_val, to_categorical(y_val), 
                                  checkpoint_path, args.epochs)
        
        # Load best model and save
        best_model = tf.keras.models.load_model(checkpoint_path)
        best_model_path = os.path.join(args.save_dir, f"model_{date}_{seed}.keras").replace(" ", "_")
        print(f'Saving best model to {best_model_path}')
        best_model.save(best_model_path)
        
        # Evaluate model
        test_accuracy, classification_rep, conf_matrix = evaluate_model(best_model, x_test, y_test)
        
        # Print dataset information
        print(f'Dataset size used - train:{len(y_train)} val:{len(y_val)} test:{len(y_test)}')
        
        # Save results
        save_results(args.save_dir, seed, test_accuracy, classification_rep, 
                    conf_matrix, model_history, args.epochs, date, args.model)
    
    print(f"Results saved in {args.save_dir}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train 2D CNN on Hyperspectral Data")
    parser.add_argument("--data", type=str, required=True, 
                      help="Path to the dataset NPZ file")
    parser.add_argument("--save_dir", required=True, 
                      help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=50, 
                      help="Number of epochs for training")
    parser.add_argument("--seed", nargs="+", type=int, default=[42], 
                      help="Random seed(s) for reproducibility")
    parser.add_argument("--model", type=str, default="2DCNN", 
                      help="Model type (2DCNN)")
    
    args = parser.parse_args()
    main(args)
