"""
Project Title: Development of an Artificial Neural Network (ANN) for Binary Classification 
Using the make_moons Dataset from Scikit-learn.

Description:
An optimized, beginner-friendly machine learning project that builds, trains,
evaluates, and visualizes an Artificial Neural Network (ANN) on non-linearly
separable data from Scikit-learn's make_moons dataset.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ensure prompt output is flushed immediately to terminal
sys.stdout.reconfigure(line_buffering=True)


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 65)
    print("  Artificial Neural Network (ANN) - Binary Classification")
    print("  Dataset: Scikit-learn make_moons (Non-linear Classification)")
    print("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Generate the Dataset
    # -------------------------------------------------------------
    print("\n[Step 1] Generating make_moons dataset...")
    # Generate 500 samples with realistic noise (0.2)
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    print(f"Total samples generated: {X.shape[0]}")
    print(f"Feature shape: {X.shape}, Label shape: {y.shape}")
    print(f"Classes: {np.unique(y)} (Binary classification: 0 and 1)")

    # -------------------------------------------------------------
    # Step 2: Prepare the Data
    # -------------------------------------------------------------
    print("\n[Step 2] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # -------------------------------------------------------------
    # Step 3: Build the Artificial Neural Network (ANN)
    # -------------------------------------------------------------
    print("\n[Step 3] Building the Neural Network architecture...")
    # Optimized Architecture:
    # - Input: 2 features (X1, X2)
    # - Hidden Layer 1: 32 neurons, ReLU activation (sufficient capacity for curved boundary)
    # - Hidden Layer 2: 16 neurons, ReLU activation (smooth hierarchical feature refinement)
    # - Output Layer: 1 neuron, Sigmoid activation (calibrated binary probability)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile with Adam optimizer (learning rate tuned to 0.005 for efficient convergence)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -------------------------------------------------------------
    # Step 4: Train the Model with Validation & Early Stopping
    # -------------------------------------------------------------
    print("\n[Step 4] Training the ANN with validation monitoring...")
    # Early stopping monitors validation loss to prevent overfitting and restore best weights
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Training completed successfully.")

    # -------------------------------------------------------------
    # Step 5: Evaluate the Model
    # -------------------------------------------------------------
    print("\n[Step 5] Evaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Sample Predictions Comparison
    print("\n--- Sample Predictions (First 10 Test Samples) ---")
    raw_predictions = model.predict(X_test[:10], verbose=0)
    predicted_classes = (raw_predictions >= 0.5).astype(int).flatten()

    print(f"{'Sample':<8}{'Actual Class':<15}{'Predicted Prob':<18}{'Predicted Class':<18}{'Status'}")
    print("-" * 68)
    for i in range(10):
        actual = y_test[i]
        prob = raw_predictions[i][0]
        pred = predicted_classes[i]
        status = "Correct" if actual == pred else "Incorrect"
        print(f"{i + 1:<8}{actual:<15}{prob:<18.4f}{pred:<18}{status}")

    # -------------------------------------------------------------
    # Step 6: Visualize the Results
    # -------------------------------------------------------------
    print("\n[Step 6] Generating visualization graphs...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Graph 1: Dataset Visualization
    ax1 = axes[0]
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], color='#1f77b4', label='Class 0', alpha=0.8, edgecolors='k')
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], color='#ff7f0e', label='Class 1', alpha=0.8, edgecolors='k')
    ax1.set_title("1. make_moons Dataset Visualization", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Feature 1 (X1)")
    ax1.set_ylabel("Feature 2 (X2)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Graph 2: Training Performance (Training & Validation Curves)
    ax2 = axes[1]
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    ax2.plot(epochs_range, history.history['loss'], label='Train Loss', color='#d62728', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], label='Val Loss', color='#ff9896', linewidth=2, linestyle='--')
    ax2.plot(epochs_range, history.history['accuracy'], label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy', color='#98df8a', linewidth=2, linestyle='--')
    ax2.set_title("2. Training & Validation Performance", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Metric Value")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Graph 3: Decision Boundary
    ax3 = axes[2]
    # Create a grid covering feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    # Predict over the entire grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = model.predict(grid_points, verbose=0)
    grid_preds = (grid_preds >= 0.5).astype(int)
    zz = grid_preds.reshape(xx.shape)

    # Plot decision boundary regions
    ax3.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.coolwarm)
    # Overlay the test samples
    ax3.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='#1f77b4', label='Class 0 (Test)', edgecolors='k')
    ax3.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='#ff7f0e', label='Class 1 (Test)', edgecolors='k')
    ax3.set_title("3. ANN Decision Boundary", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Feature 1 (X1)")
    ax3.set_ylabel("Feature 2 (X2)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # Save a high-resolution copy
    plt.savefig("results_visualization.png", dpi=300)
    print("Plot saved as 'results_visualization.png'.")
    print("Displaying graphs (close the plot window to finish)...")
    plt.show()

    print("\nProject execution finished successfully.")


if __name__ == "__main__":
    main()
