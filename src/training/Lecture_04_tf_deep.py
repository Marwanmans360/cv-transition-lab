import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from pathlib import Path

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================================
# 0) PATHS (SAME AS YOUR ORIGINAL CODE)
# ============================================================
# Your project root folder that contains src/...
PROJECT_ROOT = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab"
# CIFAR-10 python batches folder
# CIFAR_PATH = r"C:\Users\user\OneDrive - TechnoVal\Desktop\Scripts\ML\cv-transition-lab\data\cifar-10-batches-py"
# For Google Colab (uncomment if using Colab):
CIFAR_PATH = "/content/drive/MyDrive/datasets/cifar-10-batches-py"

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# Import your existing data loader
from src.data.CIFAR10_Load_Test_and_Train_Data import load_cifar10

# ============================================================
# 1) LOAD & PREPROCESS DATA (USING YOUR EXISTING PIPELINE)
# ============================================================
print("[LOADING DATA...]", flush=True)
(X_train, y_train), (X_test, y_test) = load_cifar10(CIFAR_PATH)
print("[DATA LOADED]", flush=True)

# Flatten if needed (same as your code)
if X_train.ndim == 4:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

print("Raw shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)

# Train/CV Split (same as your code)
SEED = 42
rng = np.random.default_rng(SEED)
idx = rng.permutation(X_train.shape[0])

N_cv = 10000
cv_idx = idx[:N_cv]
tr_idx = idx[N_cv:]

X_cv = X_train[cv_idx]
y_cv = y_train[cv_idx]
X_tr = X_train[tr_idx]
y_tr = y_train[tr_idx]

print("\nSplit shapes:")
print("Train:", X_tr.shape, y_tr.shape)
print("CV   :", X_cv.shape, y_cv.shape)

# Optional subsampling (same as your code)
TRAIN_SUBSET = None  # e.g. 5000 or None
CV_SUBSET = None     # e.g. 2000 or None

if TRAIN_SUBSET is not None:
    X_tr = X_tr[:TRAIN_SUBSET]
    y_tr = y_tr[:TRAIN_SUBSET]

if CV_SUBSET is not None:
    X_cv = X_cv[:CV_SUBSET]
    y_cv = y_cv[:CV_SUBSET]

print("\nAfter optional subsampling:")
print("Train:", X_tr.shape, y_tr.shape)
print("CV   :", X_cv.shape, y_cv.shape)

# Preprocessing (same as your code)
X_tr = X_tr.astype(np.float32)
X_cv = X_cv.astype(np.float32)
X_test = X_test.astype(np.float32)

mean_img = X_tr.mean(axis=0, keepdims=True)
X_tr -= mean_img
X_cv -= mean_img
X_test -= mean_img

std_img = X_tr.std(axis=0, keepdims=True) + 1e-8
X_tr /= std_img
X_cv /= std_img
X_test /= std_img

print("\nAfter preprocessing:")
print(f"Train min={X_tr.min():.3f}, max={X_tr.max():.3f}, mean={X_tr.mean():.3f}, std={X_tr.std():.3f}")
print("[PREPROCESSING COMPLETE - READY FOR TENSORFLOW TRAINING...]", flush=True)

# Convert labels to int32 for TensorFlow
y_tr = y_tr.astype(np.int32)
y_cv = y_cv.astype(np.int32)
y_test = y_test.astype(np.int32)

# ============================================================
# 2) BUILD DEEP NETWORKS (3 or 4 layers)
# ============================================================

def build_deep_network(input_dim, 
                       num_classes=10,
                       hidden_layers=[512, 256, 128],  # 3 hidden layers
                       activation='relu',
                       use_batch_norm=False,
                       dropout_rate=0.0,
                       l2_reg=1e-3,
                       seed=42):
    """
    Build a deep fully-connected network.
    
    Args:
        input_dim: Input feature dimension (3072 for CIFAR-10)
        num_classes: Number of output classes (10 for CIFAR-10)
        hidden_layers: List of hidden layer sizes, e.g. [512, 256, 128] = 3 layers
        activation: 'relu', 'leaky_relu', 'gelu', 'selu', 'tanh', 'sigmoid'
        use_batch_norm: Add batch normalization after each layer
        dropout_rate: Dropout probability (0.0 = no dropout)
        l2_reg: L2 regularization strength
        seed: Random seed for reproducibility
    
    Returns:
        Keras model
    """
    tf.random.set_seed(seed)
    
    # Activation mapping
    activation_map = {
        'relu': 'relu',
        'leaky_relu': tf.keras.layers.LeakyReLU(alpha=0.01),
        'gelu': 'gelu',
        'selu': 'selu',
        'tanh': 'tanh',
        'sigmoid': 'sigmoid',
        'swish': 'swish',
    }
    
    act_layer = activation_map.get(activation, 'relu')
    
    # Build model
    model = models.Sequential(name=f'DeepNN_{len(hidden_layers)}layers_{activation}')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i, hidden_size in enumerate(hidden_layers):
        layer_name = f'dense_{i+1}'
        
        # Dense layer with L2 regularization
        model.add(layers.Dense(
            hidden_size,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name=layer_name
        ))
        
        # Batch normalization (optional)
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        
        # Activation
        if isinstance(act_layer, str):
            model.add(layers.Activation(act_layer, name=f'{activation}_{i+1}'))
        else:
            model.add(act_layer)
        
        # Dropout (optional)
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer (no activation, softmax in loss function)
    model.add(layers.Dense(num_classes, name='output'))
    
    return model

# ============================================================
# 3) CUSTOM TRAINING LOOP WITH GRADIENT TRACKING
# ============================================================

class GradientTracker(keras.callbacks.Callback):
    """
    Callback to track gradient norms during training.
    This is the KEY part for understanding gradient flow!
    """
    def __init__(self, log_freq=10):
        super().__init__()
        self.log_freq = log_freq
        self.gradient_history = {}
        self.epoch_gradients = {}
        
    def on_train_begin(self, logs=None):
        # Initialize gradient tracking for each trainable layer
        for layer in self.model.layers:
            if layer.trainable_weights:
                layer_name = layer.name
                self.gradient_history[layer_name] = {
                    'norms': [],
                    'means': [],
                    'stds': [],
                    'epochs': []
                }
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq != 0:
            return
        
        # Store for this epoch
        self.epoch_gradients[epoch] = {}
        
        print(f"  [Gradient Tracking - Epoch {epoch}]")
        
    def compute_gradients_on_batch(self, x_batch, y_batch):
        """
        Compute gradients for a batch and track statistics.
        Call this manually during training.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(x_batch, training=True)
            loss = self.model.compiled_loss(y_batch, predictions)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Track gradient statistics for each layer
        layer_idx = 0
        for layer in self.model.layers:
            if not layer.trainable_weights:
                continue
            
            layer_name = layer.name
            
            # Get gradients for this layer's weights
            layer_grads = []
            for _ in layer.trainable_weights:
                if layer_idx < len(gradients):
                    layer_grads.append(gradients[layer_idx])
                    layer_idx += 1
            
            if layer_grads:
                # Compute statistics
                grad_concat = tf.concat([tf.reshape(g, [-1]) for g in layer_grads], axis=0)
                grad_norm = tf.norm(grad_concat).numpy()
                grad_mean = tf.reduce_mean(tf.abs(grad_concat)).numpy()
                grad_std = tf.math.reduce_std(grad_concat).numpy()
                
                # Store
                self.gradient_history[layer_name]['norms'].append(grad_norm)
                self.gradient_history[layer_name]['means'].append(grad_mean)
                self.gradient_history[layer_name]['stds'].append(grad_std)
        
        return gradients

def train_with_gradient_tracking(
    model,
    X_tr, y_tr,
    X_cv, y_cv,
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    optimizer_type='adam',
    track_gradients=True,
    gradient_track_freq=5,
    save_dir='models_tf',
    activation='relu',
    architecture='3layer'
):
    """
    Train model with comprehensive gradient tracking.
    
    Returns:
        history: Training history with gradient statistics
        gradient_tracker: GradientTracker callback with detailed gradient info
    """
    
    # Create save directory (relative to your PROJECT_ROOT)
    save_path = Path(save_dir) / f"{architecture}_{activation}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Optimizer
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = []
    
    # Gradient tracker
    gradient_tracker = None
    if track_gradients:
        gradient_tracker = GradientTracker(log_freq=gradient_track_freq)
        callbacks.append(gradient_tracker)
    
    # Model checkpoint (save best model)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(save_path / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Custom callback to track gradients on batches
    class BatchGradientTracker(keras.callbacks.Callback):
        def __init__(self, grad_tracker, track_freq, X_tr, y_tr):
            super().__init__()
            self.grad_tracker = grad_tracker
            self.track_freq = track_freq
            self.batch_count = 0
            self.X_tr = X_tr
            self.y_tr = y_tr
            
        def on_train_batch_end(self, batch, logs=None):
            self.batch_count += 1
            # Track gradients every N batches
            if self.batch_count % 50 == 0:
                # Get a batch
                idx = np.random.choice(len(self.X_tr), size=min(256, len(self.X_tr)), replace=False)
                x_batch = self.X_tr[idx]
                y_batch = self.y_tr[idx]
                
                self.grad_tracker.compute_gradients_on_batch(x_batch, y_batch)
    
    if track_gradients:
        batch_grad_tracker = BatchGradientTracker(gradient_tracker, gradient_track_freq, X_tr, y_tr)
        callbacks.append(batch_grad_tracker)
    
    # Train
    print(f"\n{'='*80}")
    print(f"Training {architecture} with {activation} activation")
    print(f"Optimizer: {optimizer_type}, LR: {learning_rate}, Batch size: {batch_size}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    history = model.fit(
        X_tr, y_tr,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_cv, y_cv),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on CV set
    val_loss, val_acc = model.evaluate(X_cv, y_cv, verbose=0)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"  Total time: {training_time:.2f}s")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Final val accuracy: {val_acc:.4f}")
    print(f"{'='*80}\n")
    
    # Save training history
    history_dict = {
        'activation': activation,
        'architecture': architecture,
        'optimizer': optimizer_type,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'final_val_accuracy': float(val_acc),
        'training_time': training_time,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    return history, gradient_tracker

# ============================================================
# 4) GRADIENT VISUALIZATION & ANALYSIS
# ============================================================

def plot_gradient_flow(gradient_tracker, activation_name, save_path=None):
    """
    Visualize gradient flow through the network layers.
    This shows you WHERE vanishing/exploding gradients occur!
    """
    if not gradient_tracker or not gradient_tracker.gradient_history:
        print("No gradient data to plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get layer names (only dense layers)
    layer_names = [name for name in gradient_tracker.gradient_history.keys() 
                   if 'dense' in name]
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    
    # Plot 1: Gradient norms over training
    ax = axes[0, 0]
    for i, layer_name in enumerate(layer_names):
        norms = gradient_tracker.gradient_history[layer_name]['norms']
        if norms:
            ax.plot(norms, label=layer_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Training Step (every 50 batches)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title(f'Gradient Norms Over Training - {activation_name}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient means (average magnitude)
    ax = axes[0, 1]
    for i, layer_name in enumerate(layer_names):
        means = gradient_tracker.gradient_history[layer_name]['means']
        if means:
            ax.plot(means, label=layer_name, color=colors[i], linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Absolute Gradient')
    ax.set_title(f'Average Gradient Magnitude - {activation_name}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final gradient distribution across layers
    ax = axes[1, 0]
    final_norms = []
    for layer_name in layer_names:
        norms = gradient_tracker.gradient_history[layer_name]['norms']
        if norms:
            final_norms.append(norms[-10:])  # Last 10 measurements
    
    if final_norms:
        positions = range(len(layer_names))
        bp = ax.boxplot(final_norms, labels=layer_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Final Gradient Distribution by Layer')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Gradient ratio (early vs late layers)
    ax = axes[1, 1]
    if len(layer_names) >= 2:
        first_layer = layer_names[0]
        last_layer = layer_names[-1]
        
        first_norms = np.array(gradient_tracker.gradient_history[first_layer]['norms'])
        last_norms = np.array(gradient_tracker.gradient_history[last_layer]['norms'])
        
        min_len = min(len(first_norms), len(last_norms))
        if min_len > 0:
            ratio = first_norms[:min_len] / (last_norms[:min_len] + 1e-10)
            ax.plot(ratio, linewidth=2, color='darkblue')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
                      label='Equal gradients')
            ax.set_xlabel('Training Step')
            ax.set_ylabel(f'{first_layer} / {last_layer}')
            ax.set_title('Gradient Ratio: First Layer / Last Layer')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Gradient Flow Analysis - {activation_name.upper()}', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / f'gradient_flow_{activation_name}.png', 
                   dpi=150, bbox_inches='tight')
    plt.show()

def diagnose_gradient_health(gradient_tracker, activation_name):
    """
    Automatic diagnosis of gradient problems.
    """
    print(f"\n{'='*80}")
    print(f"GRADIENT HEALTH DIAGNOSIS: {activation_name.upper()}")
    print(f"{'='*80}\n")
    
    layer_names = [name for name in gradient_tracker.gradient_history.keys() 
                   if 'dense' in name]
    
    for layer_name in layer_names:
        norms = gradient_tracker.gradient_history[layer_name]['norms']
        if not norms:
            continue
        
        # Statistics on final gradients
        final_norms = norms[-10:] if len(norms) >= 10 else norms
        avg_norm = np.mean(final_norms)
        std_norm = np.std(final_norms)
        max_norm = np.max(norms)
        min_norm = np.min(norms)
        
        print(f"{layer_name}:")
        print(f"  Average gradient norm: {avg_norm:.2e}")
        print(f"  Std deviation: {std_norm:.2e}")
        print(f"  Max seen: {max_norm:.2e}")
        print(f"  Min seen: {min_norm:.2e}")
        
        # Diagnosis
        if avg_norm < 1e-5:
            print(f"  ⚠️  SEVERE VANISHING - Layer has stopped learning!")
        elif avg_norm < 1e-3:
            print(f"  ⚡ MODERATE VANISHING - Weak learning signal")
        elif avg_norm > 100:
            print(f"  ⚠️  EXPLODING GRADIENTS - Training unstable!")
        else:
            print(f"  ✓  Healthy gradient flow")
        print()
    
    # Check gradient ratio (vanishing severity)
    if len(layer_names) >= 2:
        first_layer = layer_names[0]
        last_layer = layer_names[-1]
        
        first_avg = np.mean(gradient_tracker.gradient_history[first_layer]['norms'][-10:])
        last_avg = np.mean(gradient_tracker.gradient_history[last_layer]['norms'][-10:])
        
        ratio = first_avg / (last_avg + 1e-10)
        
        print(f"Gradient Flow Ratio ({first_layer}/{last_layer}): {ratio:.2e}")
        if ratio < 0.001:
            print(f"  ⚠️  SEVERE gradient vanishing in early layers!")
            print(f"  → First layer gets {1/ratio:.0f}x smaller gradients than last")
        elif ratio < 0.01:
            print(f"  ⚡ MODERATE gradient imbalance")
        else:
            print(f"  ✓  Balanced gradient flow across layers")
    
    print(f"\n{'='*80}\n")

# ============================================================
# 5) EXPERIMENT RUNNER: COMPARE ACTIVATIONS
# ============================================================

def compare_activations_deep(
    X_tr, y_tr, X_cv, y_cv,
    activations=['relu', 'leaky_relu', 'gelu', 'tanh', 'sigmoid'],
    architecture='3layer',  # '3layer' or '4layer'
    use_batch_norm=False,
    epochs=30,
    batch_size=128,
    learning_rate=0.001,
    save_dir='models_tf_deep'
):
    """
    Compare multiple activation functions on deep networks.
    """
    
    # Architecture configs
    arch_configs = {
        '3layer': [512, 256, 128],
        '4layer': [512, 256, 128, 64],
        'wide_3layer': [1024, 512, 256],
        'deep_5layer': [512, 256, 128, 64, 32]
    }
    
    hidden_layers = arch_configs.get(architecture, [512, 256, 128])
    input_dim = X_tr.shape[1]
    
    results = {}
    
    for activation in activations:
        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT: {architecture} | Activation: {activation.upper()}")
        print(f"{'#'*80}\n")
        
        # Build model
        model = build_deep_network(
            input_dim=input_dim,
            num_classes=10,
            hidden_layers=hidden_layers,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=0.0,  # Start without dropout
            l2_reg=1e-3,
            seed=SEED
        )
        
        # Print model summary
        model.summary()
        
        # Train with gradient tracking
        history, grad_tracker = train_with_gradient_tracking(
            model,
            X_tr, y_tr,
            X_cv, y_cv,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_type='adam',
            track_gradients=True,
            gradient_track_freq=5,
            save_dir=save_dir,
            activation=activation,
            architecture=architecture
        )
        
        # Analyze gradients
        diagnose_gradient_health(grad_tracker, activation)
        
        # Plot gradient flow
        save_path = Path(save_dir) / f"{architecture}_{activation}"
        plot_gradient_flow(grad_tracker, activation, save_path)
        
        # Store results
        results[activation] = {
            'best_val_acc': max(history.history['val_accuracy']),
            'final_val_acc': history.history['val_accuracy'][-1],
            'history': history.history,
            'gradient_tracker': grad_tracker
        }
    
    # Final comparison
    print(f"\n\n{'#'*80}")
    print(f"FINAL RESULTS - {architecture.upper()} ARCHITECTURE")
    print(f"{'#'*80}\n")
    
    sorted_acts = sorted(results.keys(), 
                        key=lambda k: results[k]['best_val_acc'], 
                        reverse=True)
    
    for rank, act in enumerate(sorted_acts, 1):
        r = results[act]
        print(f"{rank}. {act:15s}: Best Val Acc = {r['best_val_acc']:.4f} "
              f"| Final = {r['final_val_acc']:.4f}")
    
    return results

# ============================================================
# 6) RUN EXPERIMENTS
# ============================================================

if __name__ == "__main__":
    print("\n\n" + "="*80)
    print("STARTING DEEP NETWORK EXPERIMENTS WITH GRADIENT TRACKING")
    print("="*80 + "\n")
    
    # Experiment 1: 3-layer network WITHOUT batch norm
    # print("\n" + "#"*80)
    # print("EXPERIMENT 1: 3-Layer Network (No Batch Norm)")
    # print("This will show gradient vanishing in sigmoid/tanh!")
    # print("#"*80)
    
    # results_3layer = compare_activations_deep(
    #     X_tr, y_tr, X_cv, y_cv,
    #     activations=['relu', 'gelu', 'tanh', 'sigmoid'],
    #     architecture='3layer',
    #     use_batch_norm=False,
    #     epochs=30,
    #     batch_size=128,
    #     learning_rate=0.001,
    #     save_dir='models_tf_deep'
    # )
    
    # Experiment 2: 4-layer network WITHOUT batch norm
    # (Optional - uncomment to see gradient problems worsen)
    
    # print("\n" + "#"*80)
    # print("EXPERIMENT 2: 4-Layer Network (No Batch Norm)")
    # print("Gradient problems will be EVEN MORE SEVERE!")
    # print("#"*80)
    
    # results_4layer = compare_activations_deep(
    #     X_tr, y_tr, X_cv, y_cv,
    #     activations=['relu', 'gelu', 'tanh', 'sigmoid'],
    #     architecture='4layer',
    #     use_batch_norm=False,
    #     epochs=30,
    #     batch_size=128,
    #     learning_rate=0.001,
    #     save_dir='models_tf_deep'
    # )
    
    
    # Experiment 3: 3-layer WITH batch norm
    # (Save this for AFTER you learn about batch normalization!)
    print("\n" + "#"*80)
    print("EXPERIMENT 3: 3-Layer Network (WITH Batch Norm)")
    print("Watch the gradient vanishing DISAPPEAR!")
    print("#"*80)
    
    results_3layer_bn = compare_activations_deep(
        X_tr, y_tr, X_cv, y_cv,
        activations=['relu', 'gelu', 'tanh', 'sigmoid'],
        architecture='3layer',
        use_batch_norm=True,  # THE MAGIC SWITCH!
        epochs=30,
        batch_size=128
    )
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("Check the models_tf_deep/ folder for:")
    print("  - Saved models (best_model.keras)")
    print("  - Training history (training_history.json)")
    print("  - Gradient flow plots (gradient_flow_*.png)")
    print("="*80 + "\n")


# After training completes, test loading:
print("\n" + "="*80)
print("VERIFYING SAVED WEIGHTS")
print("="*80)

# Load the ReLU model
model_relu = keras.models.load_model('models_tf_deep/3layer_relu/best_model.keras')
print("✓ Successfully loaded ReLU model")

# Test prediction
test_pred = model_relu.predict(X_cv[:10])
print(f"✓ Predictions work! Shape: {test_pred.shape}")

# Extract weight matrix
W1 = model_relu.layers[0].get_weights()[0]
print(f"✓ First layer weights: {W1.shape}")
print(f"  Range: [{W1.min():.3f}, {W1.max():.3f}]")