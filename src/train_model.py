import numpy as np #For numerical operations and array handling
import os #For file system operations
from sklearn.model_selection import train_test_split #For splitting data
from sklearn.preprocessing import StandardScaler, LabelEncoder #For scaling and encoding
from sklearn.metrics import classification_report #For evaluation metrics
import tensorflow as tf #For building and training the model
from tensorflow.keras import layers, models #For neural network layers and models
import matplotlib.pyplot as plt #For plotting training history
import pickle #For saving label encoder and scaler

#Load data from .npy files
def load_data(data_dir='data/collected'):
    X, y = [], []
    
    #Iterate through files in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.npy'):
            label = filename.replace('.npy', '')
            data = np.load(os.path.join(data_dir, filename))
            X.extend(data)
            y.extend([label] * len(data))
    
    #Convert to numpy arrays
    return np.array(X), np.array(y)

#Normalize landmarks by centering and scaling
def normalize_landmarks(X):
    X_normalized = []
    
    #Process each sample
    for sample in X:

        #Reshape to (21, 3) for 21 landmarks with x, y, z coordinates
        landmarks = sample.reshape(21, 3)
        centered = landmarks - landmarks[0]
        scale = np.linalg.norm(centered[9])

        #Scale if norm is greater than 0
        if scale > 0:
            centered = centered / scale
        X_normalized.append(centered.flatten())
    
    return np.array(X_normalized)

#Build the neural network model
def build_model(input_shape, num_classes):

    #Define a Sequential model
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    #Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

#Main training function
def train():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes")
    
    #Normalize landmarks
    X = normalize_landmarks(X)
    
    #Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    #Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Build and train the model
    model = build_model(X_train.shape[1], len(label_encoder.classes_))
    model.summary()
    
    #Set up early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    #Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    #Plot training history
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_))
    
    #Save the trained model, label encoder, and scaler
    model.save('models/sign_language_model.h5')
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

#Main execution
if __name__ == "__main__":
    train()
