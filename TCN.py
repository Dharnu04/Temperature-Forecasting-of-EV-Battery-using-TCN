import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LayerNormalization, Dropout, Flatten, Add
from tcn import TCN
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# --- Streamlit Application Title ---
st.title("Lithium-ion Cell Temperature Forecasting - Hybrid TCN-Transformer Model")

# --- File Uploader Widget ---
# Allows the user to upload a CSV file containing battery data.
uploaded_file = st.file_uploader("Upload your battery dataset CSV", type=["csv"])

# --- Main Application Logic (executed after file upload) ---
if uploaded_file is not None:
    # Load and preprocess the dataset
    df = pd.read_csv(uploaded_file, parse_dates=['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Define features and targets for the model
    # Features are the input variables used for prediction.
    features = df[['Ambient Temp (°C)', 'Initial Temp (°C)', 'SOC (%)', 'Time Since Last Discharge (hrs)']]
    # Targets are the actual battery temperatures for Day 1, Day 2, and Day 3.
    targets = {
        "Day 1": df['Actual Battery Temp Day 1 (°C)'],
        "Day 2": df['Actual Battery Temp Day 2 (°C)'],
        "Day 3": df['Actual Battery Temp Day 3 (°C)']
    }

    # Initialize and apply MinMaxScaler to features
    # Scaling helps in normalizing the input data, which can improve model performance.
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)

    # Initialize and apply MinMaxScaler to each target variable
    # A separate scaler is used for each day's temperature to allow independent inverse transformation.
    scalers_target = {day: MinMaxScaler() for day in targets}
    targets_scaled = {day: scalers_target[day].fit_transform(targets[day].values.reshape(-1, 1)) for day in targets}

    # --- Data Sequencing Function ---
    def create_sequences(features_data, target_data, time_steps=10):
        """
        Creates time series sequences from features and target data.
        
        Args:
            features_data (np.array): Scaled feature data.
            target_data (np.array): Scaled target data.
            time_steps (int): The number of previous time steps to consider for each sequence.
            
        Returns:
            tuple: A tuple containing two numpy arrays (X, y) where X are the input sequences
                   and y are the corresponding target values.
        """
        X, y = [], []
        for i in range(len(features_data) - time_steps):
            X.append(features_data[i:(i + time_steps)])
            y.append(target_data[i + time_steps])
        return np.array(X), np.array(y)

    # Define the number of time steps for sequence creation
    time_steps = 10
    # Create sequences for training using Day 1 target (this will be overwritten for each day's training)
    X_seq, y_day1_seq = create_sequences(features_scaled, targets_scaled["Day 1"], time_steps)

    # --- Train-Test Split ---
    # Split the dataset into training and testing sets.
    # X_train, X_test will be the input sequences, and y_train, y_test will be the target values.
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_day1_seq, test_size=0.2, random_state=42)

    # --- Hybrid Model Definition (TCN + Transformer Encoder) ---
    # This section defines the neural network architecture.

    # Input layer: Defines the shape of the input data (time_steps, number_of_features).
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # TCN Layer: Temporal Convolutional Network for capturing temporal dependencies.
    # return_sequences=True ensures the output maintains the sequence dimension for the next layer.
    tcn_layer = TCN(return_sequences=True)(input_layer)

    # Transformer Encoder Block: Designed to capture long-range dependencies and global context.
    # MultiHeadAttention: Allows the model to jointly attend to information from different representation subspaces.
    attention_output = MultiHeadAttention(num_heads=8, key_dim=128)(tcn_layer, tcn_layer)
    # LayerNormalization: Stabilizes the activations and speeds up training.
    norm1 = LayerNormalization(epsilon=1e-6)(attention_output)
    
    # Feed Forward Network: A simple fully connected network within the transformer block.
    ff_dense = Dense(128, activation='relu')(norm1)
    # The output dimension matches the TCN layer output for the residual connection.
    ff_output = Dense(tcn_layer.shape[-1])(ff_dense)
    # Residual connection: Adds the input of the block to its output, helping with gradient flow.
    transformer_output = Add()([norm1, ff_output])
    norm2 = LayerNormalization(epsilon=1e-6)(transformer_output)

    # Flatten and Dense Layers for final prediction
    # Flatten: Converts the 3D output of the transformer block into a 2D array.
    flatten_layer = Flatten()(norm2)
    # Dense layers: Standard fully connected layers for regression.
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    # Dropout: Helps prevent overfitting by randomly setting a fraction of input units to 0 at each update.
    dropout_layer = Dropout(0.3)(dense_layer)
    # Output layer: Produces the single temperature prediction.
    output_layer = Dense(1, name='temp_output')(dropout_layer)

    # Create the Keras Model
    model = Model(inputs=input_layer, outputs=output_layer)
    # Compile the model: Define the optimizer and loss function.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Dictionaries to store predictions and actual values for each day
    predictions = {}
    actuals = {}
    
    # Placeholder for history object to display training/validation loss
    history = None

    # --- Model Training and Prediction Loop for Each Day ---
    for day, target_scaled in targets_scaled.items():
        st.write(f"### Processing {day} Battery Temperature Prediction...")
        
        # Re-create sequences and split data for the current day's target
        # This ensures the model is trained specifically for each day's temperature forecast.
        _, y_data_current_day = create_sequences(features_scaled, target_scaled, time_steps)
        X_train_current, X_test_current, y_train_current, y_test_current = train_test_split(
            X_seq, y_data_current_day, test_size=0.2, random_state=42
        )
        
        epochs = 50 # Number of training epochs

        # Display training progress in Streamlit
        with st.spinner(f"Training model for {day}... Please wait."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(1, epochs + 1):
                # Train the model for one epoch
                # verbose=0 suppresses the default Keras training output.
                history = model.fit(X_train_current, y_train_current, batch_size=64, epochs=1, 
                                    validation_data=(X_test_current, y_test_current), verbose=0)
                
                # Update the progress bar and status text
                progress_bar.progress(epoch / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} completed. Training Loss: {history.history['loss'][0]:.4f}, Validation Loss: {history.history['val_loss'][0]:.4f}")
            
            st.success(f"Model training completed for {day}!")

        # Make predictions on the test set
        pred_scaled = model.predict(X_test_current)
        
        # Inverse transform predictions and actual values to original scale
        pred_inverse = scalers_target[day].inverse_transform(pred_scaled)
        y_test_inverse = scalers_target[day].inverse_transform(y_test_current)

        # Store results
        predictions[day] = pred_inverse.flatten()
        actuals[day] = y_test_inverse.flatten()

    # --- Comparative Plot for All Days ---
    st.subheader("Comparative Analysis of Predicted vs. Actual Battery Temperatures")
    fig_all = go.Figure()
    colors = ['blue', 'red', 'green'] # Colors for different days

    # Add traces for actual and predicted temperatures for each day
    for i, day in enumerate(targets.keys()):
        fig_all.add_trace(go.Scatter(y=actuals[day], mode='lines', name=f'Actual {day}', 
                                     line=dict(color=colors[i], dash='dot')))
        fig_all.add_trace(go.Scatter(y=predictions[day], mode='lines', name=f'Predicted {day}', 
                                     line=dict(color=colors[i])))

    fig_all.update_layout(title="Comparative Battery Temperature Prediction Across Days",
                          xaxis_title='Time Steps',
                          yaxis_title='Battery Temperature (°C)',
                          hovermode='x')
    st.plotly_chart(fig_all)

    # --- Model Evaluation Function ---
    def evaluate_model(y_true, y_pred, day_label):
        """
        Calculates and returns common regression evaluation metrics.
        
        Args:
            y_true (np.array): Actual target values.
            y_pred (np.array): Predicted target values.
            day_label (str): Label for the day (e.g., "Day 1").
            
        Returns:
            str: A formatted string showing RMSE, MAE, R², and Accuracy.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # Accuracy is calculated as 100 * (1 - (MAE / mean of actual values))
        accuracy = 100 * (1 - (mae / np.mean(y_true))) 
        return (f"**{day_label}:** RMSE={rmse:.4f}, MAE={mae:.4f}, "
                f"R²={r2:.4f}, Accuracy={accuracy:.2f}%")

    # --- Display Final Model Performance Summary ---
    st.subheader("Model Performance Summary for Each Day")
    for day in targets.keys():
        st.markdown(evaluate_model(actuals[day], predictions[day], day))
        
    # --- Individual Day Prediction Plots ---
    st.subheader("Detailed Battery Temperature Prediction for Each Day")
    for day in targets.keys():
        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(y=actuals[day], mode='lines', name=f'Actual Temp {day}', line=dict(color='blue')))
        fig_day.add_trace(go.Scatter(y=predictions[day], mode='lines', name=f'Predicted Temp {day}', line=dict(color='red')))
        fig_day.update_layout(title=f'Battery Temperature Prediction - {day}',
                              xaxis_title='Time Steps',
                              yaxis_title='Battery Temperature (°C)',
                              hovermode='x')
        st.plotly_chart(fig_day)

    # --- Plot Model Training & Validation Loss ---
    st.subheader("Model Training & Validation Loss Over Epochs")

    # The 'history' object contains the loss values from the last trained day.
    if history: # Ensure history is not None
        train_losses = history.history['loss']
        val_losses = history.history['val_loss']

        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss', line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss', line=dict(color='red', dash='dot')))

        fig_loss.update_layout(title="Training vs. Validation Loss per Epoch",
                               xaxis_title='Epochs',
                               yaxis_title='Loss',
                               hovermode='x')
        st.plotly_chart(fig_loss)
    else:
        st.info("Train the model first to see the loss plot.")


    # --- Download Results Button ---
    # Allows users to download the actual and predicted temperatures as a CSV file.
    results_df = pd.DataFrame({
        "Actual Day 1": actuals["Day 1"],
        "Predicted Day 1": predictions["Day 1"],
        "Actual Day 2": actuals["Day 2"],
        "Predicted Day 2": predictions["Day 2"],
        "Actual Day 3": actuals["Day 3"],
        "Predicted Day 3": predictions["Day 3"],
    })

    st.download_button(
        label="Download All Predictions as CSV",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name="battery_temperature_predictions.csv",
        mime="text/csv",
        help="Click to download a CSV file containing actual and predicted temperatures for all days."
    )
