from typing import List, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import json
import time
from datetime import timedelta
import pandas as pd
import random
import argparse

class DepthPredictor:
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names
        self.model = None
        self.scaler = StandardScaler()
        
        if feature_names is not None:
            self._build_model()
    
    def _build_model(self):
        if self.feature_names is None:
            raise ValueError("Cannot build model without feature names")
            
        inputs = keras.Input(shape=(len(self.feature_names),))
        x = keras.layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
        x = keras.layers.Dense(36, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.Dense(24, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        rt_output = keras.layers.Dense(2, name='rt_output')(x)
        solved_output = keras.layers.Dense(2, name='solved_output')(x)
        
        self.model = keras.Model(inputs=inputs, 
                               outputs=[rt_output, solved_output])

    def prepare_data(self, training_data: List[Dict], fit_scaler=True):
        """Prepare training data from the new format"""
        X = []
        y_rt = []
        y_solved = []
        
        for item in training_data:
            features = item['features']
            outcomes = item['outcomes']
            
            X.append([features[f] for f in self.feature_names])
            y_rt.append(np.log(max(outcomes['rt'], 1)))  # log transform RT
            y_solved.append(outcomes['solved'])
        
        X = np.array(X)
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return (X_scaled, 
                np.array(y_rt), 
                np.array(y_solved))

    def gaussian_nll(self, y_true, y_pred_mean, y_pred_log_std, alpha=10, beta=1, prior_weight=0.1):
        """Negative log likelihood of normal distribution with precision regularization"""
        std = tf.maximum(tf.exp(y_pred_log_std), 1e-10)
        precision = 1.0 / (std ** 2)
        
        sq_diff = tf.square((y_true - y_pred_mean) / std)
        nll = 0.5 * sq_diff + y_pred_log_std + 0.5 * np.log(2 * np.pi)
        
        precision_penalty = (alpha - 1) * tf.math.log(precision) - beta * precision
        return tf.reduce_mean(nll - prior_weight * precision_penalty)
    
    def compile(self):
        def gaussian_loss(y_true, y_pred):
            return self.gaussian_nll(y_true, y_pred[:, 0], y_pred[:, 1])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.model.compile(optimizer=optimizer,
                         loss=[gaussian_loss, gaussian_loss])
    
    def fit(self, X, y_rt, y_solved, **kwargs):
        history = self.model.fit(X, [y_rt, y_solved], **kwargs)
        return history
    
    def predict(self, features: Dict):
        X = np.array([[features[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        rt_pred, solved_pred = self.model.predict(X_scaled, verbose=0)
        
        return {
            'rt': {
                'mean': rt_pred[0,0],
                'std': np.exp(rt_pred[0,1])
            },
            'solved': {
                'mean': solved_pred[0,0],
                'std': np.exp(solved_pred[0,1])
            }
        }

    def save_model(self, folder: str):
        """Save model weights and state to a folder"""
        os.makedirs(folder, exist_ok=True)
        weights_path = os.path.join(folder, 'model.weights.h5')
        state_path = os.path.join(folder, 'state.pkl')
        
        self.save_state(weights_path, state_path)

    def load_model(self, folder: str, feature_names = None):
        """Load a model from a folder containing weights and state"""
        weights_path = os.path.join(folder, 'model.weights.h5')
        state_path = os.path.join(folder, 'state.pkl')
        
        if not os.path.exists(state_path):
            raise ValueError(f"State file not found in {folder}")

        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            prev_scaler = state['scaler']
            prev_feature_names = state['feature_names']

        if feature_names is None or set(feature_names) == set(prev_feature_names):
            self.feature_names = prev_feature_names
            self.scaler = prev_scaler
            self._build_model()
            self.model.load_weights(weights_path)
        else:
            raise ValueError("Feature names do not match previous state")

    def save_state(self, weights_path, state_path):
        """Save model weights, scaler state, and feature names"""
        self.model.save_weights(weights_path)
        print(f"Saving scaler with mean: {self.scaler.mean_[:5]}")
        
        state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

    def predict_batch(self, features_list: List[Dict], verbose=0):
        """Make predictions for a batch of feature dictionaries."""
        X = np.array([[features[f] for f in self.feature_names] for features in features_list])
        X_scaled = self.scaler.transform(X)
        
        rt_preds, solved_preds = self.model.predict(X_scaled, verbose=verbose)
        
        results = []
        for i in range(len(features_list)):
            # Convert log-normal predictions to normal space
            log_mean = rt_preds[i,0]
            log_std = rt_preds[i,1]
            rt_mean = np.exp(log_mean + (log_std**2)/2)
            rt_std = np.sqrt((np.exp(log_std**2) - 1) * np.exp(2*log_mean + log_std**2))
            
            results.append({
                'final_log_rt': {
                    'mean': rt_preds[i,0],
                    'std': np.exp(rt_preds[i,1])
                },
                'final_rt': {
                    'mean': rt_mean,
                    'std': rt_std
                },
                'final_solved': {
                    'mean': solved_preds[i,0],
                    'std': np.exp(solved_preds[i,1])
                }
            })
        
        return results

    def generate_prediction_data(self, data, is_train=True, batch_size=512):
        """Generate predictions for all states"""
        def process_batch(features, metadata):
            if not features:
                return []
            
            X = np.array(features)
            X_scaled = self.scaler.transform(X)
            rt_preds, solved_preds = self.model.predict(X_scaled, verbose=0)
            
            batch_rows = []
            for j in range(len(features)):
                batch_idx, batch_features = metadata[j]
                batch_outcomes = data[batch_idx]['outcomes']
                
                row = {
                    'idx': batch_idx,
                    'is_train': is_train,
                    'pred_log_rt_mean': rt_preds[j,0],
                    'pred_log_rt_std': np.exp(rt_preds[j,1]),
                    'pred_rt_mean': np.exp(rt_preds[j,0] + (np.exp(rt_preds[j,1])**2)/2),
                    'pred_rt_std': np.sqrt((np.exp(np.exp(rt_preds[j,1])**2) - 1) * 
                                         np.exp(2*rt_preds[j,0] + np.exp(rt_preds[j,1])**2)),
                    'pred_solved_mean': solved_preds[j,0],
                    'pred_solved_std': np.exp(solved_preds[j,1]),
                    'actual_rt': batch_outcomes['rt'],
                    'actual_log_rt': np.log(max(batch_outcomes['rt'], 1)),
                    'actual_solved': batch_outcomes['solved'],
                    **{f'feature_{k}': v for k, v in batch_features.items()}
                }
                
                # Calculate log likelihoods
                rt_std = max(np.exp(rt_preds[j,1]), 1e-10)
                rt_ll = -0.5 * ((np.log(max(batch_outcomes['rt'], 1)) - rt_preds[j,0]) / rt_std)**2 - \
                        np.log(rt_std) - 0.5 * np.log(2 * np.pi)
                
                solved_std = max(np.exp(solved_preds[j,1]), 1e-10)
                solved_ll = -0.5 * ((batch_outcomes['solved'] - solved_preds[j,0]) / solved_std)**2 - \
                           np.log(solved_std) - 0.5 * np.log(2 * np.pi)
                
                row.update({
                    'rt_log_likelihood': rt_ll,
                    'solved_log_likelihood': solved_ll,
                })
                
                batch_rows.append(row)
            return batch_rows

        rows = []
        batch_features = []
        batch_metadata = []
        
        # Calculate total predictions needed
        total_predictions = len(data)
        predictions_made = 0
        start_time = time.time()
        
        for idx, item in enumerate(data):
            features = item['features']
            batch_features.append([features[f] for f in self.feature_names])
            batch_metadata.append((idx, features))
            
            if len(batch_features) >= batch_size:
                rows.extend(process_batch(batch_features, batch_metadata))
                
                # Update progress
                predictions_made += len(batch_features)
                elapsed_time = time.time() - start_time
                progress = predictions_made / total_predictions
                if progress > 0:
                    estimated_total = elapsed_time / progress
                    remaining = estimated_total - elapsed_time
                    print(f"\rGenerating predictions: {progress*100:.1f}% "
                          f"[{timedelta(seconds=int(elapsed_time))} < "
                          f"{timedelta(seconds=int(remaining))}]", end="")
                
                batch_features = []
                batch_metadata = []
        
        # Process any remaining items
        rows.extend(process_batch(batch_features, batch_metadata))
        
        total_time = time.time() - start_time
        print(f"\rPredictions complete in {timedelta(seconds=int(total_time))}")
        return pd.DataFrame(rows)

if __name__ == "__main__":
    # Create directories if they don't exist
    save_path = 'evaluation/saved_models'
    predictions_path = 'evaluation/model_output'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(predictions_path, exist_ok=True)

    # Load data
    with open("evaluation/training_data.json", "r") as f:
        data = json.load(f)
        random.shuffle(data)
    
    # Get feature names from first item
    features = list(data[0]['features'].keys())
    print(f"Using features: {features}")
    
    # Split into train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    
    # Initialize predictor with features
    predictor = DepthPredictor(features)
    
    if not args.retrain:
        try:
            predictor.load_model(save_path, feature_names=features)
            print("Loaded existing model state")
        except ValueError as e:
            print(f"Cannot load existing model: {e}")
            args.retrain = True
            print("Will train new model...")

    

    if not args.skip_train:
        print("Training model...")
        X, y_rt, y_solved = predictor.prepare_data(train_data, fit_scaler=True)
        predictor.compile()
        
        # Learning rate schedule
        initial_learning_rate = 0.0005
        decay_steps = 1000
        decay_rate = 0.9
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps,
            decay_rate
        )
        
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0
        )
        
        history = predictor.fit(X, y_rt, y_solved, 
                              epochs=100,
                              validation_split=0.2, 
                              batch_size=128)
        
        predictor.save_model(save_path)
        print("Saved model state")
    
    # Generate and save predictions
    print("Generating predictions...")
    train_predictions = predictor.generate_prediction_data(train_data, is_train=True)
    test_predictions = predictor.generate_prediction_data(test_data, is_train=False)
    
    # Combine and save
    all_predictions = pd.concat([train_predictions, test_predictions], ignore_index=True)
    all_predictions.to_csv(os.path.join(predictions_path, 'predictions.csv'), index=False)
    print(f"\nSaved predictions to {predictions_path}")