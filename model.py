import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


class BaseballPyMC:
    def __init__(self, n_hidden=16):
        self.features = ["Age", "PA", "2B", "HR", "BB", "AVG", "WAR"]
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None
        self.trace = None
        self.n_hidden = n_hidden

    def prepare_data(self, df):
        """Prepare and scale features"""
        X = df[self.features].values.astype(float)
        if "next_WAR" in df.columns:
            y = df["next_WAR"].values.astype(float)
        else:
            y = None
        return X, y

    def build_and_fit(
        self, train_df, validation_split=0.2, n_epochs=100, batch_size=32
    ):
        X, y = self.prepare_data(train_df)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Scale data
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_val_scaled = self.x_scaler.transform(X_val)
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1)).ravel()

        # Create minibatches
        n_train = X_train_scaled.shape[0]
        n_batches = n_train // batch_size

        with pm.Model() as self.model:
            # Input data
            X_shared = pm.Data("X_shared", X_train_scaled)
            y_shared = pm.Data("y_shared", y_train_scaled)

            # Weights from input to hidden layer
            weights_in_1 = pm.Normal(
                "w_in_1", 0, sigma=0.1, shape=(X_train_scaled.shape[1], self.n_hidden)
            )

            # Weights from hidden layer to output
            weights_1_out = pm.Normal("w_1_out", 0, sigma=0.1, shape=(self.n_hidden,))

            # Build neural network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(X_shared, weights_in_1))
            mu = pm.math.dot(act_1, weights_1_out)

            # Model error with tighter prior
            sigma = pm.HalfNormal("sigma", sigma=0.5)

            # Simple Normal likelihood
            out = pm.Normal(
                "out", mu=mu, sigma=sigma, observed=y_shared, total_size=n_train
            )

            # Use ADVI with lower learning rate
            inference = pm.ADVI()

            # Track validation loss
            validation_losses = []

            # Training loop
            for epoch in range(n_epochs):
                # Fit ADVI with lower learning rate
                approx = inference.fit(
                    n=n_batches,
                    progressbar=False,
                    obj_optimizer=pm.adam(learning_rate=0.001),
                )

                # Get validation loss using the mean of the approximation
                means = approx.mean.eval()
                n_weights_1 = X_train_scaled.shape[1] * self.n_hidden
                w1_mean = means[:n_weights_1].reshape(
                    (X_train_scaled.shape[1], self.n_hidden)
                )
                w2_mean = means[n_weights_1 : n_weights_1 + self.n_hidden]

                val_act_1 = np.tanh(np.dot(X_val_scaled, w1_mean))
                val_pred = np.dot(val_act_1, w2_mean)
                val_loss = np.mean((val_pred - y_val_scaled) ** 2)
                validation_losses.append(val_loss)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

            # Get final samples
            self.trace = approx.sample(1000)

        return self.trace, validation_losses

    def predict(self, test_df):
        """Make predictions with uncertainty"""
        if self.trace is None:
            raise ValueError("Model hasn't been fit yet!")

        # Prepare test data
        X_test, _ = self.prepare_data(test_df)
        X_test_scaled = self.x_scaler.transform(X_test)

        # Get samples from trace
        w1_samples = self.trace.posterior["w_in_1"].values.reshape(
            -1, X_test_scaled.shape[1], self.n_hidden
        )
        w2_samples = self.trace.posterior["w_1_out"].values.reshape(-1, self.n_hidden)

        # Generate predictions
        predictions = []
        for i in range(len(w1_samples)):
            # Forward pass
            act_1 = np.tanh(np.dot(X_test_scaled, w1_samples[i]))
            pred = np.dot(act_1, w2_samples[i])
            predictions.append(pred)

        # Transform back to original scale
        predictions = np.array(predictions)
        predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1))

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)

        return mean_pred, lower_ci, upper_ci

    def plot_prediction_distribution(self, test_df, player_name=None):
        """Plot histogram of predicted WAR values using Plotly"""
        if self.trace is None:
            raise ValueError("Model hasn't been fit yet!")

        # Get predictions
        X_test, _ = self.prepare_data(test_df)
        X_test_scaled = self.x_scaler.transform(X_test)

        # Get samples from trace
        w1_samples = self.trace.posterior["w_in_1"].values.reshape(
            -1, X_test_scaled.shape[1], self.n_hidden
        )
        w2_samples = self.trace.posterior["w_1_out"].values.reshape(-1, self.n_hidden)

        predictions = []
        for i in range(len(w1_samples)):
            act_1 = np.tanh(np.dot(X_test_scaled, w1_samples[i]))
            pred = np.dot(act_1, w2_samples[i])
            predictions.append(pred)

        predictions = np.array(predictions)
        predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1))

        # Calculate statistics
        mean_pred = float(np.mean(predictions))
        lower_ci, upper_ci = map(float, np.percentile(predictions, [2.5, 97.5]))

        # Create figure
        fig = go.Figure()

        # Add histogram of predictions
        fig.add_trace(
            go.Histogram(
                x=predictions.flatten(),
                name="Predictions",
                nbinsx=30,
                histnorm="probability density",
                marker_color="rgba(135, 206, 235, 0.6)",
            )
        )

        # Add vertical lines for statistics
        fig.add_vline(
            x=mean_pred,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Predicted WAR: {mean_pred:.2f}",
            annotation_position="top",
        )

        fig.add_vline(
            x=lower_ci,
            line_dash="dot",
            line_color="green",
            annotation_text=f"95% CI: ({lower_ci:.2f},",
            annotation_position="bottom left",
        )

        fig.add_vline(
            x=upper_ci,
            line_dash="dot",
            line_color="green",
            annotation_text=f"{upper_ci:.2f})",
            annotation_position="bottom right",
        )

        # Add actual values if available
        if "next_WAR" in test_df.columns and not pd.isna(test_df["next_WAR"].values[0]):
            next_war = float(test_df["next_WAR"].values[0])
            fig.add_vline(
                x=next_war,
                line_dash="solid",
                line_color="blue",
                annotation_text=f"Actual next WAR: {next_war:.2f}",
                annotation_position="top",
            )

        if "WAR" in test_df.columns:
            current_war = float(test_df["WAR"].values[0])
            fig.add_vline(
                x=current_war,
                line_dash="dash",
                line_color="purple",
                annotation_text=f"Current WAR: {current_war:.2f}",
                annotation_position="bottom",
            )

        # Update layout
        title = (
            f"WAR Prediction Distribution for {player_name}"
            if player_name
            else "WAR Prediction Distribution"
        )
        fig.update_layout(
            title=title,
            xaxis_title="WAR",
            yaxis_title="Density",
            showlegend=False,
            hovermode="x",
            template="plotly_white",
            width=1000,
            height=600,
        )

        return fig

    def plot_validation_curve(self, validation_losses):
        """Plot validation loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(validation_losses)
        plt.title("Validation Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.yscale("log")
        plt.grid(True)
        return plt.gcf()

    def plot_feature_importances(self):
        """Plot overall importance of each feature"""
        if self.trace is None:
            raise ValueError("Model hasn't been fit yet!")

        # Calculate feature importance as mean absolute weight across all neurons
        w1_means = np.abs(
            self.trace.posterior["w_in_1"].mean(dim=("chain", "draw")).values
        )
        feature_importance = w1_means.mean(axis=1)

        # Create importance DataFrame
        importance_df = pd.DataFrame(
            {"Feature": self.features, "Importance": feature_importance}
        ).sort_values("Importance", ascending=True)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature", color="skyblue")
        plt.title("Overall Feature Importance")
        plt.tight_layout()
        return plt.gcf()

    def save_model(self, filepath):
        """Save the model, trace, and scalers"""
        import joblib

        save_dict = {
            "trace": self.trace,
            "x_scaler": self.x_scaler,
            "y_scaler": self.y_scaler,
            "features": self.features,
            "n_hidden": self.n_hidden,
        }

        joblib.dump(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the model, trace, and scalers"""
        import joblib

        load_dict = joblib.load(filepath)

        self.trace = load_dict["trace"]
        self.x_scaler = load_dict["x_scaler"]
        self.y_scaler = load_dict["y_scaler"]
        self.features = load_dict["features"]
        self.n_hidden = load_dict["n_hidden"]

        print(f"Model loaded from {filepath}")
