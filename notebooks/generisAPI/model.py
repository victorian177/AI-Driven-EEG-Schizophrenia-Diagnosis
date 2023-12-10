class Model:
    """Wrapper class for machine learning models."""

    def __init__(self, name, model, metrics, theorem, description, version):
        """
        Constructor for the Model class.

        Args:
            name (str): A name for the model.
            model: The machine learning model to be wrapped.
            metrics (list): A list of evaluation metrics.
            theorem (str): A theorem or principle associated with the model.
            description (str): A description or summary of the model.
            version (str): The version of the features used.
        """
        self.name = name
        try:
            # Instantiate the model with a random state of 0 (if applicable).
            self.model = model(random_state=0)
        except:
            self.model = model()
        self.metrics = metrics
        self.theorem = theorem
        self.description = description
        self.version = version

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X: Input features for training.
            y: True labels for training.

        Returns:
            Fitted model.
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input features for prediction.

        Returns:
            Model predictions.
        """
        return self.model.predict(X)

    def score(self, y_test, y_pred):
        """
        Evaluate the model's performance using specified metrics.

        Args:
            y_test: True labels for evaluation.
            y_pred: Predicted labels for evaluation.

        Returns:
            List of scores for each metric.
        """
        return [metric(y_test, y_pred) for metric in self.metrics]

    def train_test(self, X_train, y_train, X_test, y_test):
        """
        Train the model on training data and evaluate on test data.

        Args:
            X_train: Input features for training.
            y_train: True labels for training.
            X_test: Input features for testing.
            y_test: True labels for testing.

        Returns:
            List of scores for each metric.
        """
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return self.score(y_test, y_pred)

    def checkpoint_save(self, checkpoints_var):
        """
        Save model and associated information to a checkpoint dictionary.

        Args:
            checkpoints_var (dict): Dictionary to store model information.

        Returns:
            Updated checkpoint dictionary.
        """
        checkpoints_var[self.name] = {
            "model": self.model,
            "Theorem": self.theorem,
            "description": self.description,
            "features_version": self.version,
        }
        return checkpoints_var
