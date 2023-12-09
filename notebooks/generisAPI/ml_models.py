class model:
    def __init__(self, name, model, metrics, theorem, description, version):
        self.name = name
        try:
            self.model = model(random_state=0)
        except:
            self.model = model()
        self.metrics = metrics
        self.theorem = theorem
        self.description = description
        self.version = version

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, y_test, y_pred):
        return [metric(y_test, y_pred) for metric in self.metrics]

    def train_test(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return self.score(y_test, y_pred)

    def checkpoint_save(self, checkpoints_var):
        checkpoints_var[self.name] = {
            "model": self.model,
            "Theorem": self.theorem,
            "description": self.description,
            "features_version": self.version,
        }
        return checkpoints_var
