import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class digitOCR:
    def __init__(self):
        self.model = MLPClassifier((100, 100), activation='logistic', solver='lbfgs')
        samples = np.loadtxt('damagedigits.data')
        responses = np.loadtxt('damagelabels.data')

        # responses = responses.reshape((samples.shape[0],1))


        # X_train, X_test, y_train, y_test = train_test_split(samples, responses, test_size=0.33, random_state=42, stratify=responses)

        # self.model.fit(X_train, y_train)
        self.model.fit(samples, responses)
        # y_pred = model.predict(X_test)
        # print(accuracy_score(y_test, y_pred))
        # print(confusion_matrix(y_test,y_pred))
    def predict(self, x):
        return self.model.predict(x)