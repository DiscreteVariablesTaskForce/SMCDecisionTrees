from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier


def decision_forest(X_train, y_train, num_trees=1):
    forest = []
    fitting = []

    # Create Decision Tree classifer object
    for i in range(num_trees):
        forest.append(DecisionTreeClassifier())

    # Train Decision Tree Classifer
    for tree in forest:
        fitting.append(tree.fit(X_train, y_train))

    return fitting
