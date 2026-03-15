import pandas as pd  # use pandas library to handle CSV file easily
import random  # use random library to get random numbers or shuffle stuff


                                                # Task 1: Extract Word Features with labels from CSV File
def extractWordFeaturesWithLabel():
    data = pd.read_csv("2- Beginner_Reviews_dataset.csv")   # load CSV data with pandas
    dataset = []  # store all data here
    for i in range(len(data)):
        text = str(data["sentence"][i])
        label = data["label"][i]
        words = text.split()  # to split sentence into words
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        items = [(w, freq[w]) for w in freq]
        items.append(label)
        dataset.append([text, items])  # add sentence and its features to the dataset
    return dataset

def extractWordFeatures(text):
    words = text.split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

                                            # Task 2: Split Dataset (80% Training set, 20% Test set)
def split():
    data = pd.read_csv("2- Beginner_Reviews_dataset.csv")
    data = data.sample(frac=1, random_state=1).reset_index(drop=True)
    cut = int(len(data) * 0.8)  # split data into train and test
    train = data[:cut]   # first 80% as training data
    test = data[cut:]    # remaining 20% as test data
    return train, test

                                            # Task 3: Binary Classifier using SGD + Hinge Loss
def convert_to_features(data):
    features = []
    labels = []
    for i in range(len(data)):
        text = str(data["sentence"].iloc[i])
        feats = extractWordFeatures(text)
        features.append(feats)   # add features to the list
        lbl = data["label"].iloc[i]
        if lbl == 0:
            lbl = -1
        labels.append(lbl)
    return features, labels

# train a model using Stochastic Gradient Descent
def sgd_train(train, learning_rate=0.01, epochs=10):
    features, labels = convert_to_features(train)
    weights = {}  # initialize empty weights
    for _ in range(epochs):
        for i in range(len(features)):
            x = features[i]
            y = labels[i]
            score = 0
            for word in x:
                score += weights.get(word, 0) * x[word]    # calculating weighted sum
            if y * score < 1:   # margin small, so model not confident ,and so we'll adjust weights
                for word in x:
                    new_value = weights.get(word, 0) + learning_rate * y * x[word]
                    weights[word] = new_value
    return weights

def predict(weights, features):
    score = sum(weights.get(w, 0) * features.get(w, 0) for w in features)
    return 1 if score >= 0 else -1


def evaluate(weights, df):
    feature_vectors, true_labels = convert_to_features(df)
    correct_predictions = 0

    for i in range(len(feature_vectors)):
        predicted = predict(weights, feature_vectors[i])   # predicted label
        if predicted == true_labels[i]:
            correct_predictions += 1   # count if prediction is correct
    return (correct_predictions / len(feature_vectors)) * 100  # returns accuracy

                                                    # Task 4: Nearest Neighbor Classifier
def distance(vec1, vec2):
    dist = 0
    for w in vec1:
        if w in vec2:
            dist += abs(vec1[w] - vec2[w])
        else:
            dist += vec1[w]   # add word count if only in vec1
    for w in vec2:
        if w not in vec1:
            dist += vec2[w]  # add word count if only in vec2

    return dist # returns total distance

def nearest_Neighbor(train, test):
    correct = 0
    for i in range(len(test)):
        test_vec = extractWordFeatures(str(test["sentence"].iloc[i]))   # get features of test sentence
        actual_label = test["label"].iloc[i]   # actual label of test sentence

        best_label = None
        best_dist = float("inf")

        for j in range(len(train)):
            train_vec = extractWordFeatures(str(train["sentence"].iloc[j]))   # get features of training sentence
            train_label = train["label"].iloc[j]   # label of training sentence

            dist = distance(test_vec, train_vec)
            if dist < best_dist:
                best_dist = dist
                best_label = train_label

        if best_label == actual_label:
            correct += 1
    return (correct / len(test)) * 100  # return accuracy

                                                        # Task 5: Compare Both Classifiers
def compare(binaryAcc, nnAcc):
    print("Binary Classifier Accuracy:", binaryAcc, "%")
    print("Nearest Neighbor Accuracy:", nnAcc, "%")
    if nnAcc > binaryAcc:
        print("Nearest Neighbor performed better.")
    elif binaryAcc > nnAcc:
        print("Binary classifier performed better.")
    else:
        print("Both performed equally good.")

# Main Function
if __name__ == "__main__":
    print("\nStep 1: Loading the dataset...")
    dataset = extractWordFeaturesWithLabel()  # calling extractWordFeaturesWithLabel function

    train, test = split()  # calling splitDataset function
    print("Step 2: splitting the Dataset...")
    print("Number of training examples:", len(train))
    print("Number of test examples    :", len(test))

    print("\nStep 3: Training the Binary Classifier (SGD)...")
    weights = sgd_train(train)  # calling sgd_train function
    train_binary_acc = evaluate(weights, train)  # calling evaluate function for training set
    test_binary_acc = evaluate(weights, test)  # calling evaluate function for test set

    print("\nBinary Classifier Results:")
    print("Training Accuracy:", round(train_binary_acc, 2), "%")
    print("Test Accuracy    :", round(test_binary_acc, 2), "%")

    print("\nStep 4: Running Nearest Neighbor Classifier...")
    nn_acc = nearest_Neighbor(train, test)  # calling nearestNeighbor function
    print("Nearest Neighbor Test Accuracy:", round(nn_acc, 2), "%")

    print("\nStep 5: Comparing both classifiers...")
    compare(test_binary_acc, nn_acc)  # Calling compare function
