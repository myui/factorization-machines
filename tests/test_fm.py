import rustpy_fm

def test_prediction():
    k = 3
    n_bins = 100
    learning_rate = 0.01
    epochs = 100

    data = [
        [("feature1", 1.0), ("feature3", 1.0)],
        [("feature1", 1.0), ("feature2", 1.0)],
        [("feature2", 1.0), ("feature3", 1.0)],
        [("feature1", 1.0), ("feature2", 1.0), ("feature3", 1.0)],
    ]

    target = [1.0, 0.0, 1.0, 0.0]

    fm = rustpy_fm.create_factorization_machine(k, n_bins)
    fm.train(data, target, learning_rate, epochs)

    test_sample = [("feature1", 1.0), ("feature3", 1.0)]
    prediction = fm.predict(test_sample)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    test_prediction()