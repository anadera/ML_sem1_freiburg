import numpy as np

if __name__ == '__main__':
    age = np.array([1, 42, 13, 25, 63, 15])
    area = np.array([50.73, 41.83, 46.54, 58.27, 72.53, 51.47])
    price = np.array([523902.67, 325104.45, 434919.86, 575719.18, 629274.54, 390576.98])

    X = np.column_stack([age, area])
    y = price

    XtX = np.matmul(X.T, X)
    Xty = np.matmul(X.T, y)

    weights = np.linalg.solve(XtX, Xty)
    w_age, w_area = weights[0], weights[1]

    print("Weights: w_years={}, w_area={}".format(w_age, w_area))

    predict_age = 10
    predict_area = 50
    predict_price = w_age * predict_age + w_area * predict_area

    print("Predicted price: {}".format(predict_price))

    real_price = 427451.10

    e = (predict_price - real_price)
    L2 = e ** 2
    print("L2 error: {}".format(L2))
    L1 = np.abs(e)
    print("L1 error: {}".format(L1))
