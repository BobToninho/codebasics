import numpy as np


def gradient_descent(x, y):
    m_curr = b_curr = 0
    learning_rate = 0.08
    num_iterations = 10000
    n = len(x)

    for i in range(num_iterations):
        y_predicted = m_curr * x + b_curr
        md = -(2 / n) * np.sum(x * (y - y_predicted))
        bd = -(2 / n) * np.sum(y - y_predicted)
        cost = (1 / n) * sum(val**2 for val in (y - y_predicted))
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        # print("m {}, b {}, iteration {}, cost {}".format(m_curr, b_curr, i, cost))

    return m_curr, b_curr, cost


def main():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])
    gradient_descent(x, y)
    pass


if __name__ == "__main__":
    main()
