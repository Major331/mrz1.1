import numpy as np
import lyrely as lr
import os

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values


def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        memory["A" + str(idx)] = A_prev

    return A_curr, memory


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):

    Y_hat_ =[]
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]

    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def start(lst, x, y, q, w1, w2):
    lr.add_Daaw(lst, x, y, q, w1, w2)
    y = np.array(y)
    x = np.array(x)
    context = np.random.rand(x.shape[0], lst[4])
    x = np.concatenate((x, context), axis=1)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    w1 = np.random.rand(5, 2)
    w2 = np.random.rand(2, 1)

    learn_weights(lst, x, y,q,  w1, w2)

    print(result_processing(lst, x, y,q,  w1, w2))


def result_processing(lst, x, y,q,  w1, w2):
    k = y[-1].reshape(1)
    X = x[-1, 0, :-lst[4]]
    out = []
    X = X[1:]
    train = np.concatenate((X, k))
    train = np.append(train, np.array([0] * lst[4]))
    w_ltr = np.matmul(train, w1)
    output = np.matmul(w_ltr, w2)
    out.append(output[0])
    return out


def ww_w(dy, i, w_ltr, lst, x, y,q,  w1, w2):
    w1 -= lst[5] * dy * np.matmul(x[i].transpose(),
                                  w2.transpose()) * lr.der_f(
        np.matmul(x[i], w1))
    w2 -= lst[5] * dy * w_ltr.transpose() * lr.der_f(
        np.matmul(w_ltr, w2))


def learn_weights(lst, x, y,q,  w1, w2):
    j = 0
    while j < lst[3]:
        error_all = 0
        i = 0
        x[:, :, -lst[4]:] = 0
        print(np.matmul(x[i], w1))
        w_ltr = lr.act_func(np.matmul(x[i], w1))

        output = lr.act_func(np.matmul(w_ltr, w2))
        dy = output - y[i]
        ww_w(dy, i, w_ltr, lst, x, y,q,  w1, w2)
        w_ltr = np.matmul(x[i], w1)
        output = np.matmul(w_ltr, w2)
        dy = output - y[i]
        error_all += (dy ** 2)[0]
        j += 1
        i += 1
        print(j, " ", error_all[0])
        if error_all <= lst[2]:
            print("last error = ", error_all)
            break


def main():
    print('Введите последовательность:')
    input_sequence = [int(x) for x in input().split()]
    print('Введите ошибку:')
    error = float(input())
    lst_1 = [input_sequence, 3, error, 500000, 2, 0.000015, 1]

    lst = []
    for i in range(7):
        lst.append(lst_1[i])
    x = []
    y = []
    q = len(lst[0])
    w1 = None
    w2 = None
    res = start(lst, x, y, q, w1, w2)
    print(res)


if __name__ == "__main__":
    main()
