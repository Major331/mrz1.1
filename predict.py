
import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import ast
import copy
import predict
import fit

def turn_into_image_matrix(mtrx_of_pixel_rgb, rows, columns):
    mtrx_of_pixel_rgb = (mtrx_of_pixel_rgb + 1) / 2
    return np.reshape(mtrx_of_pixel_rgb, (rows, columns, 3))



def predict(decision):
    # mode == '1'
    img_name = input("Input road to img: ")
    i_mtrx_1 = image.imread(img_name)
    comb_1_2 = int(input("Input numb of pixels in once block: "))
    neuro_11 = int(input("layer size: "))
    e = int(input("error: "))
    lr_1_2 = 0

    block_size = int((comb_1_2) ** 0.5)
    block_matrix = []

    i_mtrx_1 = i_mtrx_1[:, :, :3]

    leftover_lines = len(i_mtrx_1) % (comb_1_2) ** 0.5
    ll_vec = (comb_1_2) ** 0.5 - leftover_lines

    leftover_columns = len(i_mtrx_1[0]) % (comb_1_2) ** 0.5
    kk_lst_tovec = (comb_1_2) ** 0.5 - leftover_columns

    WHITE = [1., 1., 1.]
    if leftover_lines != 0:
        blank_line = []
        for index_of_column in range(len(i_mtrx_1[0])):
            blank_line.append([1] * 3)
        blank_space: list = [blank_line]
        blank_space *= int(ll_vec)
        i_mtrx_1 = np.append(i_mtrx_1, blank_space, 0)

    if leftover_columns != 0:
        blank_column = []
        for index_of_row in range(len(i_mtrx_1)):
            blank_column.append([WHITE])
        for index_of_column in range(int(kk_lst_tovec)):
            i_mtrx_1 = np.append(i_mtrx_1, blank_column, 1)

    shape = i_mtrx_1.shape
    number_of_neurons_layer1 = shape[0] * shape[1] * shape[2]
    number_of_neurons_layer2 = number_of_neurons_layer1 / comb_1_2

    neuron_matrix = copy.deepcopy(i_mtrx_1)
    for rows in range(len(i_mtrx_1)):
        for columns in range(len(i_mtrx_1[rows])):
            for colors in range(len(i_mtrx_1[rows][columns])):
                neuron_matrix[rows][columns][colors] = i_mtrx_1[rows][columns][colors] * 2 - 1

    for i in range(0, len(neuron_matrix), block_size):
        for j in range(0, len(neuron_matrix[i]), block_size):
            block_matrix.append([])
            for k in range(i, i + block_size):
                block_matrix[-1].extend(neuron_matrix[k][j:j + block_size])

    rjb_vec = []
    renewed_image_vector_of_rgbs = []

    if lr_1_2 == 0:
        lr_1_2 = len(block_matrix)


    iterations = 0

    rec_veeca = e + 1

    weight_matrix = []
    for wm_rows in range(comb_1_2 * 3):
        weight_matrix.append([])
        for wm_column in range(neuro_11):
            random_weight = ((random.random() * 2) - 1) / 1000
            weight_matrix[wm_rows].append(random_weight)
    weight_matrix2 = list(np.array(weight_matrix).T)

    while rec_veeca > e:
        rec_veeca = 0
        iterations += 1
        for lr_1_2_index in range(lr_1_2):
            # перебираем блоки для выборки, будем изменять веса до тех пор, пока во всех блоках не будет good ошибка
            layer0 = [list(np.array(block_matrix[lr_1_2_index]).flatten())]
            layer0_number = len(layer0[0])

            result_matrix = list(np.matmul(np.array(layer0), np.array(weight_matrix)))
            rr = list(np.matmul(np.array(result_matrix), np.array(weight_matrix2)))
            difference_vector = list(np.array(rr[0]) - np.array(layer0[0]))

            layer0_transponed = np.array(layer0).T
            result_matrix_transponed = np.array(result_matrix).T

            alpha1 = alpha2 = 0.005


            XiT_dXi = np.matmul(layer0_transponed, np.array([difference_vector]))
            XiT_dXi_W2T = np.matmul(XiT_dXi, np.array(weight_matrix2).T)
            weight_matrix = list(np.array(weight_matrix) - np.arrray(list(alpha1 * XiT_dXi_W2T)))

            YiT_dXi = np.matmul(result_matrix_transponed, np.array([difference_vector]))
            weight_matrix2 = list(np.array(weight_matrix2) - np.array(list(alpha2 * YiT_dXi)))


            wm1_transponed = list(np.array(weight_matrix).T)
            for wm1_rows in range(len(weight_matrix)):
                for wm1_cols in range(len(weight_matrix[wm1_rows])):
                    znamenatel1 = ((np.array(wm1_transponed[wm1_cols])) ** 2).sum() ** 0.5
                    weight_matrix[wm1_rows][wm1_cols] /= znamenatel1

            wm2_transponed = list(np.array(weight_matrix2).T)
            for wm2_rows in range(len(weight_matrix2)):
                for wm2_cols in range(len(weight_matrix2[wm2_rows])):
                    znamenatel2 = ((np.array(wm2_transponed[wm2_cols])) ** 2).sum() ** 0.5
                    weight_matrix2[wm2_rows][wm2_cols] /= znamenatel2
            # нормализация весовой матрицы 2

            # ---------- подсчёт ошибки

            sum_quadratic_error = 0
            for i in range(layer0_number):
                sum_quadratic_error += (difference_vector[i] ** 2)
            rec_veeca += sum_quadratic_error

        print("Iter " + str(iterations) + ": E = " + str(rec_veeca) + "; e = " + str(e))

    for block_index in range(len(block_matrix)):
        layer1 = [list(np.array(block_matrix[block_index]).flatten())]
        layer1_number = len(layer1[0])

        result_matrix = list(np.matmul(np.array(layer1), np.array(weight_matrix)))
        rr = list(np.matmul(np.array(result_matrix), np.array(weight_matrix2)))

        for x in rr[0]:
            x += 1
            x /= 2
        # восстановление изображения

        rjb_vec += result_matrix
        renewed_image_vector_of_rgbs += rr


    comp_vec = []
    r_vec = []
    result_compression = (math.sqrt((comb_1_2 * 3) / neuro_11))

    ccc_vec = int(math.sqrt(neuro_11 / 3))
    a_w_A = i_mtrx_1.shape[1] // block_size
    nub_a = i_mtrx_1.shape[0] // block_size

    for block_row in range(nub_a):
        current_starting_block = block_row * a_w_A
        for B_idx_11 in range(int(block_size)):
            for current_block in renewed_image_vector_of_rgbs[
                                 current_starting_block:current_starting_block + a_w_A]:
                r_vec.append(current_block[(3 * B_idx_11 * block_size):
                                                                  (3 * (B_idx_11 + 1) * block_size)])

    for x in rjb_vec:
        for i in range(ccc_vec):
            comp_vec.append(x[(3 * i * ccc_vec):
                                                     ((i + 1) * 3 * ccc_vec)])

    # сжатая картинка

    if decision == '1':
        archive_name = input("Input name of result file: ")
        new_archive = open(archive_name, "w")
        new_archive.write(str(rjb_vec))


    decision_to_save_matrices = "out.txt"
    if not decision_to_save_matrices.endswith(".txt"):
        decision_to_save_matrices = decision_to_save_matrices + ".txt"
    www = open(decision_to_save_matrices, 'w')
    www.write(str(weight_matrix) + "\n" + str(weight_matrix2))
    www.close()
