import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import ast
import copy

def turn_into_image_matrix(mtrx_of_pixel_rgb, rows, columns):
    mtrx_of_pixel_rgb = (mtrx_of_pixel_rgb + 1) / 2
    return np.reshape(mtrx_of_pixel_rgb, (rows, columns, 3))



def fit(i_mtrx_1, neuro_11, lr_1_2):
    weights_input_file_name = input("Impoer matrix (name of file): ")
    decision = input("1 - to archive\n2 - from archive ")

    img_input_to_transform = input("Input file that you want archive: ")
    www = open(weights_input_file_name, "r")

    weight_matrix = ast.literal_eval(www.readline())
    weight_matrix2 = ast.literal_eval(www.readline())

    comb_1_2 = len(weight_matrix2[0]) / 3

    if decision != '1':
        archive = open(img_input_to_transform, "r")
        compressed_image = ast.literal_eval(archive.readline())

        renewed_image_vector_of_rgbs = []
        for neuron_block in range(len(compressed_image)):
            rr = np.matmul([compressed_image[neuron_block]], weight_matrix2)
            renewed_image_vector_of_rgbs += rr
            print('train')

        r_vec = []
        result_compression = (((comb_1_2 * 3) / neuro_11) ** 0.5)
        block_size = int((comb_1_2) ** 0.5)

        ccc_vec = int((neuro_11 / 3) ** 0.5)
        a_w_A = i_mtrx_1.shape[1] // block_size
        nub_a = i_mtrx_1.shape[0] // block_size

        for block_row in range(nub_a):
            current_starting_block = block_row * a_w_A
            for B_idx_11 in range(int(block_size)):
                for current_block in renewed_image_vector_of_rgbs[
                                     current_starting_block:current_starting_block + a_w_A]:
                    r_vec.append(current_block[(3 * B_idx_11 * block_size):
                                                                      (3 * (B_idx_11 + 1) * block_size)])

        new_image = turn_into_image_matrix(list(np.array(r_vec).flatten()),
                                           i_mtrx_1.shape[0], i_mtrx_1.shape[1])
        # plt.show()
        plt.savefig(img_input_to_transform[:-4] + ".png")
        exit()

    i_mtrx_1 = image.imread(img_input_to_transform)

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

        if int(block_index / len(block_matrix) * 100) % 10 == 0 and block_index % 100 == 0:
            print(int(block_index / len(block_matrix) * 100), "%")

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


    if decision == '1':
        archive_name = input("Name of result file: ")
        new_archive = open(archive_name, "w")
        new_archive.write(str(rjb_vec))
