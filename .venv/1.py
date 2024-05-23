import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil


def dctmtx(M):
    """
    Возвращает матрицу ДКП размера MxM.
    """
    T = np.zeros((M, M))
    for p in range(M):
        for q in range(M):
            if p == 0:
                T[p, q] = 1 / np.sqrt(M)
            else:
                T[p, q] = np.sqrt(2 / M) * np.cos(np.pi * (2 * q + 1) * p / (2 * M))
    return T
def dct_matrix(image):
    """
    Выполняет двумерное дискретное косинусное преобразование (ДКП) изображения.
    """
    # Получаем размеры изображения
    height, width = image.shape

    # Создаем матрицу ДКП для строк и столбцов
    T_height = dctmtx(height)
    T_width = dctmtx(width)

    # Выполняем преобразование: T * image * T'
    dct_transformed = np.dot(np.dot(T_height, image), T_width.T)

    return dct_transformed.astype(np.int16)
def idct(dct_coeffs):
    """
    Выполняет обратное двумерное дискретное косинусное преобразование (обратное ДКП).
    """
    # Получаем размеры коэффициентов ДКП
    height, width = dct_coeffs.shape

    # Создаем матрицу ДКП для строк и столбцов
    T_height = dctmtx(height)
    T_width = dctmtx(width)

    # Выполняем обратное преобразование: T' * dct_coeffs * T
    idct_transformed = np.dot(np.dot(T_height.T, dct_coeffs), T_width)

    return idct_transformed.astype(np.int16)

def split_image(image):
    """
    Разделяет изображение на блоки размером 8x8.
    """
    height, width = image.shape
    blocks = []

    for y in range(0, height, 8):
        for x in range(0, width, 8):
            blocks.append(image[y:y+8, x:x+8])
    return np.array(blocks)
def merge_blocks(blocks, image_shape):
    """
    Объединяет блоки обратно в изображение.
    """
    height, width, _ = image_shape

    image = np.zeros((height, width), dtype=int)
    block_idx = 0
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            image[y:y+8, x:x+8] = blocks[block_idx]
            block_idx += 1
    return image

def rle(input_array):
    def append_rle(rle_line, current_char, count):
        if count > 1:
            rle_line.append('▐')
            rle_line.append(chr(count))
        rle_line.append(current_char)

    rle_line = []
    count = 1

    # Преобразуем первый элемент массива в символ
    if input_array[0] < 0:
        current_char = 'ε' + chr(-input_array[0])
    else:
        current_char = chr(input_array[0])

    for i in range(1, len(input_array)):
        if input_array[i] < 0:
            next_char = 'ε' + chr(-input_array[i])
        else:
            next_char = chr(input_array[i])

        if next_char == current_char:
            count += 1
        else:
            append_rle(rle_line, current_char, count)
            current_char = next_char
            count = 1

    # Обрабатываем последнюю последовательность символов
    append_rle(rle_line, current_char, count)

    return ''.join(rle_line)
def rle_decoder(encoded_string):
    decoded_array = []
    i = 0
    length = len(encoded_string)

    while i < length:
        if encoded_string[i] == '▐':
            # Читаем количество повторений
            count = ord(encoded_string[i + 1])
            i += 2
        else:
            count = 1

        if encoded_string[i] == 'ε':
            # Отрицательное число
            number = -ord(encoded_string[i + 1])
            i += 2
        else:
            # Положительное число
            number = ord(encoded_string[i])
            i += 1

        decoded_array.extend([number] * count)

    # Преобразование списка в ndarray
    return np.array(decoded_array)

def get_quantization_matrix(quality):
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    # Проверка, что уровень качества находится в диапазоне от 1 до 100
    if quality < 1 or quality > 100:
        raise ValueError("Уровень качества должен быть в диапазоне от 1 до 100.")

    # Масштабирование базовой матрицы квантования в соответствии с уровнем качества
    if quality<=50:
        scaling_factor=5000 / quality
    else:
        scaling_factor=200-2*quality


    scaled_matrix = np.floor((quantization_matrix * scaling_factor+50)/100)
    scaled_matrix= np.clip(scaled_matrix, 1, 255)

    return scaled_matrix.astype(np.uint8)

def quantization_dct(dct_matrix, quantization_matrix):
    quantized_matrix = np.round(dct_matrix / quantization_matrix)
    return quantized_matrix.astype(int)
def de_quantization_dct(dct_matrix, quantization_matrix):
    quantized_matrix = np.round(dct_matrix * quantization_matrix)
    return quantized_matrix.astype(int)

def rgb_to_ycbcr(mas):
    #создаем отдельные массивы по цветам
    R, G, B = mas[:, :, 0], mas[:, :, 1], mas[:, :, 2]

    #переводим в YCbCr
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0183 * B + 128

    # Объединение массивов Y, Cb и Cr в один трехмерный массив
    ycbcr_image_array = np.dstack((Y, Cb, Cr))
    # Ограничение значений компонент в диапазоне [0, 255]
    ycbcr_image_array = np.clip(ycbcr_image_array, 0, 255)
    # Преобразование массива к типу данных np.uint8 (беззнаковые 8-битные целые числа)
    ycbcr_image_array = ycbcr_image_array.astype(np.uint8)

    return ycbcr_image_array
def ycbcr_to_rgb(Y, Cb, Cr):
    # Разделение входного массива изображения на его цветовые компоненты: Y, Cb и Cr


    #меняем тип данных массивов, чтобы в них можно было хранить отрицательные числа
    Y=Y.astype(np.int16)
    Cb = Cb.astype(np.int16)
    Cr = Cr.astype(np.int16)

    # Вычисление компонент R, G и B в цветовом пространстве RGB
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    # Преобразование массивов R, G и B к типу данных np.uint8 (беззнаковые 8-битные целые числа)
    R = np.uint8(np.clip(R, 0, 255))
    G = np.uint8(np.clip(G, 0, 255))
    B = np.uint8(np.clip(B, 0, 255))

    # Объединение массивов R, G и B в один трехмерный массив
    rgb_image_array = np.dstack((R, G, B))

    return rgb_image_array

def zigzag_2(matrix):
    result = []  # Список для хранения результата
    current_row = 0  # Текущая строка
    current_col_left = 0  # Текущий столбец сверху слева
    current_col_right = 0  # Текущий столбец снизу справа
    direction_counter = 0  # Счётчик направления движения
    row, col = len(matrix), len(matrix[0])  # Количество строк и столбцов в матрице

    # Верхняя половина матрицы
    while (current_row < row):
        if (direction_counter % 2 == 0):  # Движение слева направо
            current_col_left = 0
            while (current_col_left <= current_row and current_col_left < col and current_row - current_col_left >= 0):
                # Добавление элементов по диагонали сверху слева вниз направо
                result.append(matrix[current_row - current_col_left][current_col_left])
                current_col_left += 1
        else:
            # Движение справа налево
            current_col_left = current_row
            current_col_right = 0
            while (current_col_left >= 0 and current_col_left < col and current_col_right <= current_row):
                # Добавление элементов по диагонали снизу справа вверх налево
                result.append(matrix[current_col_right][current_col_left])
                current_col_left -= 1
                current_col_right += 1
        # Переход к следующей строке
        current_row += 1
        direction_counter += 1

    # Ниже главной диагонали
    current_row = 1
    while (current_row < col):
        if (direction_counter % 2 == 0):
            # Движение слева направо
            current_col_left = row - 1
            current_col_right = current_row
            while (current_col_left >= 0 and current_col_right < col):
                # Добавление элементов по диагонали снизу слева вверх направо
                result.append(matrix[current_col_left][current_col_right])
                current_col_left -= 1
                current_col_right += 1
        else:
            # Рекурсивное добавление оставшихся элементов
            reverse(matrix, current_row, row - 1, current_row, col, result)
        direction_counter += 1
        current_row += 1

    return np.array(result)
def reverse(matrix, current_row, row, current_col_right, col, result):

    if (row >= 0 and current_col_right < col):
        reverse(matrix, current_row, row - 1, current_col_right + 1, col, result)
        result.append(matrix[row][current_col_right])

def inverse_zigzag_2(array, rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)  # Создаем пустую матрицу нужного размера
    current_index = 0  # Индекс для элементов входного массива

    # Верхняя половина матрицы
    for current_row in range(rows):
        if current_row % 2 == 0:
            # Движение слева направо
            for i in range(current_row + 1):
                matrix[current_row - i][i] = array[current_index]
                current_index += 1
        else:
            # Движение справа налево
            for i in range(current_row + 1):
                matrix[i][current_row - i] = array[current_index]
                current_index += 1

    # Нижняя половина матрицы
    for current_col in range(1, cols):
        if (rows + current_col) % 2 == 0:
            # Движение слева направо
            for i in range(rows - current_col):
                matrix[rows - 1 - i][current_col + i] = array[current_index]
                current_index += 1
        else:
            # Движение справа налево
            for i in range(rows - current_col):
                matrix[current_col + i][cols - 1 - i] = array[current_index]
                current_index += 1

    return matrix

def zigzag(matrix):
    result = []  # Список для хранения результата
    row, col = len(matrix), len(matrix[0])  # Количество строк и столбцов в матрице
    total = row + col - 1  # Общее количество диагоналей

    for current_sum in range(total):
        if current_sum % 2 == 0:
            for i in range(min(current_sum, row-1), max(-1, current_sum-col), -1):
                result.append(matrix[i][current_sum - i])
        else:
            for i in range(max(0, current_sum-col+1), min(current_sum+1, row)):
                result.append(matrix[i][current_sum - i])

    return result
def inverse_zigzag(array, rows, cols):
    matrix = np.zeros((rows, cols), dtype=int)  # Создаем пустую матрицу нужного размера
    current_index = 0  # Индекс для элементов входного массива
    total = rows + cols - 1  # Общее количество диагоналей

    for current_sum in range(total):
        if current_sum % 2 == 0:
            for i in range(min(current_sum, rows-1), max(-1, current_sum-cols), -1):
                matrix[i][current_sum - i] = array[current_index]
                current_index += 1
        else:
            for i in range(max(0, current_sum-cols+1), min(current_sum+1, rows)):
                matrix[i][current_sum - i] = array[current_index]
                current_index += 1

    return matrix




def downsampling(image, factor_h, factor_w, downsampling_type):
    height, width = image.shape
    # Проверка входных данных

    if downsampling_type==1:

    #рассчитываем количество строк, столбцов, которые необходимо исключить
        height_out, width_out = np.floor(height/factor_h), np.floor(width/factor_w);
        delete_lines_h, delete_lines_w= height- height_out, width-width_out
    #индекс начала
        start_index = 0
    #пока нужно удалить строки
        while delete_lines_h > 0:
            # Находим индексы четных строк, учитывая текущий размер матрицы
            even_indices = list(range(start_index, image.shape[0], 2))
            #count нужен, так как при удалении строки из матрицы индексы сдвигаются
            count_delete=0
            for index in even_indices:
                if delete_lines_h > 0:
                    #удаляем четную стркоу (с учетом удаленных строк)
                    image = np.delete(image, index-count_delete, axis=0)
                    count_delete+=1
                    delete_lines_h -= 1

            # Обновляем start_index, чтобы начать с начала, если достигнут конец матрицы
            start_index = max(0, start_index - image.shape[0])

        start_index = 0
        while delete_lines_w > 0:
            # Находим индексы четных столбцов, учитывая текущий размер матрицы
            even_indices = list(range(start_index, image.shape[1], 2))
            # count нужен, так как при удалении строки из матрицы индексы сдвигаются
            count_delete = 0
            for index in even_indices:
                if delete_lines_w > 0:
                    # удаляем четную стркоу (с учетом удаленных строк)
                    image = np.delete(image, index-count_delete, axis=1)
                    count_delete += 1
                    delete_lines_w -= 1

            # Обновляем start_index, чтобы начать с начала, если достигнут конец матрицы
            start_index = max(0, start_index - image.shape[1])
        return image

    if downsampling_type == 2:

        #считаем новые высоту и длину
        height_out, width_out = int((height / factor_h)), int((width / factor_w))
        #размеры блоков
        block_h, block_w=int(np.floor(height/height_out)), int(np.floor(width / width_out))
        # Создание нового массива для уменьшенного изображения
        downsampling_image = np.zeros((height_out, width_out), dtype=np.uint8)

        # Проход по уменьшенному изображению
        for i in range(height_out):
            for j in range(width_out):
                # Вычисление среднего значения блока (все, что с i - работа со строками, что с j - работа с столбцами)
                # Выделяем блоки
                block = image[i * block_h: (i + 1) * block_h, j * block_w: (j + 1) * block_w]
                #среднее значение
                average = np.mean(block)
                # Запись среднего значения в уменьшенное изображение
                downsampling_image[i, j] = int(average)
        return downsampling_image

    if downsampling_type == 3:
        # считаем новые высоту и длину
        height_out, width_out = int((height / factor_h)), int((width / factor_w))
        # размеры блоков
        block_h, block_w = int(np.floor(height / height_out)), int(np.floor(width / width_out))

        # Создание нового массива для уменьшенного изображения
        downsampling_image = np.zeros((height_out, width_out), dtype=np.uint8)

        # Проход по уменьшенному изображению
        for i in range(height_out):
            for j in range(width_out):
                # Вычисление среднего значения блока (все, что с i - работа со строками, что с j - работа с столбцами)
                # Выделяем блоки
                block = image[i * block_h: (i + 1) * block_h, j * block_w: (j + 1) * block_w]
                # среднее значение
                average = np.mean(block)
                #матрица разницы элементов блока и среднего значения
                diff_matrix=abs(block- average)
                #индекс элемента, разница со средним которого минимальна
                index_of_nearest_pixel = diff_matrix.argmin()
                #находим элемент по индексу
                nearest_pixel = block.flat[index_of_nearest_pixel]
                # Запись среднего значения в уменьшенное изображение
                downsampling_image[i, j] = int(nearest_pixel)
        return downsampling_image
def upsampling_image(image, factor_h, factor_w):
    height, width = image.shape
    # считаем новые высоту и длину
    height_out, width_out = int((height * factor_h)), int((width * factor_w))
    # размеры блоков
    block_h, block_w = int(np.floor(height_out / height)), int(np.floor(width_out / width))
    # Создание нового массива для увеличенного изображения
    upsampling_image = np.zeros((height_out, width_out), dtype=np.uint8)

    # Проход по увеличенному изображению
    for i in range(height):
        for j in range(width):
            # Выделяем блок из исходного изображения
            block = image[i,j]
            # Копируем блок в увеличенное изображение
            upsampling_image[i * factor_h: (i + 1) * factor_h, j * factor_w: (j + 1) * factor_w] = block
    return upsampling_image


def save_image_to_txt(image_array, filename):
    """Сохраняет изображение, представленное массивом NumPy, в текстовый файл. """

    # Открываем файл для записи
    with open(filename, 'w') as file:
        # Перебираем все пиксели в изображении
        for row in image_array:
            for pixel in row:
                # Записываем значения каналов YCbCr в файл, разделяя их пробелами
                file.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
            # Добавляем перенос строки после каждой строки пикселей
            file.write("\n")

def full_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################
    'Записываем массив фото RGB в файл'
    ########################################################################################
    save_image_to_txt(image_array_rgb, '0_RGB_image.txt')
    ########################################################################################

    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################
    'Записываем массив фото YCbCrв файл'
    ########################################################################################
    save_image_to_txt(image_array_ycbcr, '1_YCbCr_image.txt')
    ########################################################################################

    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 10
    compression_w = 10

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################
    'Записываем массивы фото после сабсемплинга в файл'
    ########################################################################################
    ycbcr_sub = np.dstack((Y, Cb, Cr))
    save_image_to_txt(ycbcr_sub, '2_subsempling_image.txt')
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################
    'Записываем массивы фото после ДКТ в файл'
    ########################################################################################
    'Объединяем блоки'
    Y_dct = merge_blocks(coeffs_blocks_Y, image_array_rgb.shape)
    Cb_dct = merge_blocks(coeffs_blocks_Cb, image_array_rgb.shape)
    Cr_dct = merge_blocks(coeffs_blocks_Cr, image_array_rgb.shape)

    'Сохраняем txt'
    coeffs_blocks = np.dstack((Y_dct, Cb_dct, Cr_dct))
    save_image_to_txt(coeffs_blocks, '3_DCT_image.txt')
    ########################################################################################

    print(f"4. Шаг: Квантование матриц c Q=50")
    'Квантование'
    ########################################################################################
    'Ставим качество 50 и создаем матрицу квантования'
    quality = 50
    quantization_matrix_quality = get_quantization_matrix(quality)

    'Квантование'
    quantization_coeffs_Y = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Y])
    quantization_coeffs_Cb = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cb])
    quantization_coeffs_Cr = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cr])
    ########################################################################################
    'Объединение блоков'
    ########################################################################################
    Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
    Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
    Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
    ########################################################################################
    'Записываем массивы фото после Квантования в файл'
    ########################################################################################
    merge_blocks_ = np.dstack((Y_merge, Cb_merge, Cr_merge))
    save_image_to_txt(merge_blocks_, '4_Квантование_image.txt')
    ########################################################################################

    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_merge)
    zigzag_Cb = zigzag(Cb_merge)
    zigzag_Cr = zigzag(Cr_merge)
    ########################################################################################
    'Записываем фото после Зиг Зага в файл'
    ########################################################################################
    zigzag_blocks = np.dstack((zigzag_Y, zigzag_Cb, zigzag_Cr))
    save_image_to_txt(zigzag_blocks, '5_ZigZag_image.txt')
    ########################################################################################

    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################
    'Записываем фото после РЛЕ в файл'
    ########################################################################################
    rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
    with open("6_RLE_code.txt", 'w', encoding='utf-8') as file:
        file.write(rle_finish)
    ########################################################################################
    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################
    'Путь к исходной папке'
    source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

    'Создаем целевую папку, если её нет'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    'Перебираем все файлы в исходной папке'
    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
        if filename.endswith('.txt'):
            # Формируем полные пути к файлу в исходной и целевой папках
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Копируем файл из исходной папки в целевую
            shutil.copy2(source_path, target_path)

    print("Копирование завершено.")

    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым
        if filename.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(source_folder, filename)
            # Удаляем файл
            os.remove(file_path)
    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"3. Шаг: Обратное квантование")
    'Обратное квантование матриц'
    ########################################################################################
    de_quantization_Y = de_quantization_dct(de_blocks_Y, quantization_matrix_quality)
    de_quantization_Cb = de_quantization_dct(de_blocks_Cb, quantization_matrix_quality)
    de_quantization_Cr = de_quantization_dct(de_blocks_Cr, quantization_matrix_quality)
    ########################################################################################

    print(f"4. Шаг: Обратное ДКТ")
    'Обратное квантование матриц'
    ########################################################################################
    de_DCT_Y = np.array([idct(block) for block in de_quantization_Y])
    de_DCT_Cb = np.array([idct(block) for block in de_quantization_Cb])
    de_DCT_Cr = np.array([idct(block) for block in de_quantization_Cr])
    ########################################################################################

    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_DCT_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_DCT_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_DCT_Cr, image_array_rgb.shape)
    ########################################################################################

    print(f"5. Шаг: Декодирование в RGB")
    'Обратное кодирование в RGB'
    ########################################################################################
    rgb_array = ycbcr_to_rgb(de_Y_merge, de_Cb_merge, de_Cr_merge)
    ########################################################################################

    image_rgb_decoder = Image.fromarray(rgb_array, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()

def without_YCbCr_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################
    'Записываем массив фото RGB в файл'
    ########################################################################################
    save_image_to_txt(image_array_rgb, '0_RGB_image.txt')
    ########################################################################################

    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_rgb[:, :, 0], image_array_rgb[:, :, 1], image_array_rgb[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 2
    compression_w = 2

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Y = downsampling(Y, compression_h, compression_w, 2)
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Y = upsampling_image(downsampling_matrix_Y, compression_h, compression_w)
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################
    'Записываем массивы фото после сабсемплинга в файл'
    ########################################################################################
    ycbcr_sub = np.dstack((Y, Cb, Cr))
    save_image_to_txt(ycbcr_sub, '2_subsempling_image.txt')
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################
    'Записываем массивы фото после ДКТ в файл'
    ########################################################################################
    'Объединяем блоки'
    Y_dct = merge_blocks(coeffs_blocks_Y, image_array_rgb.shape)
    Cb_dct = merge_blocks(coeffs_blocks_Cb, image_array_rgb.shape)
    Cr_dct = merge_blocks(coeffs_blocks_Cr, image_array_rgb.shape)

    'Сохраняем txt'
    coeffs_blocks = np.dstack((Y_dct, Cb_dct, Cr_dct))
    save_image_to_txt(coeffs_blocks, '3_DCT_image.txt')
    ########################################################################################

    print(f"4. Шаг: Квантование матриц c Q=50")
    'Квантование'
    ########################################################################################
    'Ставим качество 50 и создаем матрицу квантования'
    quality = 50
    quantization_matrix_quality = get_quantization_matrix(quality)

    'Квантование'
    quantization_coeffs_Y = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Y])
    quantization_coeffs_Cb = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cb])
    quantization_coeffs_Cr = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cr])
    ########################################################################################
    'Объединение блоков'
    ########################################################################################
    Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
    Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
    Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
    ########################################################################################
    'Записываем массивы фото после Квантования в файл'
    ########################################################################################
    merge_blocks_ = np.dstack((Y_merge, Cb_merge, Cr_merge))
    save_image_to_txt(merge_blocks_, '4_Квантование_image.txt')
    ########################################################################################

    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_merge)
    zigzag_Cb = zigzag(Cb_merge)
    zigzag_Cr = zigzag(Cr_merge)
    ########################################################################################
    'Записываем фото после Зиг Зага в файл'
    ########################################################################################
    zigzag_blocks = np.dstack((zigzag_Y, zigzag_Cb, zigzag_Cr))
    save_image_to_txt(zigzag_blocks, '5_ZigZag_image.txt')
    ########################################################################################

    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################
    'Записываем фото после РЛЕ в файл'
    ########################################################################################
    rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
    with open("6_RLE_code.txt", 'w', encoding='utf-8') as file:
        file.write(rle_finish)
    ########################################################################################
    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################
    'Путь к исходной папке'
    source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

    'Создаем целевую папку, если её нет'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    'Перебираем все файлы в исходной папке'
    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
        if filename.endswith('.txt'):
            # Формируем полные пути к файлу в исходной и целевой папках
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Копируем файл из исходной папки в целевую
            shutil.copy2(source_path, target_path)

    print("Копирование завершено.")

    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым
        if filename.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(source_folder, filename)
            # Удаляем файл
            os.remove(file_path)
    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"3. Шаг: Обратное квантование")
    'Обратное квантование матриц'
    ########################################################################################
    de_quantization_Y = de_quantization_dct(de_blocks_Y, quantization_matrix_quality)
    de_quantization_Cb = de_quantization_dct(de_blocks_Cb, quantization_matrix_quality)
    de_quantization_Cr = de_quantization_dct(de_blocks_Cr, quantization_matrix_quality)
    ########################################################################################

    print(f"4. Шаг: Обратное ДКТ")
    'Обратное квантование матриц'
    ########################################################################################
    de_DCT_Y = np.array([idct(block) for block in de_quantization_Y])
    de_DCT_Cb = np.array([idct(block) for block in de_quantization_Cb])
    de_DCT_Cr = np.array([idct(block) for block in de_quantization_Cr])
    ########################################################################################

    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_DCT_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_DCT_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_DCT_Cr, image_array_rgb.shape)
    ########################################################################################


    rgb_array_ = np.dstack((de_Y_merge, de_Cb_merge, de_Cr_merge))
    rgb_array_ = rgb_array_.astype('uint8')
    image_rgb_decoder = Image.fromarray(rgb_array_, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()
def without_Subsempling_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################
    'Записываем массив фото RGB в файл'
    ########################################################################################
    save_image_to_txt(image_array_rgb, '0_RGB_image.txt')
    ########################################################################################

    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################
    'Записываем массив фото YCbCrв файл'
    ########################################################################################
    save_image_to_txt(image_array_ycbcr, '1_YCbCr_image.txt')
    ########################################################################################

    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################


    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################
    'Записываем массивы фото после ДКТ в файл'
    ########################################################################################
    'Объединяем блоки'
    Y_dct = merge_blocks(coeffs_blocks_Y, image_array_rgb.shape)
    Cb_dct = merge_blocks(coeffs_blocks_Cb, image_array_rgb.shape)
    Cr_dct = merge_blocks(coeffs_blocks_Cr, image_array_rgb.shape)

    'Сохраняем txt'
    coeffs_blocks = np.dstack((Y_dct, Cb_dct, Cr_dct))
    save_image_to_txt(coeffs_blocks, '3_DCT_image.txt')
    ########################################################################################

    print(f"4. Шаг: Квантование матриц c Q=50")
    'Квантование'
    ########################################################################################
    'Ставим качество 50 и создаем матрицу квантования'
    quality = 50
    quantization_matrix_quality = get_quantization_matrix(quality)

    'Квантование'
    quantization_coeffs_Y = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Y])
    quantization_coeffs_Cb = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cb])
    quantization_coeffs_Cr = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cr])
    ########################################################################################
    'Объединение блоков'
    ########################################################################################
    Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
    Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
    Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
    ########################################################################################
    'Записываем массивы фото после Квантования в файл'
    ########################################################################################
    merge_blocks_ = np.dstack((Y_merge, Cb_merge, Cr_merge))
    save_image_to_txt(merge_blocks_, '4_Квантование_image.txt')
    ########################################################################################

    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_merge)
    zigzag_Cb = zigzag(Cb_merge)
    zigzag_Cr = zigzag(Cr_merge)
    ########################################################################################
    'Записываем фото после Зиг Зага в файл'
    ########################################################################################
    zigzag_blocks = np.dstack((zigzag_Y, zigzag_Cb, zigzag_Cr))
    save_image_to_txt(zigzag_blocks, '5_ZigZag_image.txt')
    ########################################################################################

    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################
    'Записываем фото после РЛЕ в файл'
    ########################################################################################
    rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
    with open("6_RLE_code.txt", 'w', encoding='utf-8') as file:
        file.write(rle_finish)
    ########################################################################################
    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################
    'Путь к исходной папке'
    source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

    'Создаем целевую папку, если её нет'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    'Перебираем все файлы в исходной папке'
    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
        if filename.endswith('.txt'):
            # Формируем полные пути к файлу в исходной и целевой папках
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Копируем файл из исходной папки в целевую
            shutil.copy2(source_path, target_path)

    print("Копирование завершено.")

    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым
        if filename.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(source_folder, filename)
            # Удаляем файл
            os.remove(file_path)
    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"3. Шаг: Обратное квантование")
    'Обратное квантование матриц'
    ########################################################################################
    de_quantization_Y = de_quantization_dct(de_blocks_Y, quantization_matrix_quality)
    de_quantization_Cb = de_quantization_dct(de_blocks_Cb, quantization_matrix_quality)
    de_quantization_Cr = de_quantization_dct(de_blocks_Cr, quantization_matrix_quality)
    ########################################################################################

    print(f"4. Шаг: Обратное ДКТ")
    'Обратное квантование матриц'
    ########################################################################################
    de_DCT_Y = np.array([idct(block) for block in de_quantization_Y])
    de_DCT_Cb = np.array([idct(block) for block in de_quantization_Cb])
    de_DCT_Cr = np.array([idct(block) for block in de_quantization_Cr])
    ########################################################################################

    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_DCT_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_DCT_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_DCT_Cr, image_array_rgb.shape)
    ########################################################################################

    print(f"5. Шаг: Декодирование в RGB")
    'Обратное кодирование в RGB'
    ########################################################################################
    rgb_array = ycbcr_to_rgb(de_Y_merge, de_Cb_merge, de_Cr_merge)
    ########################################################################################

    image_rgb_decoder = Image.fromarray(rgb_array, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()
def without_DCT_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################
    'Записываем массив фото RGB в файл'
    ########################################################################################
    save_image_to_txt(image_array_rgb, '0_RGB_image.txt')
    ########################################################################################

    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################
    'Записываем массив фото YCbCrв файл'
    ########################################################################################
    save_image_to_txt(image_array_ycbcr, '1_YCbCr_image.txt')
    ########################################################################################

    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 10
    compression_w = 10

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################
    'Записываем массивы фото после сабсемплинга в файл'
    ########################################################################################
    ycbcr_sub = np.dstack((Y, Cb, Cr))
    save_image_to_txt(ycbcr_sub, '2_subsempling_image.txt')
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"4. Шаг: Квантование матриц c Q=50")
    'Квантование'
    ########################################################################################
    'Ставим качество 50 и создаем матрицу квантования'
    quality = 50
    quantization_matrix_quality = get_quantization_matrix(quality)

    'Квантование'
    quantization_coeffs_Y = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in blocks_Y])
    quantization_coeffs_Cb = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in blocks_Cb])
    quantization_coeffs_Cr = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in blocks_Cr])
    ########################################################################################
    'Объединение блоков'
    ########################################################################################
    Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
    Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
    Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
    ########################################################################################
    'Записываем массивы фото после Квантования в файл'
    ########################################################################################
    merge_blocks_ = np.dstack((Y_merge, Cb_merge, Cr_merge))
    save_image_to_txt(merge_blocks_, '4_Квантование_image.txt')
    ########################################################################################

    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_merge)
    zigzag_Cb = zigzag(Cb_merge)
    zigzag_Cr = zigzag(Cr_merge)
    ########################################################################################
    'Записываем фото после Зиг Зага в файл'
    ########################################################################################
    zigzag_blocks = np.dstack((zigzag_Y, zigzag_Cb, zigzag_Cr))
    save_image_to_txt(zigzag_blocks, '5_ZigZag_image.txt')
    ########################################################################################

    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################
    'Записываем фото после РЛЕ в файл'
    ########################################################################################
    rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
    with open("6_RLE_code.txt", 'w', encoding='utf-8') as file:
        file.write(rle_finish)
    ########################################################################################
    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################
    'Путь к исходной папке'
    source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

    'Создаем целевую папку, если её нет'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    'Перебираем все файлы в исходной папке'
    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
        if filename.endswith('.txt'):
            # Формируем полные пути к файлу в исходной и целевой папках
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Копируем файл из исходной папки в целевую
            shutil.copy2(source_path, target_path)

    print("Копирование завершено.")

    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым
        if filename.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(source_folder, filename)
            # Удаляем файл
            os.remove(file_path)
    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"3. Шаг: Обратное квантование")
    'Обратное квантование матриц'
    ########################################################################################
    de_quantization_Y = de_quantization_dct(de_blocks_Y, quantization_matrix_quality)
    de_quantization_Cb = de_quantization_dct(de_blocks_Cb, quantization_matrix_quality)
    de_quantization_Cr = de_quantization_dct(de_blocks_Cr, quantization_matrix_quality)
    ########################################################################################


    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_quantization_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_quantization_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_quantization_Cr, image_array_rgb.shape)
    ########################################################################################

    print(f"5. Шаг: Декодирование в RGB")
    'Обратное кодирование в RGB'
    ########################################################################################
    rgb_array = ycbcr_to_rgb(de_Y_merge, de_Cb_merge, de_Cr_merge)
    ########################################################################################

    image_rgb_decoder = Image.fromarray(rgb_array, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()
def without_quant_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################
    'Записываем массив фото RGB в файл'
    ########################################################################################
    save_image_to_txt(image_array_rgb, '0_RGB_image.txt')
    ########################################################################################

    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################
    'Записываем массив фото YCbCrв файл'
    ########################################################################################
    save_image_to_txt(image_array_ycbcr, '1_YCbCr_image.txt')
    ########################################################################################

    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 10
    compression_w = 10

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################
    'Записываем массивы фото после сабсемплинга в файл'
    ########################################################################################
    ycbcr_sub = np.dstack((Y, Cb, Cr))
    save_image_to_txt(ycbcr_sub, '2_subsempling_image.txt')
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################
    'Записываем массивы фото после ДКТ в файл'
    ########################################################################################
    'Объединяем блоки'
    Y_dct = merge_blocks(coeffs_blocks_Y, image_array_rgb.shape)
    Cb_dct = merge_blocks(coeffs_blocks_Cb, image_array_rgb.shape)
    Cr_dct = merge_blocks(coeffs_blocks_Cr, image_array_rgb.shape)

    'Сохраняем txt'
    coeffs_blocks = np.dstack((Y_dct, Cb_dct, Cr_dct))
    save_image_to_txt(coeffs_blocks, '3_DCT_image.txt')
    ########################################################################################

    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_dct)
    zigzag_Cb = zigzag(Cb_dct)
    zigzag_Cr = zigzag(Cr_dct)
    ########################################################################################
    'Записываем фото после Зиг Зага в файл'
    ########################################################################################
    zigzag_blocks = np.dstack((zigzag_Y, zigzag_Cb, zigzag_Cr))
    save_image_to_txt(zigzag_blocks, '5_ZigZag_image.txt')
    ########################################################################################

    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################
    'Записываем фото после РЛЕ в файл'
    ########################################################################################
    rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
    with open("6_RLE_code.txt", 'w', encoding='utf-8') as file:
        file.write(rle_finish)
    ########################################################################################
    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################
    'Путь к исходной папке'
    source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

    'Создаем целевую папку, если её нет'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    'Перебираем все файлы в исходной папке'
    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
        if filename.endswith('.txt'):
            # Формируем полные пути к файлу в исходной и целевой папках
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # Копируем файл из исходной папки в целевую
            shutil.copy2(source_path, target_path)

    print("Копирование завершено.")

    for filename in os.listdir(source_folder):
        # Проверяем, является ли файл текстовым
        if filename.endswith('.txt'):
            # Формируем полный путь к файлу
            file_path = os.path.join(source_folder, filename)
            # Удаляем файл
            os.remove(file_path)
    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"4. Шаг: Обратное ДКТ")
    'Обратное квантование матриц'
    ########################################################################################
    de_DCT_Y = np.array([idct(block) for block in de_blocks_Y])
    de_DCT_Cb = np.array([idct(block) for block in de_blocks_Cb])
    de_DCT_Cr = np.array([idct(block) for block in de_blocks_Cr])
    ########################################################################################

    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_DCT_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_DCT_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_DCT_Cr, image_array_rgb.shape)
    ########################################################################################

    print(f"5. Шаг: Декодирование в RGB")
    'Обратное кодирование в RGB'
    ########################################################################################
    rgb_array = ycbcr_to_rgb(de_Y_merge, de_Cb_merge, de_Cr_merge)
    ########################################################################################

    image_rgb_decoder = Image.fromarray(rgb_array, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()

def compress_and_decode_image(image_path, target_folder, Q):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################


    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################


    'YCbCr по массивам'
    ########################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 10
    compression_w = 10

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################


    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################

    print(f"4. Шаг: Квантование матриц c Q=50")
    'Квантование'
    ########################################################################################
    'Ставим качество 50 и создаем матрицу квантования'

    quantization_matrix_quality = get_quantization_matrix(Q)

    'Квантование'
    quantization_coeffs_Y = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Y])
    quantization_coeffs_Cb = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cb])
    quantization_coeffs_Cr = np.array(
        [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cr])
    ########################################################################################
    'Объединение блоков'
    ########################################################################################
    Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
    Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
    Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
    ########################################################################################


    print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
    'Зиг заг'
    ########################################################################################
    zigzag_Y = zigzag(Y_merge)
    zigzag_Cb = zigzag(Cb_merge)
    zigzag_Cr = zigzag(Cr_merge)
    ########################################################################################


    print(f"6. Шаг: Сжатие с помощью RLE")
    'RLE'
    ########################################################################################
    rle_Y = rle(zigzag_Y)
    rle_Cb = rle(zigzag_Cb)
    rle_Cr = rle(zigzag_Cr)
    ########################################################################################

    print(f"Сжатие завершено!")

    'Переписываем файлы txt в отдельную папку'
    ########################################################################################

    ########################################################################################

    print(f"_____________________________________________________________________________________________________\n\n")

    print(f"Проверка кодирования декодированием:")
    print(f"1. Шаг: Разжатие RLE")
    'декодер RLE'
    ########################################################################################
    de_rle_Y = rle_decoder(rle_Y)
    de_rle_Cb = rle_decoder(rle_Cb)
    de_rle_Cr = rle_decoder(rle_Cr)
    ########################################################################################

    print(f"2. Шаг: Обрантный зиг заг")
    'декодер RLE'
    ########################################################################################
    de_zigzag_Y = inverse_zigzag(de_rle_Y, height, width)
    de_zigzag_Cb = inverse_zigzag(de_rle_Cb, height, width)
    de_zigzag_Cr = inverse_zigzag(de_rle_Cr, height, width)
    ########################################################################################

    'разделение изображения на блоки 8x8'
    ########################################################################################
    de_blocks_Y = split_image(de_zigzag_Y)
    de_blocks_Cb = split_image(de_zigzag_Cb)
    de_blocks_Cr = split_image(de_zigzag_Cr)
    ########################################################################################

    print(f"3. Шаг: Обратное квантование")
    'Обратное квантование матриц'
    ########################################################################################
    de_quantization_Y = de_quantization_dct(de_blocks_Y, quantization_matrix_quality)
    de_quantization_Cb = de_quantization_dct(de_blocks_Cb, quantization_matrix_quality)
    de_quantization_Cr = de_quantization_dct(de_blocks_Cr, quantization_matrix_quality)
    ########################################################################################

    print(f"4. Шаг: Обратное ДКТ")
    'Обратное квантование матриц'
    ########################################################################################
    de_DCT_Y = np.array([idct(block) for block in de_quantization_Y])
    de_DCT_Cb = np.array([idct(block) for block in de_quantization_Cb])
    de_DCT_Cr = np.array([idct(block) for block in de_quantization_Cr])
    ########################################################################################

    'Объединение блоков матриц'
    ########################################################################################
    de_Y_merge = merge_blocks(de_DCT_Y, image_array_rgb.shape)
    de_Cb_merge = merge_blocks(de_DCT_Cb, image_array_rgb.shape)
    de_Cr_merge = merge_blocks(de_DCT_Cr, image_array_rgb.shape)
    ########################################################################################

    print(f"5. Шаг: Декодирование в RGB")
    'Обратное кодирование в RGB'
    ########################################################################################
    rgb_array = ycbcr_to_rgb(de_Y_merge, de_Cb_merge, de_Cr_merge)
    ########################################################################################

    image_rgb_decoder = Image.fromarray(rgb_array, 'RGB')
    image_rgb_decoder.save('decode_image.jpg')

    # проверка изображений на кодирование
    fig, axs = plt.subplots(1, 2)
    # Отображение изображений на Axes
    axs[0].imshow(image)
    axs[0].set_title("Изначальное изображение")
    axs[1].imshow(image_rgb_decoder)
    axs[1].set_title("Декодированное")
    # Установка расстояния между изображениями
    plt.tight_layout()
    # Отображение Figure
    fig.suptitle("СЖАТИЕ ИЗОБРАЖЕНИЯ")

    plt.show()

def quantization_compress_and_decode_image(image_path, target_folder):
    "основная часть"
    #######################################################################################################################################################################

    'Считывание фото'
    ########################################################################################
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    image_array_rgb = np.array(image_rgb)
    width, height = image.size
    ########################################################################################


    print(f"1. Шаг: Преобразование из RGB в YCbCr")
    'YCbCr'
    ########################################################################################
    'массив в формате YCbCr'
    image_array_ycbcr = rgb_to_ycbcr(image_array_rgb)
    ########################################################################################

    'YCbCr по массивам'
    #######################################################################################
    Y, Cb, Cr = image_array_ycbcr[:, :, 0], image_array_ycbcr[:, :, 1], image_array_ycbcr[:, :, 2]
    ########################################################################################

    print(f"2. Шаг: Сабсемплинг матриц")
    'сабсемплинг'
    ########################################################################################
    "коэф. сжатия"
    compression_h = 10
    compression_w = 10

    "даунсемплинг Cb Cr 2 методом даунсемплинга"
    downsampling_matrix_Cb = downsampling(Cb, compression_h, compression_w, 2)
    downsampling_matrix_Cr = downsampling(Cr, compression_h, compression_w, 2)

    "апсемплинг Cb Cr, чтобы вернуть к размеру Y"
    Cb = upsampling_image(downsampling_matrix_Cb, compression_h, compression_w)
    Cr = upsampling_image(downsampling_matrix_Cr, compression_h, compression_w)
    ########################################################################################


    'разделение изображения на блоки 8x8'
    ########################################################################################
    blocks_Y = split_image(Y)
    blocks_Cb = split_image(Cb)
    blocks_Cr = split_image(Cr)
    ########################################################################################

    print(f"3. Шаг: Применение ДКТ матрицей")
    'ДКТ матрицей'
    ########################################################################################

    'применяем дкт'
    coeffs_blocks_Y = np.array([dct_matrix(block) for block in blocks_Y])
    coeffs_blocks_Cb = np.array([dct_matrix(block) for block in blocks_Cb])
    coeffs_blocks_Cr = np.array([dct_matrix(block) for block in blocks_Cr])
    ########################################################################################

    quality = 95
    for i in range(0, 18):

        print(f"4. Шаг: Квантование матриц c Q={quality}", )
        'Квантование'
        ########################################################################################
        'Ставим качество 50 и создаем матрицу квантования'
        quantization_matrix_quality = get_quantization_matrix(quality)

        'Квантование'
        quantization_coeffs_Y = np.array(
            [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Y])
        quantization_coeffs_Cb = np.array(
            [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cb])
        quantization_coeffs_Cr = np.array(
            [quantization_dct(block, quantization_matrix_quality) for block in coeffs_blocks_Cr])
        ########################################################################################
        'Объединение блоков'
        ########################################################################################
        Y_merge = merge_blocks(quantization_coeffs_Y, image_array_rgb.shape)
        Cb_merge = merge_blocks(quantization_coeffs_Cb, image_array_rgb.shape)
        Cr_merge = merge_blocks(quantization_coeffs_Cr, image_array_rgb.shape)
        ########################################################################################

        print(f"5. Шаг: Преобразоваие матриц в строку с помощью обхода зиг-загом")
        'Зиг заг'
        ########################################################################################
        zigzag_Y = zigzag(Y_merge)
        zigzag_Cb = zigzag(Cb_merge)
        zigzag_Cr = zigzag(Cr_merge)
        ########################################################################################

        print(f"6. Шаг: Сжатие с помощью RLE")
        'RLE'
        ########################################################################################
        rle_Y = rle(zigzag_Y)
        rle_Cb = rle(zigzag_Cb)
        rle_Cr = rle(zigzag_Cr)
        ########################################################################################
        'Записываем фото после РЛЕ в файл'
        ########################################################################################
        rle_finish = rle_Y + "ք" + rle_Cb + "ք" + rle_Cr
        name = f"RLE_{quality}_code.txt"
        with open(name, 'w', encoding='utf-8') as file:
            file.write(rle_finish)
        ########################################################################################
        print(f"Сжатие завершено!")
        quality -= 5
        'Переписываем файлы txt в отдельную папку'
        ########################################################################################
        'Путь к исходной папке'
        source_folder = 'C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv'

        'Создаем целевую папку, если её нет'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        'Перебираем все файлы в исходной папке'
        for filename in os.listdir(source_folder):
            # Проверяем, является ли файл текстовым файлом (например, имеет ли он расширение .txt)
            if filename.endswith('.txt'):
                # Формируем полные пути к файлу в исходной и целевой папках
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)

                # Копируем файл из исходной папки в целевую
                shutil.copy2(source_path, target_path)

        print("Копирование завершено.")

        for filename in os.listdir(source_folder):
            # Проверяем, является ли файл текстовым
            if filename.endswith('.txt'):
                # Формируем полный путь к файлу
                file_path = os.path.join(source_folder, filename)
                # Удаляем файл
                os.remove(file_path)
    ########################################################################################




def dct_check(image_path):
    'проверка работы ДКТ с помощью матрицы'
    #######################################################################################################################################################################
    """
    ------------------------------------------------------------------------------------------------
    Время выполнения dct_2: 4 секунд
    Время выполнения dct: 207 секунд
    ДКТ с помощью матрицы работает быстрее обычного ДКТ в: 47 раз
    ------------------------------------------------------------------------------------------------
    """
    print(f"Проверка работы ДКТ с помощью матрицы")
    print(f"------------------------------------------------------------------------------------------------")

    #считываем изображение
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")

    # создаем массив RGB
    image_array_rgb = np.array(image_rgb)

    #массивы по цветам
    R, G, B = image_array_rgb[:, :, 0], image_array_rgb[:, :, 1], image_array_rgb[:, :, 2]

    #разделение изображения на блоки 8x8
    blocks_R = split_image(R)
    blocks_G = split_image(G)
    blocks_B = split_image(B)

    #засекаем время для ДКТ матрицей
    start_time = time.time()
    #применяем дкт
    coeffs_blocks_R = np.array([dct_matrix(block) for block in blocks_R])
    coeffs_blocks_G = np.array([dct_matrix(block) for block in blocks_G])
    coeffs_blocks_B = np.array([dct_matrix(block) for block in blocks_B])

    #декодируем дкт
    de_coeffs_blocks_R = np.array([idct(block) for block in coeffs_blocks_R])
    de_coeffs_blocks_G = np.array([idct(block) for block in coeffs_blocks_G])
    de_coeffs_blocks_B = np.array([idct(block) for block in coeffs_blocks_B])
    dct_matrix_time = time.time() - start_time


    #засекаем время для ДКТ
    start_time = time.time()

    #применяем ДКТ для всех болков всех цветов
    coeffs_blocks_R_1 = np.array([dct(block) for block in blocks_R])
    coeffs_blocks_G_1 = np.array([dct(block) for block in blocks_G])
    coeffs_blocks_B_1 = np.array([dct(block) for block in blocks_B])

    #применяем декодер ДКТ для всех болков всех цветов
    de_coeffs_blocks_R_ = np.array([revers_dct(block) for block in coeffs_blocks_R_1])
    de_coeffs_blocks_G_ = np.array([revers_dct(block) for block in coeffs_blocks_G_1])
    de_coeffs_blocks_B_ = np.array([revers_dct(block) for block in coeffs_blocks_B_1])

    dct_time = time.time() - start_time


    #объединение всех блоков в массивы цветов
    R_=merge_blocks(de_coeffs_blocks_R, image_array_rgb.shape)
    G_=merge_blocks(de_coeffs_blocks_G, image_array_rgb.shape)
    B_=merge_blocks(de_coeffs_blocks_B, image_array_rgb.shape)

    #объединение массивов цветов в изображение
    rgb_image_array_ = np.dstack((R_, G_, B_))
    rgb_image_array_=rgb_image_array_.astype(np.uint8)
    image_rgb_ = Image.fromarray(rgb_image_array_, 'RGB')


    #объединение всех блоков в массивы цветов
    R__=merge_blocks(de_coeffs_blocks_R_, image_array_rgb.shape)
    G__=merge_blocks(de_coeffs_blocks_G_, image_array_rgb.shape)
    B__=merge_blocks(de_coeffs_blocks_B_, image_array_rgb.shape)

    #объединение массивов цветов в изображение
    rgb_image_array__ = np.dstack((R__, G__, B__))
    rgb_image_array__=rgb_image_array__.astype(np.uint8)
    image_rgb__ = Image.fromarray(rgb_image_array__, 'RGB')

    #изображение ДКТ
    image_rgb__.show()
    #изображение ДКТ матрицей
    image_rgb_.show()

    print(f"Время выполнения dct_matrix: {int(dct_matrix_time)} секунд")
    print(f"Время выполнения dct: {int(dct_time)} секунд")

    print(f"ДКТ с помощью матрицы работает быстрее обычного ДКТ в: {int(dct_time/dct_matrix_time)} раз")
    print(f"------------------------------------------------------------------------------------------------")
    #######################################################################################################################################################################


if __name__ == "__main__":

    # image_array = np.random.randint(0, 256, (1600, 1600, 3), dtype=np.uint8)
    # # Создаем изображение из массива
    # image = Image.fromarray(image_array, 'RGB')
    # # Сохраняем изображение в файл
    # image.save('1600_random_image.jpg')
    #
    # # Определяем цвет (например, зеленый)
    #color = (0, 0, 150)  # RGB: (R, G, B)
     # Создаем изображение размером 1600x1600 с указанным цветом
    #image = Image.new('RGB', (1600, 1600), color)
     # Сохраняем изображение в файл
    #image.save('1600_solid_image.jpg')
    # dct_check("800.jpg")

    # compress_and_decode_image("1600.jpg", "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_image_1600",90)
    # compress_and_decode_image("1600.jpg", "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_image_1600",
    #                           80)
    # compress_and_decode_image("1600.jpg", "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_random_image_1600",60)
    # compress_and_decode_image("1600.jpg", "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_solid_image_1600",40)
    # compress_and_decode_image("1600.jpg", "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_image_1600",
    #                           20)
    # compress_and_decode_image("1600.jpg",
    #                           "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_full_code_txt/txt_random_image_1600", 10)




    # without_YCbCr_compress_and_decode_image("1600.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_YCbCr_code_txt/txt_1600")
    #without_YCbCr_compress_and_decode_image("1600_random_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_YCbCr_code_txt/txt_random_image_1600")
    # without_YCbCr_compress_and_decode_image("800_solid_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_YCbCr_code_txt/txt_solid_image_800")



    # without_Subsempling_compress_and_decode_image("1600.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_Subsempling_code_txt/txt_1600")
    # without_Subsempling_compress_and_decode_image("1600_random_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_Subsempling_code_txt/txt_random_image_1600")
    # without_Subsempling_compress_and_decode_image("800_solid_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_Subsempling_code_txt/txt_solid_image_800")


    # without_DCT_compress_and_decode_image("1600.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_DCT_code_txt/txt_1600")
    # without_DCT_compress_and_decode_image("1600_random_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_DCT_code_txt/txt_random_image_1600")
    # without_DCT_compress_and_decode_image("800_solid_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_DCT_code_txt/txt_solid_image_800")



    # without_quant_compress_and_decode_image("1600.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_quant_code_txt/txt_1600")
    # without_quant_compress_and_decode_image("1600_random_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_quant_code_txt/txt_random_image_1600")
    # without_quant_compress_and_decode_image("800_solid_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/0_without_quant_code_txt/txt_solid_image_800")
    x=0

    # quantization_compress_and_decode_image("1600.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/1_quant_code_txt/txt_1600")
    #
    # quantization_compress_and_decode_image("1600_random_image.jpg",
    #                                         "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/1_quant_code_txt/txt_random_image_1600")
    #
    # quantization_compress_and_decode_image("800_solid_image.jpg",
    #                                          "C:/labs/aisd/4 семестр/2 лаба/Отчет/.venv/1_quant_code_txt/txt_solid_image_800")


#######################################################################################################################################################################