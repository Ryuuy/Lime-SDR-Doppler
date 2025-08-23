# -*- coding: utf-8 -*-

import os
import numpy as np

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print("Directory {} created.".format(directory_name))
    else:
        print("Directory {} already exists.".format(directory_name))

def read_binary_file(file_name):
    # 使用 numpy 读取整个二进制文件，并将其解释为 int16 数据
    data = np.fromfile(file_name, dtype=np.int16)
    
    # 将 I 和 Q 数据分开
    I_data = data[::2]  # 每隔两个取 I 数据
    Q_data = data[1::2]  # 每隔两个取 Q 数据
    
    return I_data, Q_data

def save_data_to_txt(I_data, Q_data, output_file):
    with open(output_file, 'w') as f:
        f.write("I Data\tQ Data\n")  # 添加表头
        for i, q in zip(I_data, Q_data):
            f.write("{}\t{}\n".format(i, q))  # 保存 I/Q 数据，用 tab 分隔
    print("Data saved to {}".format(output_file))

if __name__ == "__main__":
    # 创建文件夹
    folder_name = "read_files"
    create_directory(folder_name)

    # 二进制文件名
    binary_file_name = "sdr_data_frames/sdr_data_frame_0.bin"
    
    # 读取整个二进制文件中的 I/Q 数据
    I_data, Q_data = read_binary_file(binary_file_name)

    # 保存数据到 read_files 目录中的 txt 文件
    output_txt_file = os.path.join(folder_name, "sdr_data_frame_1.txt")
    save_data_to_txt(I_data, Q_data, output_txt_file)
