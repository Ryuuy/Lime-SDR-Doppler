# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft
from scipy.signal import windows,stft,gaussian,convolve2d
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os
from matplotlib.animation import FFMpegWriter,FuncAnimation
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from collections import defaultdict
import finufft
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from numpy.fft import fft, ifft, fftshift, fftfreq
from numpy.linalg import inv, LinAlgError
from matplotlib import ticker as mticker
def count_continuous_segments(arr):
    """
    统计数组中连续为 1 和连续为 0 的子数组长度及其出现次数。
    在指定范围内过滤长度为 21、20、8 和 9 的段，打印过滤后的段位置。
    
    Args:
        arr (list): 输入的二进制数组（0 和 1）。
    
    Returns:
        tuple: 包含连续 1 和连续 0 子数组长度统计的两个数组。
    """
    n = len(arr)
    ones_lengths = []
    zeros_lengths = []
    segments = []  # 保存段的起始位置和长度
    current_count = 1

    # 遍历数组统计连续段，仅处理指定范围
    for i in range(1, n):  # 控制索引范围
        if arr[i] == arr[i - 1]:
            current_count += 1
        else:
            # 保存当前段的起始位置和长度
            segments.append((i - current_count, current_count, arr[i - 1]))
            if arr[i - 1] == 1:
                ones_lengths.append(current_count)
            else:
                zeros_lengths.append(current_count)
            current_count = 1

    # 最后一段处理
    segments.append((n - current_count, current_count, arr[n - 101]))

    # 过滤掉指定长度的段
    filtered_segments = [seg for seg in segments if seg[1] not in {21, 20, 8, 9}]
    print(f"过滤后的段位置和长度: {[(start, length) for start, length, value in filtered_segments]}")

    # 统计每个长度出现的次数
    ones_count = Counter(ones_lengths)
    zeros_count = Counter(zeros_lengths)

    # 转换为数组表示
    max_ones_length = max(ones_count.keys(), default=0)
    max_zeros_length = max(zeros_count.keys(), default=0)
    array_of_1 = [ones_count.get(i, 0) for i in range(max_ones_length + 1)]
    array_of_0 = [zeros_count.get(i, 0) for i in range(max_zeros_length + 1)]

    return array_of_1, array_of_0


def process_files(folder_name):
    """
    frame_length每个频率的长度frame
    """
    bin_files = [
        os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.bin')
    ]

    # 按中心频率和帧编号排序
    sorted_files = sorted(
        bin_files,
        key=lambda x: (
            int(x.split("center")[1].split("frame")[0]),  # 提取中心频率
            int(x.split("frame")[1].split(".bin")[0])     # 提取时间帧编号
        )
    )

    # 提取频率和帧编号
    frequency_groups = {}
    for file in sorted_files:
        base_name = os.path.basename(file)
        center_freq = int(base_name.split("center")[1].split("frame")[0])
        frame_number = int(base_name.split("frame")[1].split(".bin")[0])
        
        if center_freq not in frequency_groups:
            frequency_groups[center_freq] = []
        frequency_groups[center_freq].append(frame_number)

    # 检测最短频率段的帧长度（检测断层）
    gap_length = None
    for freq, frames in frequency_groups.items():
        sorted_frames = sorted(frames)
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] > 1:  # 发现断层
                gap_length = sorted_frames[i - 1] + 1  # 检测的断层点
                break
        if gap_length is not None:
            frame_length = gap_length  # 每组帧长度为断层点
            break  # 只需要检测一次断层
        else:
            frame_length = len(sorted_frames)

    # 计算帧长度和每组的分组数量
    
    frequency_frame_groups = [
        len(frames) // frame_length for frames in frequency_groups.values()  # 每个频率的分组数量
    ]

    # 返回结果
    center_freqs = sorted(frequency_groups.keys())  # 中心频率数组
    return sorted_files, center_freqs, frame_length, frequency_frame_groups

def process_filesIntimedomain(folder_name):
    bin_files = [
        os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.bin')
    ]

    # 按帧编号排序（仅根据 frame 排序）
    sorted_files = sorted(
        bin_files,
        key=lambda x: int(x.split("frame")[1].split(".bin")[0])  # 提取时间帧编号
    )

    # 提取帧编号和中心频率（按时间域顺序分组）
    time_domain_groups = {}
    for file in sorted_files:
        base_name = os.path.basename(file)
        center_freq = int(base_name.split("center")[1].split("frame")[0])
        frame_number = int(base_name.split("frame")[1].split(".bin")[0])

        if center_freq not in time_domain_groups:
            time_domain_groups[center_freq] = []
        time_domain_groups[center_freq].append(frame_number)

    # 检测最短频率段的帧长度（检测断层）
    gap_length = None
    frame_length = 0
    for freq, frames in time_domain_groups.items():
        sorted_frames = sorted(frames)
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] > 1:  # 发现断层
                gap_length = sorted_frames[i - 1] + 1  # 检测的断层点
                break
        if gap_length is not None:
            frame_length = gap_length  # 每组帧长度为断层点
            break  # 只需要检测一次断层
        else:
            frame_length = len(sorted_frames)

    # 计算帧长度和每组的分组数量
    frequency_frame_groups = [
        len(frames) // frame_length for frames in time_domain_groups.values()  # 每个频率的分组数量
    ]

    # 返回结果
    center_freqs = sorted(time_domain_groups.keys())  # 中心频率数组
    return sorted_files, center_freqs, frame_length, frequency_frame_groups

def process_files_fast(folder_name):
    bin_files = [
        os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.bin')
    ]

    # 使用正则表达式快速提取 center 和 frame（如果需要更灵活）
    sorted_files = sorted(
        bin_files,
        key=lambda x: (
            int(x.split("center")[1].split("frame")[0]),  # 提取中心频率
            int(x.split("frame")[1].split(".bin")[0])     # 提取时间帧
        )
    )

    # 打印排序结果
    
    return sorted_files
def widerdiffertime(folder_name, FFT_Calibration, sampling_rate):
    # 获取所有 .bin 文件路径和频率信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_files(folder_name)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长 0.8V 12bit ADC
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001
    
    # 构造完整的频率轴数组 (超长数组)
    freq_axis = []
    for i, center_freq in enumerate(center_freqs):
        start_freq = center_freq * 1e6 - sampling_rate / 2
        end_freq = center_freq * 1e6 + sampling_rate / 2
        freq_axis.append(np.linspace(start_freq, end_freq, FFTLength, endpoint=False))
    freq_axis = np.concatenate(freq_axis)

    # 初始化存储 dBm 数据的矩阵
    dbm_matrix = np.zeros((frame_length * frequency_frame_groups[0], len(center_freqs), FFTLength))

    # 遍历每个中心频率和时间帧，填充 dBm 数据矩阵
    index = 0  # 文件索引
    total_files = len(sorted_files)
    for k, center_freq in enumerate(center_freqs):
        for frame in range(frame_length * frequency_frame_groups[k]):
            bin_file_path = sorted_files[index]
            index += 1  # 更新索引

            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)

            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data

            # 进行 FFT 并应用校准
            fft_result = fft(complex_signal)
            if FFT_Calibration[0] != 0:
                fft_result[0] = fft_result[0] - FFT_Calibration[0]
            fft_result = fftshift(fft_result)
            fft_magnitude = np.abs(fft_result)

            # 计算 dBm 值
            fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
            # max_value = np.max(fft_dbm)  # 获取最大值
            # max_index = np.argmax(fft_dbm)  # 获取最大值对应的索引
            # print(f"最大 dBm 值: {max_value:.2f} dBm, 位置索引: {max_index}")
            # 存储到矩阵
            dbm_matrix[frame, k, :] = fft_dbm

        print(f"Frequency band {k + 1}/{len(center_freqs)} processed ({index}/{total_files} files).")

    # 构造时间轴
    time_axis = np.arange(1, frame_length * frequency_frame_groups[0] + 1) * FFTLength / sampling_rate

    print(f"All {total_files} frames processed for {len(center_freqs)} frequency bands.")

    # 根据校准生成文件名
    base_folder_name = os.path.basename(folder_name)
    output_dir = 'picture'
    os.makedirs(output_dir, exist_ok=True)

    if all(calib == 0 for calib in FFT_Calibration):  # 无校准
        file_suffix = ".png"
    else:  # 有校准
        file_suffix = "withDCcalibration.png"

    output_file = os.path.join(output_dir, f"{base_folder_name}{file_suffix}")

    # 绘制结果
    plot_3d_dbm(dbm_matrix, frame_length * frequency_frame_groups[0], time_axis, FFTLength, freq_axis[::-1], output_file)

    return dbm_matrix, freq_axis, time_axis

def plot_3d_dbm(dbm_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file):
    # 展开矩阵，用于绘图
    dbm_matrix_expanded = dbm_matrix.transpose(1, 2, 0).reshape(len(freq_axis), FrameNumber)

    # 检查维度匹配R
    if dbm_matrix_expanded.shape != (len(freq_axis), len(time_axis)):
        raise ValueError(
            f"Matrix shape {dbm_matrix_expanded.shape} does not match freq_axis ({len(freq_axis)}) "
            f"and time_axis ({len(time_axis)})."
        )

    freq_axis_file = "freq_axis_output.txt"
    with open(freq_axis_file, "w") as f:
        for freq in freq_axis:
            f.write(f"{freq}\n")
    print(f"Frequency axis saved to {freq_axis_file}")

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})  # 全局字体大小设置

    # 绘制 3D dBm 图像
    plt.figure(figsize=(10, 6))  # 调整图像尺寸
    plt.imshow(
        dbm_matrix_expanded, aspect='auto', cmap='RdYlBu_r', vmin=-70, vmax=-30,
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    )
    plt.colorbar(label='dBm', pad=0.02, shrink=0.8)  # 调整颜色条的位置和大小
    
    plt.title('dBm Plot on 25.1Hz 55Mhz bandwidth', fontsize=20)  # 加大标题字体
    plt.ylabel('Frequency (MHz)', fontsize=18)  # 加大Y轴标签字体
    plt.xlabel('Time (s)', fontsize=18)  # 加大X轴标签字体
    plt.xticks(fontsize=16)  # 调整X轴刻度字体
    plt.yticks(fontsize=16)  # 调整Y轴刻度字体

    # 调整图像布局以减少留白
    plt.tight_layout()

    # 保存绘图
    plt.savefig(output_file, bbox_inches='tight')  # 去掉多余留白
    print(f"Plot saved to {output_file}")
    plt.close()

def plot_2d_amp(dbm_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file):
    # 展开矩阵，用于绘图
    dbm_matrix_expanded = dbm_matrix.transpose(1, 2, 0).reshape(len(freq_axis), FrameNumber)

    # 检查维度匹配
    if dbm_matrix_expanded.shape != (len(freq_axis), len(time_axis)):
        raise ValueError(
            f"Matrix shape {dbm_matrix_expanded.shape} does not match freq_axis ({len(freq_axis)}) "
            f"and time_axis ({len(time_axis)})."
        )

    freq_axis_file = "freq_axis_output.txt"
    with open(freq_axis_file, "w") as f:
        for freq in freq_axis:
            f.write(f"{freq}\n")
    print(f"Frequency axis saved to {freq_axis_file}")

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})  # 全局字体大小设置
    # 绘制 3D dBm 图像
    plt.figure(figsize=(10, 6))  # 调整图像尺寸
    plt.imshow(
        dbm_matrix_expanded, aspect='auto', cmap='plasma', vmin=0, vmax=1.2,
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    )
    plt.colorbar(label='multi', pad=0.02, shrink=0.8)  # 调整颜色条的位置和大小
    plt.title('', fontsize=20)  # 加大标题字体
    plt.ylabel('Frequency (MHz)', fontsize=18)  # 加大Y轴标签字体
    plt.xlabel('Time (s)', fontsize=18)  # 加大X轴标签字体
    plt.xticks(fontsize=16)  # 调整X轴刻度字体
    plt.yticks(fontsize=16)  # 调整Y轴刻度字体

    # 调整图像布局以减少留白
    plt.tight_layout()

    # 保存绘图
    plt.savefig(output_file, bbox_inches='tight')  # 去掉多余留白
    print(f"Plot saved to {output_file}")
    plt.close()

def plot_3d_amp(dbm_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file, title_base=""):
    """
    修改版：绘制格式更优的3D信道响应图。
    - 动态设置标题，移除 "Channel Response"。
    - 将频率轴(Y轴)单位从Hz转换为MHz，并优化刻度显示。
    
    Args:
        dbm_matrix (np.ndarray): 信道响应矩阵
        FrameNumber (int): 帧数
        time_axis (np.ndarray): 时间轴
        FFTLength (int): FFT长度
        freq_axis (np.ndarray): 频率轴 (单位: Hz)
        output_file (str): 输出文件路径
        title_base (str): 用于生成标题的基础名称 (例如: 文件夹名)
    """
    # 展开矩阵用于绘图
    dbm_matrix_expanded = dbm_matrix.transpose(1, 2, 0).reshape(len(freq_axis), FrameNumber)

    # 确保维度匹配
    if dbm_matrix_expanded.shape != (len(freq_axis), len(time_axis)):
        raise ValueError(
            f"矩阵形状 {dbm_matrix_expanded.shape} 与坐标轴长度不匹配 "
            f"(freq: {len(freq_axis)}, time: {len(time_axis)})."
        )

    # --- 核心修改点 ---
    # 1. 将频率轴从 Hz 转换为 MHz
    freq_axis_mhz = freq_axis / 1e6

    # 2. 使用转换后的频率轴生成网格数据
    T, F_mhz = np.meshgrid(time_axis, freq_axis_mhz)

    # 创建 3D 图像
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置振幅范围在0到10之间
    dbm_matrix_clipped = np.clip(dbm_matrix_expanded, 0, 10)

    # 绘制 3D 曲面图
    surf = ax.plot_surface(T, F_mhz, dbm_matrix_clipped, cmap='plasma', edgecolor='none', vmin=0, vmax=1.2)

    # 添加颜色条并强制范围
    cbar = fig.colorbar(surf, ax=ax, shrink=0.3, aspect=10, label='Amplitude')
    cbar.mappable.set_clim(0, 1.2)

    # --- 核心修改点 ---
    # 3. 设置新的、更简洁的坐标轴标签和标题
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (MHz)', fontsize=12) # 标签仍为MHz
    ax.set_zlabel('Amplitude', fontsize=12)
    #ax.set_title(f'3D Amplitude - {title_base}', fontsize=16) # 标题不再写死

    # 4. 优化Y轴（频率）的刻度，使其不那么密集
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))

    # 调整视角
    ax.view_init(elev=30, azim=225)

    # 保存图片
    plt.savefig(output_file, bbox_inches='tight')
    print(f"3D Plot saved to {output_file}")
    plt.show() # 在脚本中通常建议注释掉show()，避免程序阻塞
    plt.close(fig)


def Average_getdata_ampplot(folder_name, sampling_rate):
    # 调用 process_filesIntimedomain 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)
    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)
    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001
    # 初始化存储每帧平均 dBm 的数组
    avg_values = []
    # 获取文件夹的基本名称
    base_folder_name = os.path.basename(os.path.normpath(folder_name))
    # 创建目标目录
    output_dir = os.path.join("picture", "avgamp")
    os.makedirs(output_dir, exist_ok=True)
    N = 1
    # 遍历每个 bin 文件
    with tqdm(total=len(sorted_files[3000:]), desc="Processing files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files[3000:]):
            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)
            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data
            for i in range(N):
                # 进行 FFT
                fft_result = fft(complex_signal)
            # 如果 DC 校准非零，则应用校准
                fft_result = fftshift(fft_result)
                fft_magnitude = np.abs(fft_result)  # 保留整个频谱
                avg_amp = np.mean(fft_magnitude)
                # 计算 dBm 振幅
                # 计算 dBm 值
                avg_values.append(avg_amp)
            # 更新进度条
            progress_bar.update(1)
    # 根据 FFTLength 和 sampling_rate 计算时间轴
    time_axis = np.arange(len(avg_values)) * FFTLength / sampling_rate  # 每帧时间间隔
    # 绘制平均 dBm 的变化图
    print("Finished processing all files.")

    plt.figure(figsize=(12, 8))
    plt.plot(time_axis, avg_values, '-', linewidth=2)
    plt.title('Average dBm per Frame', fontsize=30, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=28, fontweight='bold')
    plt.ylabel('Average dBm', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, linestyle='--', linewidth=1)

    plt.show()  # 这将显示图形窗口

    return avg_values

def process_bin_files_to_video(folder_name, frame_rate, sampling_rate, max_frames=None, DCcalibration=0):
    # 调用 process_filesIntimedomain 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)

    # 如果 max_frames 提供了，就限制处理的文件数量
    if max_frames is not None:
        sorted_files = sorted_files[:max_frames]

    # 创建目标目录
    output_dir = "video"
    os.makedirs(output_dir, exist_ok=True)

    # 动态生成视频文件名
    base_folder_name = os.path.basename(os.path.dirname(folder_name.rstrip("/")))  # 提取 wifi24
    channel_name = os.path.basename(folder_name.rstrip("/"))  # 提取 Channel1

    if DCcalibration == 0:
        # 没有 DC 校准
        video_name = f"{base_folder_name}_{channel_name}.mp4"
    else:
        # 有 DC 校准
        video_name = f"{base_folder_name}_{channel_name}_withDCcalibration.mp4"

    # 构造视频文件路径
    video_path = os.path.join(output_dir, video_name)

    # 创建视频写入对象
    metadata = dict(title='FFT Video', artist='Matplotlib', comment='FFT visualization')
    writer = FFMpegWriter(fps=frame_rate, metadata=metadata)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # 创建频率轴数组 (X 轴)
    freq_axis = []
    for center_freq in center_freqs:
        start_freq = center_freq * 1e6 - sampling_rate / 2
        end_freq = center_freq * 1e6 + sampling_rate / 2
        freq_axis.append(np.linspace(start_freq, end_freq, FFTLength, endpoint=False))
    freq_axis = np.concatenate(freq_axis)  # 合并成完整的超长数组

    # 设置图形窗口大小
    fig, ax = plt.subplots(figsize=(6.94, 5.2))  # 窗口尺寸与 MATLAB 代码一致
    ax.set_ylim([-100, 0])  # 设置 dBm 范围
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (dBm)')

    # 遍历文件并写入视频
    with tqdm(total=len(sorted_files), desc="Processing files") as progress_bar, writer.saving(fig, video_path, dpi=100):
        for idx, bin_file_path in enumerate(sorted_files):
            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)

            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data - (DCcalibration)

            # 进行完整的 FFT
            fft_result = fft(complex_signal)

            # 如果 DC 校准非零，则应用校准
            fft_result = fftshift(fft_result)
            fft_magnitude = np.abs(fft_result)  # 保留整个频谱

            # 计算 dBm 振幅
            fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)

            # 绘制 FFT 结果
            ax.clear()
            ax.plot(freq_axis, fft_dbm)
            ax.set_ylim([-100, 0])  # 设置 dBm 范围

            # 计算时间并更新标题
            time = idx * FFTLength / sampling_rate  # 时间戳计算
            ax.set_title(f'time={time:.6f}s')  # 标题为时间

            # 设置轴标签
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude (dBm)')

            # 将帧写入视频
            writer.grab_frame()

            # 在 tqdm 中更新进度
            progress_bar.set_postfix_str(f"Frame {idx + 1}/{len(sorted_files)}")
            progress_bar.update(1)

    print(f"Video saved to {video_path}")

def STFT_normalized(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, 
                    Gaussian=False, interpolation_factor=1, freq_range=(-7.5e6, 7.5e6), normalize=False):
    """
    Perform Short-Time Fourier Transform (STFT) on a complex signal with optional amplitude normalization.

    Parameters:
    - complex_signal: ndarray, the complex signal (I + jQ)
    - Start_point: int, starting index of the analysis window
    - Start_diff: int, step size for moving the window
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each window for FFT
    - Gaussian: bool, whether to apply a Gaussian window in the time domain
    - interpolation_factor: int, factor to interpolate the frequency spectrum for smoothness
    - freq_range: tuple, frequency range to display (min_freq, max_freq)
    - normalize: bool, whether to normalize dBm values (default=False)

    Returns:
    - dbm_results_smoothed: list of (optionally normalized) dBm values for each window
    """

    # Constants for dBm calculation
    adc_step = 0.000195313  # ADC step size (0.8V range, 12-bit ADC)
    resistance = 50         # Ohm
    FFTLength = resolution // 2  # Effective FFT length
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # Initialize results container
    dbm_results = []
    time_axis = []  # To track time points

    # Generate Gaussian window if needed
    if Gaussian:
        std_dev = resolution / 4  # Standard deviation (adjustable)
        gaussian_window = gaussian(resolution, std=std_dev)
        gaussian_window /= np.max(gaussian_window)  # Normalize window

    # Ensure valid index range
    num_windows = (Final_point - Start_point) // Start_diff
    for i in range(num_windows):
        start_idx = Start_point + i * Start_diff
        end_idx = start_idx + resolution

        # Stop if the window exceeds Final_point
        if end_idx > Final_point or end_idx > len(complex_signal):
            break

        # Extract the current window
        window_signal = complex_signal[start_idx:end_idx].copy()

        # Apply Gaussian window in time domain if enabled
        if Gaussian:
            window_signal *= gaussian_window

        # Perform FFT with optional interpolation
        fft_result = fft(window_signal, n=resolution * interpolation_factor)  
        fft_result = fftshift(fft_result)
        fft_magnitude = np.abs(fft_result)

        # Calculate dBm values
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
        dbm_results.append(fft_dbm)

        # Record time point
        time_point = start_idx / sample_rate
        time_axis.append(time_point)

    # Ensure all dBm results have the same length
    max_length = max(len(dbm) for dbm in dbm_results)
    dbm_results_padded = [np.pad(dbm, (0, max_length - len(dbm)), constant_values=np.nan) for dbm in dbm_results]

    # Generate frequency axis correctly
    freq_axis = np.fft.fftshift(np.fft.fftfreq(resolution * interpolation_factor, d=1/sample_rate))

    # Apply frequency range filtering
    min_freq, max_freq = freq_range
    freq_mask = (freq_axis >= min_freq) & (freq_axis <= max_freq)
    freq_axis_filtered = freq_axis[freq_mask]
    dbm_results_filtered = [dbm[freq_mask] for dbm in dbm_results_padded]

    # Perform optional amplitude normalization for each time frame
    dbm_results_final = []
    for dbm_frame in dbm_results_filtered:
        if normalize:
            dbm_frame -= np.nanmax(dbm_frame)  # Normalize max to 0 dB
        dbm_results_final.append(dbm_frame)

    # Convert to numpy array
    dbm_results_final = np.array(dbm_results_final)

    # Apply Gaussian smoothing to improve visualization
    dbm_results_smoothed = gaussian_filter(dbm_results_final, sigma=(1, 1))  # Smooth in both time & freq

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.imshow(
        dbm_results_smoothed.T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], min_freq, max_freq],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label='Power (dB' + (' Normalized' if normalize else '') + ')')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('STFT' + (' with Normalization' if normalize else '') + (' and Gaussian Window' if Gaussian else ''))
    plt.grid()
    plt.show()

    return dbm_results_smoothed

def STFT(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, 
         Gaussian=False, interpolation_factor=4, freq_range=None):
    """
    Perform Short-Time Fourier Transform (STFT) on a complex signal.

    Parameters:
    - complex_signal: ndarray, the complex signal (I + jQ)
    - Start_point: int, starting index of the analysis window
    - Start_diff: int, step size for moving the window
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each window for FFT
    - Gaussian: bool, whether to apply a Gaussian window in the time domain
    - interpolation_factor: int, factor to interpolate the frequency spectrum for smoothness
    - freq_range: tuple or None, frequency range to display (min_freq, max_freq). 
                  If None, display full spectrum.

    Returns:
    - dbm_results: list of dBm values for each window
    """
    # Constants for dBm calculation
    adc_step = 0.000195313  # ADC step size (0.8V range, 12-bit ADC)
    resistance = 50         # Ohm
    FFTLength = resolution // 2  # Effective FFT length
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # Initialize results container
    dbm_results = []
    time_axis = []  # To track time points

    # Generate Gaussian window if needed
    if Gaussian:
        std_dev = resolution / 4  # Standard deviation (adjustable, typically resolution/8)
        gaussian_window = gaussian(resolution, std=std_dev)
        gaussian_window /= np.max(gaussian_window)

    # Generate full frequency axis BEFORE STFT processing
    freq_axis = np.linspace(-sample_rate / 2, sample_rate / 2, resolution * interpolation_factor, endpoint=False)

    print("freqrange", sample_rate / 2, len(freq_axis))

    # **计算 `freq_range` 在 `freq_axis` 中的索引**
    if freq_range is not None:
        min_freq, max_freq = freq_range

        # **检查 freq_range 是否超出 freq_axis**
        if min_freq < np.min(freq_axis) or max_freq > np.max(freq_axis):
            raise ValueError(f"freq_range {freq_range} 超出可用频率范围 [{np.min(freq_axis)}, {np.max(freq_axis)}]")

        startidx = np.searchsorted(freq_axis, min_freq)
        finishidx = np.searchsorted(freq_axis, max_freq)
    else:
        # 如果 `freq_range=None`，则使用整个频谱
        startidx = 0
        finishidx = len(freq_axis)

    # Iteratively process each window
    for i in range(len(complex_signal)):
        start_idx = Start_point + i * Start_diff
        end_idx = start_idx + resolution

        # Stop if the window exceeds Final_point
        if end_idx > Final_point:
            break

        # Extract the current window
        window_signal = complex_signal[start_idx:end_idx].copy()

        # Apply Gaussian window in time domain if enabled
        if Gaussian:
            window_signal *= gaussian_window

        # Perform FFT
        fft_result = fft(window_signal, n=resolution * interpolation_factor)  # Interpolated FFT
        fft_result = fftshift(fft_result)
        fft_magnitude = np.abs(fft_result)

        # Calculate dBm values
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
        dbm_results.append(fft_dbm)  # **完整保存 STFT 结果（频率 × 时间）**

        # Record time point
        time_point = start_idx / sample_rate
        time_axis.append(time_point)

    # **转换 `dbm_results` 为 numpy 数组**
    dbm_results = np.array(dbm_results).T  # 变成 `频率 × 时间` 形状

    # **截取 freq_range 对应的 STFT 结果**
    dbm_results_filtered = dbm_results[startidx:finishidx, :]
    freq_axis_filtered = freq_axis[startidx:finishidx]
    print("STFT",freq_axis_filtered)

    # 绘制 STFT 结果
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        dbm_results_filtered,  # **截取后的 STFT 结果**
        extent=[time_axis[0], time_axis[-1], freq_axis_filtered[0], freq_axis_filtered[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet',
        vmin=-140,  # 直接在 imshow 里设置颜色范围
        vmax=-60
    )
    cbar = plt.colorbar(im, label='Channel Strength (dB)')

    # Adjust the tick label size for both axes
    plt.tick_params(axis='both', which='major', labelsize=20)  # Axis labels

    # Increase the font size of colorbar ticks
    cbar.ax.tick_params(labelsize=20)  # Adjust '14' to your preferred size

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Short-Time Fourier Transform (STFT)' + (' with Gaussian Window' if Gaussian else ''))
    plt.grid()
    plt.show()

    return dbm_results_filtered

# def nolinearSTFT(timeaxis, complexsignal, resolution, freq_range=None, output_filename_base=None):
#     """
#     Perform Non-Uniform Short-Time Fourier Transform (NU-STFT) and save the plot.

#     Parameters:
#     - ... (其他参数保持不变)
#     - output_filename_base: str or None, the base name for the output image file.
#     """
#     K_FFT = 0.000195313 ** 2 / 50 / 0.001
#     dbm_results = []
#     time_axis = []
    
#     total_time = timeaxis[-1] - timeaxis[0]
#     total_time_point = len(timeaxis)
#     samplerate = total_time_point / total_time
#     freq_axis = np.linspace(-samplerate / 2, samplerate / 2, resolution)

#     if freq_range is not None:
#         min_freq, max_freq = freq_range
#         if min_freq < np.min(freq_axis) or max_freq > np.max(freq_axis):
#             raise ValueError(f"freq_range {freq_range} 超出可用频率范围 [{np.min(freq_axis)}, {np.max(freq_axis)}]")
#         startidx = np.searchsorted(freq_axis, min_freq)
#         finishidx = np.searchsorted(freq_axis, max_freq)
#     else:
#         startidx = 0
#         finishidx = len(freq_axis)

#     for i in range(len(timeaxis)):
#         start_idx = i
#         end_idx = start_idx + resolution
#         if end_idx > len(timeaxis):
#             break
#         window_time = timeaxis[start_idx:end_idx]
#         window_signal = complexsignal[start_idx:end_idx]
#         normalized_time = (window_time - window_time[0]) / (window_time[-1] - window_time[0]) * 2 * np.pi
#         spectrum_nufft = finufft.nufft1d1(normalized_time, window_signal.astype(np.complex128), resolution)
#         fft_magnitude = np.abs(spectrum_nufft)
#         fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
#         fft_dbm_filtered = fft_dbm[startidx:finishidx]
#         dbm_results.append(fft_dbm_filtered)
#         time_axis.append(timeaxis[start_idx])

#     dbm_results = np.array(dbm_results).T
#     dbm_results = dbm_results[::-1, :]
    
#     plt.figure(figsize=(12, 8))
#     time_axis = np.array(time_axis)
#     plt.pcolormesh(time_axis, freq_axis[startidx:finishidx], dbm_results, shading='auto', cmap='jet', vmin=-80, vmax=-50)
    
#     cbar = plt.colorbar(label='Channel Strength (dB)')
#     cbar.ax.tick_params(labelsize=24)
#     plt.xlabel('Time (s)', fontsize=20)
#     plt.ylabel('Doppler Frequency (Hz)', fontsize=20)
#     plt.title('Non-Uniform Short-Time Fourier Transform (NU-STFT)', fontsize=18)
#     plt.xticks(fontsize=24)
#     plt.yticks(fontsize=24)
#     plt.grid()

#     # --- 新增的保存图片逻辑 ---
#     if output_filename_base:
#         # 定义保存路径
#         save_dir = 'picture/doppler/'
#         # 如果文件夹不存在，则创建它
#         os.makedirs(save_dir, exist_ok=True)
#         # 拼接完整的文件路径
#         save_path = os.path.join(save_dir, f"{output_filename_base}.png")
#         # 保存图片
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"图片已保存至: {save_path}")
#     # --- 逻辑结束 ---

#     #plt.show()
#     #plt.close() # 保存并显示后关闭图形，防止内存泄漏

#     return dbm_results


import numpy as np
import finufft
import matplotlib.pyplot as plt
import os

def calculate_global_doppler_spectrum(timeaxis, complexsignal, freq_range=None, output_filename_base=None, time_stretch_factor=1.7):
    """
    计算并绘制非均匀采样信号的全局多普勒频谱（修正版）。

    该函数对整个信号执行一次NUFFT，生成一个“多普勒频率 vs 功率”的二维折线图。
    此版本包含两项重要更新：
    1.  引入 time_stretch_factor 来缩放频率轴，实现对多普勒中心的“变焦”。
    2.  修正了NUFFT的变换方向，使多普勒频率与物理直觉一致。

    Args:
        timeaxis (np.ndarray): 信号采样点的非均匀时间戳数组 (单位: 秒)。
        complexsignal (np.ndarray): 输入的复数信号数组。
        freq_range (tuple, optional): 一个元组 (min_freq, max_freq)，用于缩放绘图的频率轴。默认为 None。
        output_filename_base (str, optional): 用于保存绘图文件的基础文件名。默认为 None。
        time_stretch_factor (float, optional): 时间轴拉伸因子。大于1会压缩频率范围（放大中心），
                                               小于1会扩展频率范围。默认为 1.0 (不拉伸)。

    Returns:
        tuple: 一个元组，包含:
            - freq_axis_selected (np.ndarray): 绘制出的频谱对应的频率轴。
            - spectrum_dbm_selected (np.ndarray): 计算出的频谱功率 (单位: dBm)。
    """
    # --- 1. 参数和常量定义 ---
    if len(complexsignal) < 2 or len(timeaxis) < 2:
        print("Warning: Signal is too short for spectrum analysis.")
        return np.array([]), np.array([])
        
    K_FFT = 0.000195313 ** 2 / 50 / 0.001
    resolution = len(complexsignal)

    timeaxis_for_calc = np.copy(timeaxis) * time_stretch_factor
    
    # --- 2. 计算频率轴 ---
    total_time = timeaxis_for_calc[-1] - timeaxis_for_calc[0]
    if total_time <= 0:
        print("Warning: Total time duration is zero or negative. Cannot calculate spectrum.")
        return np.array([]), np.array([])
        
    total_points = len(timeaxis_for_calc)
    equivalent_samplerate = total_points / total_time
    
    freq_axis = np.linspace(-equivalent_samplerate / 2, equivalent_samplerate / 2, resolution)

    # --- 3. 对整个信号执行一次性 NUFFT ---
    time_span = timeaxis_for_calc[-1] - timeaxis_for_calc[0]
    
    if time_span == 0:
        normalized_time = np.zeros_like(timeaxis_for_calc)
    else:
        normalized_time = (timeaxis_for_calc - timeaxis_for_calc[0]) / time_span * -2 * np.pi

    spectrum_nufft = finufft.nufft1d1(normalized_time, complexsignal.astype(np.complex128), resolution)
    
    fft_magnitude = np.abs(spectrum_nufft)
    spectrum_dbm = 10 * np.log10(K_FFT * (fft_magnitude ** 2) + 1e-20)

    # --- 4. 根据 freq_range 筛选数据 ---
    if freq_range is not None:
        min_freq, max_freq = freq_range
        if min_freq >= max_freq:
             raise ValueError(f"freq_range的最小值 {min_freq} 必须小于最大值 {max_freq}")
        
        start_idx = np.searchsorted(freq_axis, min_freq, side='left')
        finish_idx = np.searchsorted(freq_axis, max_freq, side='right')
        
        freq_axis_selected = freq_axis[start_idx:finish_idx]
        spectrum_dbm_selected = spectrum_dbm[start_idx:finish_idx]
    else:
        freq_axis_selected = freq_axis
        spectrum_dbm_selected = spectrum_dbm

    # --- 5. 绘制二维频谱图 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(freq_axis_selected, spectrum_dbm_selected)
    
    ax.axvline(x=2.615, color='r', linestyle='--')
    
    y_lim = ax.get_ylim()
    ax.text(2.615, y_lim[1] * 0.95, ' 2.615 Hz', color='r', fontsize=16, verticalalignment='top')
    
    ax.set_xlabel('Doppler Frequency (Hz)', fontsize=22)
    ax.set_ylabel('Channel Strength (dB)', fontsize=22)
    
    ax.tick_params(axis='both', labelsize=18)
    
    ax.grid(True)
    
    # ★★★ 关键修正：在保存和显示前，调用此函数来自动调整布局，防止标签被截断 ★★★
    fig.tight_layout()
    
    if output_filename_base:
        save_dir = 'picture/doppler_spectrum/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{output_filename_base}_spectrum_stretch_{time_stretch_factor}.png")
        # 保持 bbox_inches='tight' 也是一个好习惯，它会裁掉多余的白边
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f"频谱图已保存至: {save_path}")

    plt.show()
    plt.close(fig)

    return freq_axis_selected, spectrum_dbm_selected


def nolinearSTFT(timeaxis, complexsignal, resolution, freq_range=None, output_filename_base=None, time_stretch_factor=1.7, Big_picture=True):
    """
    此版本根据您的要求进行了修改：
    1. 新增 Big_picture 参数，默认为 False。
    2. 当 Big_picture 设置为 True 时，会显著增大所有坐标轴、标题和颜色条上文字的字号，以适应PPT演示。
    """
    
    # --- 根据 Big_picture 参数设置字体和图像大小 ---
    if Big_picture:
        print("--- 启动大图模式 (Big Picture Mode) ---")
        # 适用于PPT投影的“夸张”字号
        tick_size = 30
        cbar_label_size = 32
        axis_label_size = 36
        figsize = (12, 8) # 同时也可以适当增加画布大小
    else:
        # 用于普通查看的“标准”字号
        tick_size = 18
        cbar_label_size = 20
        axis_label_size = 22
        figsize = (12, 5)

    print(f"len timeaxis= {len(timeaxis)}")
    # --- 计算逻辑 (保持不变) ---
    timeaxis_for_freq_calc = np.copy(timeaxis) * time_stretch_factor
    timeaxis_for_plotting = np.array(timeaxis, dtype=np.float64) * time_stretch_factor

    K_FFT = 0.000195313 ** 2 / 50 / 0.001
    dbm_results = []
    time_axis_plot_points = []
    
    total_time = timeaxis_for_freq_calc[-1] - timeaxis_for_freq_calc[0]
    total_time_point = len(timeaxis_for_freq_calc)
    samplerate = total_time_point / total_time
    
    freq_axis = np.linspace(-samplerate / 2, samplerate / 2, resolution)

    if freq_range is not None:
        min_freq, max_freq = freq_range
        if min_freq < np.min(freq_axis) or max_freq > np.max(freq_axis):
            raise ValueError(f"freq_range {freq_range} 超出可用频率范围 [{np.min(freq_axis)}, {np.max(freq_axis)}]")
        startidx = np.searchsorted(freq_axis, min_freq)
        finishidx = np.searchsorted(freq_axis, max_freq)
    else:
        min_freq, max_freq = freq_axis[0], freq_axis[-1]
        startidx = 0
        finishidx = len(freq_axis)

    for i in range(len(timeaxis) // 5):
        start_idx = i * 5
        end_idx = start_idx + resolution
        if end_idx > len(timeaxis):
            break
        
        window_time = timeaxis_for_plotting[start_idx:end_idx]
        window_signal = complexsignal[start_idx:end_idx]
        
        time_span = window_time[-1] - window_time[0]
        if time_span == 0:
            continue

        normalized_time = (window_time - window_time[0]) / time_span * 2 * np.pi
        spectrum_nufft = finufft.nufft1d1(normalized_time, window_signal.astype(np.complex128), resolution)
        fft_magnitude = np.abs(spectrum_nufft)
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2 + 1e-20) 
        fft_dbm_filtered = fft_dbm[startidx:finishidx]
        dbm_results.append(fft_dbm_filtered)
        
        time_axis_plot_points.append(timeaxis_for_plotting[start_idx])

    if not dbm_results:
        print("Warning: No results were generated to plot.")
        return None

    dbm_results = np.array(dbm_results).T
    dbm_results = dbm_results[::-1, :]
    
    # --- 绘图逻辑 ---
    fig, ax = plt.subplots(figsize=figsize)
    
    mesh = ax.pcolormesh(time_axis_plot_points, freq_axis[startidx:finishidx], dbm_results, shading='auto', cmap='jet', vmin=-80, vmax=-30)
    
    # 使用预设的变量来控制X轴和Y轴刻度数字的字号
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    cbar = fig.colorbar(mesh, ax=ax)
    # 使用预设的变量来控制颜色条标题和刻度数字的字号
    cbar.set_label('Channel Strength (dB)', fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    # 使用预设的变量来控制X轴和Y轴标题的字号
    ax.set_xlabel('Time (s)', fontsize=axis_label_size)
    ax.set_ylabel('Doppler Frequency (Hz)', fontsize=axis_label_size)
    
    ax.grid(True)

    if freq_range is not None:
        min_freq, max_freq = freq_range
        current_yticks = list(ax.get_yticks())
        desired_yticks = set(current_yticks + [min_freq, max_freq])
        y_min_lim, y_max_lim = ax.get_ylim()
        final_yticks = sorted([tick for tick in desired_yticks if y_min_lim <= tick <= y_max_lim])
        ax.set_yticks(final_yticks)

    fig.tight_layout(pad=0.5)

    if output_filename_base:
        save_dir = 'picture/doppler/'
        os.makedirs(save_dir, exist_ok=True)
        
        # 为大图模式添加后缀以区分文件名
        save_filename = f"{output_filename_base}_large.png" if Big_picture else f"{output_filename_base}.png"
        save_path = os.path.join(save_dir, save_filename)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f"图片已保存至: {save_path}")

    # 通常在函数内部不调用 plt.show()，以方便在脚本中连续调用
    # plt.show() 
    plt.close(fig)

    return dbm_results

def plot_spectrum(time_axis, freq_axis, data, output_file, title, ylabel, vmin=None, vmax=None):
    print(output_file)
    """Plot and save the spectrum with minimized white space."""
    plt.figure(figsize=(12, 8))
    time_axis = np.array(time_axis)
    
    plt.pcolormesh(time_axis, freq_axis, data.T, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(label='')
    
    plt.xlabel('Time (s)', fontsize=24)
    plt.ylabel(ylabel, fontsize=24)  # 让 ylabel 可变
    cbar.ax.tick_params(labelsize=24)  # 调整 colorbar 数值的字体大小
    
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.title(title, fontsize=20)

    # **减少留白**
    plt.tight_layout()  # 自动调整布局，减少留白
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)  # 手动微调边界

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)  # 彻底去掉外部空白
    plt.close()


def CalculateDoppler(real_signal, Start_point, Start_diff, Final_point, sample_rate, resolution):
    """
    Perform Short-Time Fourier Transform (STFT) on a complex signal.

    Parameters:
    - complex_signal: ndarray, the complex signal (I + jQ)
    - Start_point: int, starting index of the analysis window
    - Start_diff: int, step size for moving the window
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each window for FFT

    Returns:
    - dbm_results: list of dBm values for each window
    """
    # Constants for dBm calculation
    adc_step = 0.000195313  # ADC step size (0.8V range, 12-bit ADC)
    resistance = 50         # Ohm
    FFTLength = resolution // 2  # Effective FFT length
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # Initialize results container
    dbm_results = []
    time_axis = []  # To track time points

    # Iteratively process each window
    i = 0
    while True:
        start_idx = Start_point + i * Start_diff
        end_idx = start_idx + resolution

        # Stop if the window exceeds Final_point
        if end_idx > Final_point:
            break

        # Extract the current window
        window_signal = real_signal[start_idx:end_idx]

        # Perform FFT and FFTShift
        fft_result = fft(window_signal)
        fft_result = fftshift(fft_result)
        fft_magnitude = np.abs(fft_result)

        # Calculate dBm values
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
        dbm_results.append(fft_dbm)

        # Record time point
        time_point = start_idx / sample_rate
        time_axis.append(time_point)

        # Move to the next window
        i += 1

    # Plot results
    max_length = max(len(dbm) for dbm in dbm_results)
    dbm_results_padded = [np.pad(dbm, (0, max_length - len(dbm)), constant_values=np.nan) for dbm in dbm_results]

    # Filter to 0-100 Hz range
    freq_axis = np.linspace(-sample_rate/2, sample_rate/2, max_length)
    freq_mask = (freq_axis >= 0) & (freq_axis <= 80)
    dbm_results_filtered = np.array(dbm_results_padded)[:, freq_mask]
    freq_axis_filtered = freq_axis[freq_mask]


    # Create output directory
    output_dir = os.path.join("picture", "doppler")
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    output_file = os.path.join(output_dir, "stft_doppler_0_80Hz.png")
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        dbm_results_filtered.T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], freq_axis_filtered[0]*0.012, freq_axis_filtered[-1]*0.012],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    cbar = plt.colorbar(im, label='Power (dBm)')

    # Set font size to 20 for all elements
    plt.tick_params(axis='both', which='major', labelsize=20)  # X, Y axis tick labels
    cbar.ax.tick_params(labelsize=20)  # Colorbar tick labels

    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Velocity (m/s)', fontsize=20)
    plt.title('Speed Component', fontsize=20)
    plt.grid()

    # Save figure
    plt.savefig(output_file)
    plt.close()
    print(f"STFT plot (0-80 Hz) saved to {output_file}")

    return dbm_results


def plot_3d_angle_ignoreNan(angle_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file):
    # 展开矩阵，用于绘图
    angle_matrix_expanded = angle_matrix.transpose(1, 2, 0).reshape(len(freq_axis), FrameNumber)

    # 检查维度匹配
    if angle_matrix_expanded.shape != (len(freq_axis), len(time_axis)):
        raise ValueError(
            f"Matrix shape {angle_matrix_expanded.shape} does not match freq_axis ({len(freq_axis)}) "
            f"and time_axis ({len(time_axis)})."
        )

    # 自动缩放：找到有效的范围（非 NaN 部分）
    valid_rows = np.any(~np.isnan(angle_matrix_expanded), axis=1)  # 按行检查非 NaN
    valid_cols = np.any(~np.isnan(angle_matrix_expanded), axis=0)  # 按列检查非 NaN

    if not valid_rows.any() or not valid_cols.any():
        print("No valid data to plot.")
        return  # 如果全是 NaN，则直接退出

    # 计算有效范围
    row_start, row_end = np.where(valid_rows)[0][[0, -1]]
    col_start, col_end = np.where(valid_cols)[0][[0, -1]]

    # 裁剪数据到有效范围
    cropped_data = angle_matrix_expanded[row_start:row_end+1, col_start:col_end+1]
    cropped_freq_axis = freq_axis[row_start:row_end+1]
    cropped_time_axis = time_axis[col_start:col_end+1]

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})  # 全局字体大小设置

    # 绘制裁剪后的 3D 角度图像
    plt.figure(figsize=(10, 6))  # 调整图像尺寸
    plt.imshow(
        cropped_data, aspect='auto', cmap='twilight', vmin=-np.pi, vmax=np.pi,
        extent=[cropped_time_axis[0], cropped_time_axis[-1], cropped_freq_axis[0], cropped_freq_axis[-1]]
    )
    plt.colorbar(label='Angle (radians)', pad=0.02, shrink=0.8)  # 调整颜色条的位置和大小
    plt.title('Angle Plot (Cropped)', fontsize=20)  # 修改标题
    plt.ylabel('Frequency (MHz)', fontsize=18)  # 加大Y轴标签字体
    plt.xlabel('Time (s)', fontsize=18)  # 加大X轴标签字体
    plt.xticks(fontsize=16)  # 调整X轴刻度字体
    plt.yticks(fontsize=16)  # 调整Y轴刻度字体

    # 调整图像布局以减少留白
    plt.tight_layout()

    # 保存绘图
    plt.savefig(output_file, bbox_inches='tight')  # 去掉多余留白
    print(f"Plot saved to {output_file}")
    plt.close()


def DCcalibration(folder_name):
    # 调用 process_files 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_files(folder_name)

    # 初始化存储 FFT DC 分量的二维数组
    # 行数为 frame_length，每列对应一个频率段
    FFTdcresult = np.zeros((frame_length, len(center_freqs)), dtype=complex)

    # 使用 tqdm 显示进度条
    total_frames = frame_length * len(center_freqs)
    with tqdm(total=total_frames, desc="Processing frames") as progress_bar:
        for freq_idx, center_freq in enumerate(center_freqs):
            for frame_idx in range(frame_length):
                # 获取对应的文件路径
                file_index = freq_idx * frame_length + frame_idx
                bin_file_path = sorted_files[file_index]

                # 读取 I/Q 数据
                I_data, Q_data = read_bin_file(bin_file_path)

                # 计算复数形式的信号
                complex_signal = I_data + 1j * Q_data
                fft_dcresult = fft(complex_signal)

                # 存储 FFT DC 分量
                FFTdcresult[frame_idx, freq_idx] = fft_dcresult[0]

                # 更新进度条
                progress_bar.update(1)

    # 初始化存储每个频率段 DC Offset 中位数的数组
    FFT_complex_median_array = np.zeros(len(center_freqs), dtype=complex)

    for freq_idx in range(len(center_freqs)):
        FFT_real_median = np.median(FFTdcresult[:, freq_idx].real)
        FFT_imag_median = np.median(FFTdcresult[:, freq_idx].imag)
        FFT_complex_median_array[freq_idx] = FFT_real_median + 1j * FFT_imag_median

    # 打印每个频率段的中位数
    for freq_idx, center_freq in enumerate(center_freqs):
        print(
            f"中心频率 {center_freq} 的 FFT DC 分量中位数: "
            f"{FFT_complex_median_array[freq_idx].real:.6f} + {FFT_complex_median_array[freq_idx].imag:.6f}i"
        )

    return FFT_complex_median_array

def read_bin_file(file_name):
    # 读取 bin 文件并解释为 int16 数据
    raw_data = np.fromfile(file_name, dtype=np.int16)

    # 分离 I 和 Q 数据
    I_data = raw_data[0::2]
    Q_data = raw_data[1::2]

    return I_data, Q_data

def read_bin_file_matrix(file_name, ClipFFTLength, n):
    # 读取 bin 文件并解释为 int16 数据
    with open(file_name, "rb") as f:
        raw_data = np.fromfile(f, dtype=np.int16)

    # 将 I 和 Q 数据分离并存入矩阵
    I_data_matrix = raw_data[0::2].reshape(-1, n)[:ClipFFTLength, :]
    Q_data_matrix = raw_data[1::2].reshape(-1, n)[:ClipFFTLength, :]
    
    return I_data_matrix, Q_data_matrix

def DCcalibrationtest(folder_name):
    # 调用 process_files 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_files(folder_name)

    # 使用 tqdm 显示进度条
    total_frames = frame_length * len(center_freqs)
    with tqdm(total=total_frames, desc="Testing FFT results") as progress_bar:
        for freq_idx, center_freq in enumerate(center_freqs):
            for frame_idx in range(frame_length):
                # 获取对应的文件路径
                file_index = freq_idx * frame_length + frame_idx
                bin_file_path = sorted_files[file_index]

                # 读取 I/Q 数据
                I_data, Q_data = read_bin_file(bin_file_path)

                # 计算复数形式的信号
                complex_signal = I_data + 1j * Q_data
                fft_dcresult = fft(complex_signal)

                # 打印 fft_dcresult 的关键信息
                max_value = np.max(np.abs(fft_dcresult))
                max_index = np.argmax(np.abs(fft_dcresult))
                fft_length = len(fft_dcresult)

                # 获取前三大的频率分量
                sorted_indices = np.argsort(np.abs(fft_dcresult))[::-1]  # 按幅值从大到小排序
                top_frequencies = [(idx, fft_dcresult[idx]) for idx in sorted_indices[:3]]

                # 打印结果
                print(f"中心频率 {center_freq} 帧 {frame_idx} 的 FFT 结果:")
                print(f"  FFT 长度: {fft_length}")
                print(f"  最大值: {max_value:.6f} at index {max_index}")
                print(f"  前三大频率分量:")
                for i, (idx, value) in enumerate(top_frequencies):
                    print(f"    {i + 1}. Index: {idx}, Value: {value.real:.6f} + {value.imag:.6f}i")

                # 更新进度条
                progress_bar.update(1)

    print("DC Calibration Test Completed.")

def widerdiffertimetry(folder_name, FFT_Calibration, sampling_rate):
    # 获取所有 .bin 文件路径和频率信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_files(folder_name)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长 0.8V 12bit ADC
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # 构造频率裁剪范围 (fgap)
    freq_axis = []  # 最终完整频率轴
    fgap_ratios = []  # 存储各个频率段保留的比例


    fgap = (center_freqs[1] - center_freqs[0]) * 1e6  # fgap in Hz
    fgap_ratio = fgap / sampling_rate  # 转化为 FFT 长度的比例

    # 初始化存储 dBm 数据的矩阵（无法提前声明完整长度，用动态追加的方式）
    dbm_matrix = []
    cropped_freq_axes = []

    # 遍历每个中心频率和时间帧，填充 dBm 数据矩阵
    index = 0  # 文件索引
    total_files = len(sorted_files)

    for k, center_freq in enumerate(center_freqs):
        start_idx = int((1 - fgap_ratio) * FFTLength / 2)
        end_idx = int((1 + fgap_ratio) * FFTLength / 2)

        cropped_freq_axis = np.linspace(
            center_freq * 1e6 - fgap_ratio * sampling_rate / 2,
            center_freq * 1e6 + fgap_ratio * sampling_rate / 2,
            end_idx - start_idx,
            endpoint=False,
        )
        cropped_freq_axes.append(cropped_freq_axis)
        print(fgap_ratio,"fgap")
        print(center_freq * 1e6 - fgap_ratio * sampling_rate / 2,
            center_freq * 1e6 + fgap_ratio * sampling_rate / 2)

        for frame in range(frame_length * frequency_frame_groups[k]):
            bin_file_path = sorted_files[index]
            index += 1  # 更新索引

            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)

            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data

            # 进行 FFT 并应用校准
            fft_result = fft(complex_signal)
            if FFT_Calibration[0] != 0:
                fft_result[0] = fft_result[0] - FFT_Calibration[0]
            fft_result = fftshift(fft_result)
            fft_magnitude = np.abs(fft_result)

            # 裁剪 FFT 结果
            # 计算 dBm 值
            fft_dbm_crop = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
            fft_dbm = fft_dbm_crop[start_idx:end_idx]

            # 存储到矩阵（动态追加）
            if len(dbm_matrix) <= frame:
                dbm_matrix.append([])
            dbm_matrix[frame].append(fft_dbm)

        print(f"Frequency band {k + 1}/{len(center_freqs) - 1} processed ({index}/{total_files} files).")

    # 将 dbm_matrix 转为 numpy 数组
    dbm_matrix = np.array(dbm_matrix)

    # 构造时间轴
    time_axis = np.arange(1, frame_length * frequency_frame_groups[0] + 1) * FFTLength / sampling_rate

    print(f"All {total_files} frames processed for {len(center_freqs) - 1} frequency bands.")

    # 根据校准生成文件名
    base_folder_name = os.path.basename(folder_name)
    output_dir = 'picture'
    os.makedirs(output_dir, exist_ok=True)

    if all(calib == 0 for calib in FFT_Calibration):  # 无校准
        file_suffix = "_cropped.png"
    else:  # 有校准
        file_suffix = "_cropped_withDCcalibration.png"

    output_file = os.path.join(output_dir, f"{base_folder_name}{file_suffix}")

    # 绘制结果
    plot_3d_dbm(dbm_matrix, frame_length * frequency_frame_groups[0], time_axis, len(cropped_freq_axes[0]), np.concatenate(cropped_freq_axes)[::-1], output_file)

    return dbm_matrix, cropped_freq_axes, time_axis

def maxsignalpricise(complex_signal, Start_number, Finish_number, peaknumber, Output):
    """
    Analyzes a specified segment of a complex signal, computes its DFT, and extracts peak information.

    Parameters:
    - complex_signal: ndarray, the complex signal array to be analyzed
    - Start_number: int, starting index of the segment to analyze (inclusive)
    - Finish_number: int, ending index of the segment to analyze (exclusive)
    - peaknumber: int, number of peaks to extract from the DFT result
    - Output: list, the output list where the extracted data will be stored
    [
        Start_number,                # frame, 当前信号段的起始索引
        [
            [relative_n1, peak1],    # 第一个峰值：[相对频率索引, 峰值复数]
            [relative_n2, peak2],    # 第二个峰值：[相对频率索引, 峰值复数]
            ...
        ]
    ]
    Returns:
    - None (Output is modified in place)
    """
    # Extract the signal segment
    signal_segment = complex_signal[Start_number:Finish_number]

    # Perform DFT on the segment
    fft_result = fftshift(fft(signal_segment))

    # Compute the magnitude of the DFT result
    magnitude = np.abs(fft_result)

    # Get the indices of the `peaknumber` largest magnitudes
    peak_indices = np.argsort(magnitude)[-peaknumber:][::-1]

    # Prepare the output for this segment
    peaks = []

    for peak_idx in peak_indices:
        N = Finish_number - Start_number
        relative_n = peak_idx - N // 2  # Adjust for fftshift
        if N % 2 != 0 and peak_idx > N // 2:
            relative_n += 1  # Correct for asymmetry when N is odd
        peak_complex_value = fft_result[peak_idx]  # Complex value at the peak
        peaks.append([relative_n, peak_complex_value])

    # Sort peaks by relative_n
    peaks.sort(key=lambda x: x[0])  # Sort by `relative_n`

    # Append to the Output list
    Output.append([Start_number, peaks])



def LongDCCalibration(folder_name, start_index):
    """
    计算从排序文件的第 N 个索引开始到最后每个文件的 I/Q 数据均值。

    参数：
    - folder_name: str, 包含所有文件路径的文件夹名称
    - start_index: int, 从第几个文件开始处理（索引从 0 开始）

    返回：
    - average_results: list, 每个文件的平均复数值
    """
    average_results = []

    # 调用 process_files 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_files(folder_name)

    # 从指定索引开始处理文件
    total_frames = len(sorted_files[start_index:])

    # 初始化存储每帧的复数均值
    folder_average = 0 + 0j

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc=f"Processing files from index {start_index}") as progress_bar:
        for file_idx, bin_file_path in enumerate(sorted_files[start_index:], start=start_index):
            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)

            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data

            # 累加复数信号均值
            folder_average += np.mean(complex_signal)

            # 更新进度条
            progress_bar.update(1)

    # 计算文件的复数均值
    folder_average /= total_frames
    average_results.append(folder_average)

    print(f"Processed files from index {start_index}, average: {folder_average.real:.6f} + {folder_average.imag:.6f}i")

    return average_results

def CWOFFsetpicture(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, peaknumber, output):
    """
    Performs channel sounding by iteratively analyzing segments of a complex signal.

    Parameters:
    - complex_signal: ndarray, the complex signal array
    - Start_point: int, starting index of the first segment
    - Start_diff: int, step size for each segment
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each segment for FFT
    - peaknumber: int, number of peaks to extract in each frame
    - output: list, stores the results of the analysis

    Returns:
    - None (output modified in place)
    """
    i = 0
    while True:
        Start_number = Start_point + i * Start_diff
        Finish_number = Start_number + resolution
        if Finish_number > Final_point:
            break
        maxsignalpricise(complex_signal, Start_number, Finish_number, peaknumber, output)
        i += 1

    # Plot the results
    x_axis = np.arange(i) * Start_diff / sample_rate  # 时间帧的时间点
    y_axis_actual_phase = []  # 实测相位
    y_axis_reference_phase = []  # 参考相位
    time_axis = []  # 时间轴
    frequency_axis = []  # 频率轴
    frequency_axis2 = []  # 新增的频率2轴

    last_delta_phase = None  # 用于存储上一帧的 delta_phase
    k=0
    for idx, frame in enumerate(output):
        peaks = frame[1]  # 取每一帧的峰值信息
        if len(peaks) > 0:
            n_value = peaks[0][0]  # 取 n 值最小的
            complex_value = peaks[0][1]
            phase_actual = np.angle(complex_value)  # 实测相位
            y_axis_actual_phase.append(phase_actual)

            # 计算参考相位
            if idx == 0:
                y_axis_reference_phase.append(phase_actual)
                time_axis.append(0)  # 初始时间点
                frequency_axis.append(0)  # 初始频率
                frequency_axis2.append(0)  # 初始频率2
            else:
                freq = n_value * sample_rate / resolution  # 峰值频率
                time_step = Start_diff / sample_rate  # 时间步长
                ref_phase = y_axis_reference_phase[-1] + freq * time_step * 2 * np.pi
                # 调整相位范围到 [-π, π)
                ref_phase = (ref_phase + np.pi) % (2 * np.pi) - np.pi
                y_axis_reference_phase.append(ref_phase)

                # 计算频率： (实际相位 - 参考相位) / (2π * 时间步长)
                delta_phase = phase_actual - ref_phase
                delta_phase = (delta_phase + np.pi) % (2 * np.pi) - np.pi  # 调整到 [-π, π)

                # 计算频率2：基于相位差变化率
                if last_delta_phase is not None:
                    delta_phase_diff = delta_phase - last_delta_phase

                    #校正相位跳跃
                    if delta_phase_diff > np.pi:
                        delta_phase_diff -= 2 * np.pi
                        print("减过")
                        k-=1
                    elif delta_phase_diff < -np.pi:
                        delta_phase_diff += 2 * np.pi
                        print("加过")
                        k+=1

                    # 计算频率2
                    frequency = (delta_phase+k*2*np.pi)/(2 * np.pi * time_step * idx)
                    frequency2 = delta_phase_diff/(2 * np.pi * time_step)
                else:
                    frequency2 = 0  # 初始化时设置为 0
                    frequency = (delta_phase+k*2*np.pi)/(2 * np.pi * time_step * idx)

                last_delta_phase = delta_phase

                # 存储时间和频率
                time_axis.append(idx * time_step)
                frequency_axis.append(frequency)
                frequency_axis2.append(frequency2)

    # 计算 Measured - Reference Phase
    phase_difference = np.array(y_axis_actual_phase) - np.array(y_axis_reference_phase)

    # 调整到 [-π, π)
    phase_difference = (phase_difference + np.pi) % (2 * np.pi) - np.pi

    # 绘制相位对比图
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_axis_actual_phase, label="Measured Phase", marker="o")
    plt.plot(x_axis, y_axis_reference_phase, label="Reference Phase", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.title("Phase Comparison: Measured vs Reference")
    plt.legend()
    plt.grid()
    plt.xticks(np.linspace(0, x_axis[-1], num=10))  # 精确刻度
    output_dir = os.path.join("picture", "Channelsounding")
    os.makedirs(output_dir, exist_ok=True)
    base_folder_name = "Channelsounding"
    output_file = os.path.join(output_dir, f"{base_folder_name}_phasecompare.png")
    plt.savefig(output_file)
    plt.close()

    # 绘制频率随时间变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, frequency_axis, label="Frequency", marker="o", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Frequency vs Time")
    plt.legend()
    plt.grid()
    plt.xticks(np.linspace(0, time_axis[-1], num=10))  # 精确刻度
    plt.savefig(os.path.join(output_dir, f"{base_folder_name}_freq_vs_time.png"))
    plt.close()

    # 绘制频率2随时间变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, frequency_axis2, label="Frequency2", marker="o", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency2 (Hz)")
    plt.title("Frequency2 vs Time")
    plt.legend()
    plt.grid()
    plt.xticks(np.linspace(0, time_axis[-1], num=10))  # 精确刻度
    plt.savefig(os.path.join(output_dir, f"{base_folder_name}_freq2_vs_time.png"))
    plt.close()

    # 绘制 Measured - Reference Phase 图
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, phase_difference, label="Measured - Reference Phase", color="green", linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase Difference (radians)")
    plt.title("Measured - Reference Phase vs Time")
    plt.legend()
    plt.grid()

    # 设置自适应的 y 轴范围，根据相位差的实际最小值和最大值调整
    y_min = np.min(phase_difference) - 0.1  # 给最小值留一定余量
    y_max = np.max(phase_difference) + 0.1  # 给最大值留一定余量
    plt.ylim(y_min, y_max)

    # 设置精确的 x 轴刻度
    plt.xticks(np.linspace(0, x_axis[-1], num=10))

    # 保存图像
    output_file_diff = os.path.join(output_dir, f"{base_folder_name}_phase_difference.png")
    plt.savefig(output_file_diff)
    print(f"Phase difference plot saved to {output_file_diff}")
    plt.close()

def maxsignal_analysis(folder_name, file_number, start_number, finish_number, peak_number, sampling_rate):
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)

    if len(sorted_files) < file_number:
        print("Not enough files in the folder.")
        return

    target_file = sorted_files[file_number - 1]  # Select the file based on file_number

    with open(target_file, "rb") as f:
        data = np.fromfile(f, dtype=np.int16)

    I_data = data[::2]
    Q_data = data[1::2]
    complex_signal = I_data + 1j * Q_data

    # Analyze signal segment
    signal_segment = complex_signal[start_number:finish_number]
    fft_result = fftshift(fft(signal_segment))

    # Compute relative indices `n`
    N = finish_number - start_number
    indices = np.arange(N)
    relative_n = indices - N // 2
    if N % 2 != 0:
        relative_n[relative_n > 0] -= 1

    frequencies = relative_n * sampling_rate / N

    # Print all `n`, corresponding frequencies, and complex values
    print(f"Analysis of file: {target_file}")
    print("All n, corresponding frequencies (Hz), and complex values:")
    for n, freq, complex_value in zip(relative_n, frequencies, fft_result):
        print(f"n: {n}, Frequency: {freq/1e6:.3f} MHz, Complex value: {complex_value}")

    # Zero-padding for improved resolution
    N_padded = N * 4
    signal_segment_padded = np.zeros(N_padded, dtype=complex)
    signal_segment_padded[:N] = signal_segment

    fft_result_padded = fftshift(fft(signal_segment_padded))
    magnitude_padded = np.abs(fft_result_padded)
    frequencies_padded = np.linspace(-sampling_rate / 2, sampling_rate / 2, N_padded, endpoint=False)

    # Find peak and refine with interpolation
    peak_idx_padded = np.argmax(magnitude_padded)
    peak_frequency = frequencies_padded[peak_idx_padded]

    if 0 < peak_idx_padded < len(magnitude_padded) - 1:
        a = magnitude_padded[peak_idx_padded - 1]
        b = magnitude_padded[peak_idx_padded]
        c = magnitude_padded[peak_idx_padded + 1]
        delta = (a - c) / (2 * (a - 2 * b + c)) if (a - 2 * b + c) != 0 else 0
        interpolated_peak_frequency = frequencies_padded[0] + (peak_idx_padded + delta) * (frequencies_padded[1] - frequencies_padded[0])
        interpolated_peak_magnitude = b - (a - c) ** 2 / (8 * (a - 2 * b + c))

        phase_minus_1 = np.angle(fft_result_padded[peak_idx_padded - 1])
        phase_0 = np.angle(fft_result_padded[peak_idx_padded])
        phase_plus_1 = np.angle(fft_result_padded[peak_idx_padded + 1])

        interpolated_phase = phase_0 + delta * (phase_plus_1 - phase_minus_1) / 2

        print(f"Parabolic interpolated peak frequency: {interpolated_peak_frequency / 1e6:.6f} MHz")
        print(f"Parabolic interpolated peak magnitude: {interpolated_peak_magnitude:.6f}")
        print(f"Parabolic interpolated phase: {interpolated_phase:.6f} radians")

        # Refine interpolation to 128x resolution
        finer_resolution = 128
        finer_frequencies = np.linspace(frequencies_padded[peak_idx_padded - 1], frequencies_padded[peak_idx_padded + 1], finer_resolution)
        finer_magnitude = np.interp(finer_frequencies, frequencies_padded, magnitude_padded)
        finer_peak_idx = np.argmax(finer_magnitude)
        finer_peak_frequency = finer_frequencies[finer_peak_idx]

        finer_I = np.real(fft_result_padded[peak_idx_padded])
        finer_Q = np.imag(fft_result_padded[peak_idx_padded])

        print(f"Refined interpolated peak frequency: {finer_peak_frequency / 1e6:.6f} MHz")
        print(f"Corresponding I: {finer_I}, Q: {finer_Q}")
    else:
        print("Peak index is at the boundary; parabolic interpolation skipped.")

    print(f"Zero-padded interpolated peak frequency: {peak_frequency / 1e6:.6f} MHz")

    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_padded / 1e6, magnitude_padded, label='Zero-padded FFT Spectrum')
    plt.title('Zero-padded FFT Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()

                # 存储时间和频率



def CalculateFrequencyOffset(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, peaknumber, output):

    """
    Calculates a single frequency offset by iteratively analyzing segments of a complex signal.
    这是每一个FFTframe里面的计算，意图是测这一帧offset的大小，猜测在极短尺度下，多普勒导致的frequencyoffset频率不变
    Parameters:
    - complex_signal: ndarray, the complex signal array
    - Start_point: int, starting index of the first segment
    - Start_diff: int, step size for each segment
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each segment for FFT
    - peaknumber: int, number of peaks to extract in each frame
    - output: list, stores the results of the analysis

    Returns:
    - frequency_offset: float, the calculated frequency offset
    """

    i = 0
    while True:
        Start_number = Start_point + i * Start_diff
        Finish_number = Start_number + resolution
        if Finish_number > Final_point:
            break
        maxsignalpricise(complex_signal, Start_number, Finish_number, peaknumber, output)
        i += 1

    # Plot the results
    x_axis = np.arange(i) * Start_diff / sample_rate  # 时间帧的时间点
    y_axis_actual_phase = []  # 实测相位
    y_axis_reference_phase = []  # 参考相位
    time_axis = []  # 时间轴

    frequency_div = []  # 新增的频率2轴
    frequency_total = []
    last_delta_phase = None  # 用于存储上一帧的 delta_phase
    k=0
    for idx, frame in enumerate(output[2:]):
        peaks = frame[1]  # 取每一帧的峰值信息
        if len(peaks) > 0:
            n_value = peaks[0][0]  # 取 n 值最小的,第一个0指代最小，第二个0指代n值
            complex_value = peaks[0][1]
            phase_actual = np.angle(complex_value)  # 实测相位
            y_axis_actual_phase.append(phase_actual)

            # 计算参考相位
            if idx == 0:
                y_axis_reference_phase.append(phase_actual)
                time_axis.append(0)  # 初始时间点

            else:
                freq = n_value * sample_rate / resolution  # 峰值频率
                time_step = Start_diff / sample_rate  # 时间步长
                ref_phase = y_axis_reference_phase[-1] + freq * time_step * 2 * np.pi
                # 调整相位范围到 [-π, π)
                ref_phase = (ref_phase + np.pi) % (2 * np.pi) - np.pi
                y_axis_reference_phase.append(ref_phase)

                # 计算频率： (实际相位 - 参考相位) / (2π * 时间步长)
                delta_phase = phase_actual - ref_phase
                delta_phase = (delta_phase + np.pi) % (2 * np.pi) - np.pi  # 调整到 [-π, π)

                # 计算频率2：基于相位差变化率
                if last_delta_phase is not None:
                    delta_phase_diff = delta_phase - last_delta_phase

                    #校正相位跳跃
                    if delta_phase_diff > np.pi:
                        delta_phase_diff -= 2 * np.pi
                        k-=1
                    elif delta_phase_diff < -np.pi:
                        delta_phase_diff += 2 * np.pi
                        k+=1

                    # 计算频率2
                    
                    frequency2 = delta_phase_diff/(2 * np.pi * time_step)+freq
                    frequency_div.append(frequency2)
                    #print(freq,frequency2,n_value,(n_value+1) * sample_rate / resolution )


                last_delta_phase = delta_phase

                # 存储时间和频率


                #frequency_total.append(freq+frequency2)
    variance = np.var(frequency_div)
    frequencyoffset = np.mean(frequency_div)

    # 计算 Measured - Reference Phase

    return frequencyoffset,variance

def TimedomainFreqoffset(folder_name, Start_point, Start_diff, Final_point, sample_rate, resolution, peaknumber, output, framejumppoint=0):
    """
    Calculates and plots frequency offsets and variances across time for I/Q data files.

    Parameters:
    - folder_name: str, folder containing the I/Q data files
    - Start_point, Start_diff, Final_point: int, FFT windowing parameters
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each FFT segment
    - peaknumber: int, number of peaks to extract
    - output: list, placeholder for FFT output
    - framejumppoint: int, number of frames to skip during processing (default is 0, meaning process every frame)

    Returns:
    - frequency_offsets: list of calculated frequency offsets
    - variances: list of calculated variances
    """
    # 调用 process_filesIntimedomain 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # 初始化存储每帧频率偏移和方差的数组
    frequency_offsets = []  # 存储 FrequencyOffset 的数组
    variances = []  # 存储 variance 的数组

    # 获取文件夹的基本名称
    base_folder_name = os.path.basename(os.path.normpath(folder_name))

    # 创建目标目录
    output_dir = os.path.join("picture", "avgamp")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个 bin 文件
    frame_count = 0  # 用于时间轴的计数
    with tqdm(total=len(sorted_files), desc="Processing files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files[2:]):
            # 如果 framejumppoint > 0，跳过指定的帧
            if framejumppoint > 0 and idx % (framejumppoint + 1) != 0:
                continue

            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)

            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data
            output = []
            
            # 计算 FrequencyOffset 和 variance
            FrequencyOffset, variance = CalculateFrequencyOffset(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, peaknumber, output)
            
            # 存储到对应数组
            frequency_offsets.append(FrequencyOffset)
            variances.append(variance)

            # 更新进度条和时间计数
            frame_count += 1
            progress_bar.update(1)

    # 根据 FFTLength 和采样率计算新的时间轴（考虑跳跃）
    time_axis = np.arange(len(frequency_offsets)) * FFTLength * (framejumppoint + 1) / sample_rate
    print(f"Length of time_axis: {len(time_axis)}")
    print(f"Length of frequency_offsets: {len(frequency_offsets)}")

    print(frequency_offsets)
    # 绘制 FrequencyOffset 随时间的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, frequency_offsets, label="Frequency Offset", marker="o", color="blue")
    plt.title("Frequency Offset vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency Offset (Hz)")
    plt.grid()
    plt.legend()
    output_file_freq_offset = os.path.join(output_dir, f"{base_folder_name}_frequency_offset.png")
    plt.savefig(output_file_freq_offset)
    print(f"Frequency Offset plot saved to {output_file_freq_offset}")
    plt.close()

    # 绘制 Variance 随时间的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, variances, label="Variance", marker="o", color="red")
    plt.title("Variance vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Variance")
    plt.grid()
    plt.legend()
    output_file_variance = os.path.join(output_dir, f"{base_folder_name}_variance.png")
    plt.savefig(output_file_variance)
    print(f"Variance plot saved to {output_file_variance}")
    plt.close()

    return frequency_offsets, variances

# 你可以使用原来的 `channel_response_matrix`，`time_axis` 和 `freq_axis`
# 作为参数来调用 plot_3d_matplotlib 函数
def CompareChannel(Folder, DC_Calibration, sampling_rate, avgminus=False):
    """
    Compare the channel response by dividing the spectra from RxFolder and TxFolder.

    Parameters:
    - RxFolder: str, path to the folder containing Rx data.
    - TxFolder: str, path to the folder containing Tx data.
    - DC_Calibration: list, DC calibration values for each frequency band.
    - sampling_rate: float, sampling rate of the signal.
    - FFTLength: int, FFT length.

    Returns:
    - channel_response_matrix: ndarray, complex channel response for all frames and bands.
    - time_axis: ndarray, time axis for the frames.
    - freq_axis: ndarray, frequency axis for the FFT.
    """
    TxFolder = os.path.join(Folder,"Channel0")#不变的
    RxFolder = os.path.join(Folder,"Channel1")#人动的
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    print(total_data_count)
    # 确保 Rx 和 Tx 文件结构一致
    if len(rx_sorted_files) != len(tx_sorted_files):
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The number of files in RxFolder and TxFolder must be the same.")
    if rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The structure of RxFolder and TxFolder must match.")
        
    # 初始化存储信道响应的矩阵
    total_frames = rx_frame_length * rx_frequency_frame_groups[0]
    num_freq_bands = len(rx_center_freqs)
    channel_response_matrix = np.zeros((total_frames, num_freq_bands, FFTLength), dtype=complex)
    Tx_result = np.zeros((total_frames, num_freq_bands, FFTLength), dtype=complex)
    Rx_result = np.zeros((total_frames, num_freq_bands, FFTLength), dtype=complex)

    # 遍历每个中心频率和时间帧，计算信道响应
    index = 0
    total_files = len(rx_sorted_files)
    channel_response_mean=0
    time_axis = np.arange(1, total_frames + 1) * FFTLength / sampling_rate
    freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, FFTLength, endpoint=False)
    for k, center_freq in enumerate(rx_center_freqs):
        for frame in range(rx_frame_length * rx_frequency_frame_groups[k]):
            # 获取 Rx 和 Tx 对应的文件
            rx_file_path = rx_sorted_files[frame]
            tx_file_path = tx_sorted_files[frame]


            # 读取 Rx 和 Tx 的 I/Q 数据
            rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
            tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
            # 计算复数形式的 Rx 和 Tx 信号
            rx_complex_signal = rx_I_data + 1j * rx_Q_data
            tx_complex_signal = tx_I_data + 1j * tx_Q_data

            # 进行 FFT 并应用校准
            rx_fft_result = fftshift(fft(rx_complex_signal))
            tx_fft_result = fftshift(fft(tx_complex_signal))

            if DC_Calibration[k] != 0:
                rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0]*FFTLength
                tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1]*FFTLength

            # 计算信道响应
            channel_response = rx_fft_result / tx_fft_result
            channel_response_matrix[frame, k, :] = channel_response
            # Rx_result[frame, k, :]= rx_fft_result 
            # Tx_result[frame, k, :]= tx_fft_result

        print(f"Frequency band {k + 1}/{num_freq_bands} processed ({index}/{total_files} files).")

    # 构造时间轴和频率轴

    print(f"All {total_files} frames processed for {num_freq_bands} frequency bands.")
    # 根据校准生成文件名
    base_folder_name = os.path.basename(os.path.dirname(RxFolder))
    if avgminus==True:

        avg_value = np.mean(channel_response_matrix)
        print(avg_value)
        avg_value = np.median(channel_response_matrix)
        print(avg_value)
        channel_response_matrix -= avg_value
    print("meanagain",np.mean(channel_response_matrix))

    output_dir = 'picture/Channel/phase'
    os.makedirs(output_dir, exist_ok=True)
    if all(calib == 0 for calib in DC_Calibration):  # 无校准
        file_suffix = ".png"
    else:  # 有校准
        file_suffix = "withDCcalibration.png"

    output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

    # 绘制绝对值信道响应
    abs_channel_response = np.angle(channel_response_matrix)
    plot_3d_angle(abs_channel_response, total_frames, time_axis, FFTLength, freq_axis[::-1], output_file)

    output_dir = 'picture/Channel/Amp'
    os.makedirs(output_dir, exist_ok=True)
    if all(calib == 0 for calib in DC_Calibration):  # 无校准
        file_suffix = ".png"
    else:  # 有校准
        file_suffix = "withDCcalibration.png"

    output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

    # 绘制绝对值信道响应
    abs_channel_response = np.abs(channel_response_matrix)
    plot_3d_amp(abs_channel_response, total_frames, time_axis, FFTLength, freq_axis[::-1], output_file)
    #channel_response_matrix = channel_response_matrix.transpose(1, 2, 0).reshape(len(freq_axis[::-1],), total_frames)
    # Rx_result = Rx_result.transpose(1, 2, 0).reshape(len(freq_axis[::-1],), total_frames)
    # Tx_result = Tx_result.transpose(1, 2, 0).reshape(len(freq_axis[::-1],), total_frames)
    # print("Rx_result shape:", np.shape(Rx_result))
    # print("Tx_result shape:", np.shape(Tx_result))

    # # 定义保存文件名
    # output_mat_file_1 = "Rx1Humanmoving_response.mat"
    # output_mat_file_2 = "Rx2ClosetoTx_response.mat"

    # # 分别保存到 .mat 文件
    # sio.savemat(output_mat_file_1, {'Rx1Humanmoving_response': Rx_result})
    # sio.savemat(output_mat_file_2, {'Rx2ClosetoTx_response': Tx_result})

    # print(f"数据已保存到 {output_mat_file_1} 和 {output_mat_file_2}")

# 保存到 .mat 文件
    
    return channel_response_matrix, freq_axis, time_axis



def CompareChannelwithTime(Folder, sampling_rate, time_axis, avgminus=False):
    """
    Compare the channel response by dividing the spectra from RxFolder and TxFolder.
    """
    TxFolder = os.path.join(Folder, "Channel0")  # 不变的
    RxFolder = os.path.join(Folder, "Channel1")  # 人动的
    
    rx_sorted_files, rx_center_freqs, rx_frame_length, _ = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, _ = process_files(TxFolder)
    
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
    FFTLength = len(total_data) // 2  # 计算FFT长度
    
    if len(rx_sorted_files) != len(tx_sorted_files) or rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        raise ValueError("The structure of RxFolder and TxFolder must match.")
    
    total_frames = rx_frame_length
    channel_response_matrix = np.zeros((total_frames, FFTLength), dtype=complex)
    freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, FFTLength, endpoint=False)
    
    for frame in range(total_frames):
        rx_file_path = rx_sorted_files[frame]
        tx_file_path = tx_sorted_files[frame]
        
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
        
        rx_complex_signal = rx_I_data + 1j * rx_Q_data
        tx_complex_signal = tx_I_data + 1j * tx_Q_data
        
        rx_fft_result = fftshift(fft(rx_complex_signal))
        tx_fft_result = fftshift(fft(tx_complex_signal))
        
        channel_response_matrix[frame, :] = rx_fft_result / tx_fft_result
    
    if avgminus:
        avg_value = np.median(channel_response_matrix)
        channel_response_matrix -= avg_value
    
    base_folder_name = os.path.basename(os.path.dirname(RxFolder))
    os.makedirs('picture/NewChannel/phase', exist_ok=True)
    os.makedirs('picture/NewChannel/Amp', exist_ok=True)
    
    output_phase = f'picture/NewChannel/phase/{base_folder_name}_channel_response.png'
    plot_spectrum(time_axis, freq_axis, np.angle(channel_response_matrix), output_phase, title='Phase Response', ylabel='Frequency (Hz)')
    
    output_amp = f'picture/NewChannel/Amp/{base_folder_name}_channel_response.png'
    plot_spectrum(time_axis, freq_axis, np.abs(channel_response_matrix), output_amp, title='Amplitude Response', ylabel='Frequency (Hz)', vmin=0, vmax=1.6)
    
    return channel_response_matrix, freq_axis, time_axis

def OnlyCompareChannel(Folder, DC_Calibration, sampling_rate, avgminus=False):
    """
    Compare the channel response by dividing the spectra from RxFolder and TxFolder.

    Parameters:
    - RxFolder: str, path to the folder containing Rx data.
    - TxFolder: str, path to the folder containing Tx data.
    - DC_Calibration: list, DC calibration values for each frequency band.
    - sampling_rate: float, sampling rate of the signal.
    - FFTLength: int, FFT length.

    Returns:
    - channel_response_matrix: ndarray, complex channel response for all frames and bands.
    - time_axis: ndarray, time axis for the frames.
    - freq_axis: ndarray, frequency axis for the FFT.
    """
    TxFolder = os.path.join(Folder,"Channel0")#不变的
    RxFolder = os.path.join(Folder,"Channel1")#人动的
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    print(total_data_count)
    # 确保 Rx 和 Tx 文件结构一致
    if len(rx_sorted_files) != len(tx_sorted_files):
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The number of files in RxFolder and TxFolder must be the same.")
    if rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The structure of RxFolder and TxFolder must match.")
        
    # 初始化存储信道响应的矩阵
    total_frames = rx_frame_length * rx_frequency_frame_groups[0]
    num_freq_bands = len(rx_center_freqs)
    channel_response_matrix = np.zeros((total_frames, num_freq_bands, FFTLength), dtype=complex)

    # 遍历每个中心频率和时间帧，计算信道响应
    index = 0
    total_files = len(rx_sorted_files)
    channel_response_mean=0
    time_axis = np.arange(1, total_frames + 1) * FFTLength / sampling_rate
    freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, FFTLength, endpoint=False)
    for k, center_freq in enumerate(rx_center_freqs):
        for frame in range(rx_frame_length * rx_frequency_frame_groups[k]):
            # 获取 Rx 和 Tx 对应的文件
            rx_file_path = rx_sorted_files[frame]
            tx_file_path = tx_sorted_files[frame]


            # 读取 Rx 和 Tx 的 I/Q 数据
            rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
            tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
            # 计算复数形式的 Rx 和 Tx 信号
            rx_complex_signal = rx_I_data + 1j * rx_Q_data
            tx_complex_signal = tx_I_data + 1j * tx_Q_data

            # 进行 FFT 并应用校准
            rx_fft_result = fftshift(fft(rx_complex_signal))
            tx_fft_result = fftshift(fft(tx_complex_signal))

            if DC_Calibration[k] != 0:
                rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0]*FFTLength
                tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1]*FFTLength

            # 计算信道响应
            channel_response = rx_fft_result / tx_fft_result
            channel_response_matrix[frame, k, :] = channel_response

        print(f"Frequency band {k + 1}/{num_freq_bands} processed ({index}/{total_files} files).")

    # 构造时间轴和频率轴

    print(f"All {total_files} frames processed for {num_freq_bands} frequency bands.")
    # 根据校准生成文件名

    channel_response_matrix = channel_response_matrix.transpose(1, 2, 0).reshape(len(freq_axis[::-1],), total_frames)



    
    return channel_response_matrix



def apply_square_filter(matrix, freq_window, time_window):
    """
    对 channel_response_matrix 进行方形平滑滤波（滑动均值滤波）
    
    Args:
        matrix (np.ndarray): 原始的 channel_response_matrix，形状为 (freq, time)
        freq_window (int): 频率方向上的平滑窗口大小
        time_window (int): 时间方向上的平滑窗口大小

    Returns:
        np.ndarray: 平滑后的矩阵
    """
    # 生成均值滤波核
    kernel = np.ones((freq_window, time_window)) / (freq_window * time_window)

    # 使用 2D 卷积进行平滑
    smoothed_matrix = convolve2d(np.abs(matrix), kernel, mode='valid', boundary='wrap')

    return smoothed_matrix



def CompareChannelOld(Folder, DC_Calibration, sampling_rate, signalfilter=False,avgminus=False):
    """
    Compare the channel response by dividing the spectra from RxFolder and TxFolder.

    Parameters:
    - RxFolder: str, path to the folder containing Rx data.
    - TxFolder: str, path to the folder containing Tx data.
    - DC_Calibration: list, DC calibration values for each frequency band.
    - sampling_rate: float, sampling rate of the signal.
    - FFTLength: int, FFT length.

    Returns:
    - channel_response_matrix: ndarray, complex channel response for all frames and bands.
    - time_axis: ndarray, time axis for the frames.
    - freq_axis: ndarray, frequency axis for the FFT.
    """
    TxFolder = os.path.join(Folder,"Channel0")#不变的
    RxFolder = os.path.join(Folder,"Channel1")#人动的
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    # 确保 Rx 和 Tx 文件结构一致
    if len(rx_sorted_files) != len(tx_sorted_files):
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The number of files in RxFolder and TxFolder must be the same.")
    if rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        print(rx_frame_length,tx_frame_length)
        raise ValueError("The structure of RxFolder and TxFolder must match.")
        
    # 初始化存储信道响应的矩阵
    total_frames = rx_frame_length * rx_frequency_frame_groups[0]
    num_freq_bands = len(rx_center_freqs)
    channel_response_matrix = np.zeros((total_frames, 1, FFTLength), dtype=complex)

    # 遍历每个中心频率和时间帧，计算信道响应
    index = 0
    total_files = len(rx_sorted_files)
    if not signalfilter:
        time_axis = np.arange(1, total_frames + 1) * FFTLength / sampling_rate
        freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, FFTLength, endpoint=False)
        for k, center_freq in enumerate(rx_center_freqs):
            for frame in range(rx_frame_length * rx_frequency_frame_groups[k]):
                # 获取 Rx 和 Tx 对应的文件
                rx_file_path = rx_sorted_files[frame]
                tx_file_path = tx_sorted_files[frame]


                # 读取 Rx 和 Tx 的 I/Q 数据
                rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
                tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

                # 计算复数形式的 Rx 和 Tx 信号
                rx_complex_signal = rx_I_data + 1j * rx_Q_data
                tx_complex_signal = tx_I_data + 1j * tx_Q_data

                # 进行 FFT 并应用校准
                rx_fft_result = fftshift(fft(rx_complex_signal))
                tx_fft_result = fftshift(fft(tx_complex_signal))

                if DC_Calibration[k] != 0:
                    rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0]*FFTLength
                    tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1]*FFTLength

                # 计算信道响应
                channel_response = rx_fft_result / tx_fft_result
                channel_response_matrix[frame, k, :] = channel_response

            print(f"Frequency band {k + 1}/{num_freq_bands} processed ({index}/{total_files} files).")
    else:
        valid_frames = []
        with open(os.path.join(os.path.dirname(RxFolder), "BothSignal.txt"), "r") as file:
            filtered_frames = [int(line.strip()) for line in file if line.strip().isdigit()]
        
        for k, center_freq in enumerate(rx_center_freqs):
            for frame in range(rx_frame_length * rx_frequency_frame_groups[k]):
                if frame in filtered_frames:
                # 获取 Rx 和 Tx 对应的文件
                    rx_file_path = rx_sorted_files[frame]
                    tx_file_path = tx_sorted_files[frame]
                    # 读取 Rx 和 Tx 的 I/Q 数据
                    rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
                    tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

                    # 计算复数形式的 Rx 和 Tx 信号
                    rx_complex_signal = rx_I_data + 1j * rx_Q_data
                    tx_complex_signal = tx_I_data + 1j * tx_Q_data

                    # 进行 FFT 并应用校准
                    rx_fft_result = fftshift(fft(rx_complex_signal))
                    tx_fft_result = fftshift(fft(tx_complex_signal))

                    if DC_Calibration[k] != 0:
                        rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0]*FFTLength
                        tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1]*FFTLength

                    # 计算信道响应
                    channel_response = rx_fft_result / tx_fft_result
                    channel_response_matrix[frame, k, :] = channel_response
                    valid_frames.append(frame)

            print(f"Frequency band {k + 1}/{num_freq_bands} processed ({index}/{total_files} files).")
    # 构造时间轴和频率轴

    print(f"All {total_files} frames processed for {num_freq_bands} frequency bands.")
    # 根据校准生成文件名
    base_folder_name = os.path.basename(os.path.dirname(RxFolder))
    if avgminus==True:
        avg_value = np.mean(channel_response_matrix)
        print(avg_value)
        channel_response_matrix -= avg_value
    print("meanagain",np.mean(channel_response_matrix))
    if not signalfilter:
        output_dir = 'picture/Channel/phase'
        os.makedirs(output_dir, exist_ok=True)
        if all(calib == 0 for calib in DC_Calibration):  # 无校准
            file_suffix = ".png"
        else:  # 有校准
            file_suffix = "withDCcalibration.png"

        output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

        # 绘制绝对值信道响应
        abs_channel_response = np.angle(channel_response_matrix)
        plot_3d_angle(abs_channel_response, total_frames, time_axis, FFTLength, freq_axis[::-1], output_file)

        output_dir = 'picture/Channel/Amp'
        os.makedirs(output_dir, exist_ok=True)
        if all(calib == 0 for calib in DC_Calibration):  # 无校准
            file_suffix = ".png"
        else:  # 有校准
            file_suffix = "withDCcalibration.png"

        output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

        # 绘制绝对值信道响应
        abs_channel_response = np.abs(channel_response_matrix)
        plot_3d_amp(abs_channel_response, total_frames, time_axis, FFTLength, freq_axis[::-1], output_file)
    else:
        valid_frames_time = np.array(valid_frames)
        time_axis = valid_frames_time * FFTLength / sampling_rate
        valid_frames = np.unique(valid_frames)  # 确保唯一
        filtered_channel_response_matrix = channel_response_matrix[valid_frames, :, :]
        freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, FFTLength, endpoint=False)
        output_dir = 'picture/Channel/phase'
        os.makedirs(output_dir, exist_ok=True)
        if all(calib == 0 for calib in DC_Calibration):  # 无校准
            file_suffix = "OnlySignal.png"
        else:  # 有校准
            file_suffix = "OnlySignalwithDCcalibration.png"

        output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

        # 绘制绝对值信道响应
        abs_channel_response = np.angle(filtered_channel_response_matrix)
        plot_3d_angle(abs_channel_response, len(time_axis), time_axis, FFTLength, freq_axis[::-1], output_file)
        output_dir = 'picture/Channel/Amp'
        os.makedirs(output_dir, exist_ok=True)
        if all(calib == 0 for calib in DC_Calibration):  # 无校准
            file_suffix = "OnlySignal.png"
        else:  # 有校准
            file_suffix = "OnlySignalwithDCcalibration.png"

        output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")
        # 绘制绝对值信道响应
        abs_channel_response = np.abs(filtered_channel_response_matrix)
        plot_3d_amp(abs_channel_response, len(time_axis), time_axis, FFTLength, freq_axis[::-1], output_file)
    return channel_response_matrix, freq_axis, time_axis

def CompareChannelAvgFreq(Folder, DC_Calibration, sampling_rate, signalfilter=False,AvgFreqband = None):
    """
    Compare the channel response by dividing the spectra from RxFolder and TxFolder.

    Parameters:
    - Folder: str, path to the folder containing Channel0 and Channel1 data.
    - DC_Calibration: list, DC calibration values for each frequency band.
    - sampling_rate: float, sampling rate of the signal.

    Returns:
    - channel_response_matrix: ndarray, complex channel response for all frames and bands.
    - time_axis: ndarray, time axis for the frames.
    """
    TxFolder = os.path.join(Folder, "Channel0")#基准
    RxFolder = os.path.join(Folder, "Channel1")#变量
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)

    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    # 确保 Rx 和 Tx 文件结构一致
    if len(rx_sorted_files) != len(tx_sorted_files):
        raise ValueError("The number of files in RxFolder and TxFolder must be the same.")
    if rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        raise ValueError("The structure of RxFolder and TxFolder must match.")

    # 初始化存储信道响应的矩阵
    channel_response_matrix = []
    valid_frames = []
    if AvgFreqband is not None:
        print(AvgFreqband)

    for k, center_freq in enumerate(rx_center_freqs):
        for frame in range(rx_frame_length * rx_frequency_frame_groups[k]):
            
            rx_file_path = rx_sorted_files[frame]
            tx_file_path = tx_sorted_files[frame]
            rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
            tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

            # 计算复数形式的 Rx 和 Tx 信号
            rx_complex_signal = rx_I_data + 1j * rx_Q_data
            tx_complex_signal = tx_I_data + 1j * tx_Q_data

            # 进行 FFT 并应用校准
            rx_fft_result = fftshift(fft(rx_complex_signal))
            tx_fft_result = fftshift(fft(tx_complex_signal))

            if DC_Calibration[k] != 0:
                rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0] * FFTLength
                tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1] * FFTLength

            # 计算信道响应
            channel_response = rx_fft_result / tx_fft_result
            # channel_response_matrix.append(np.mean(channel_response))
            if AvgFreqband is not None:
                startidx = int((AvgFreqband[0]/sampling_rate+0.5)*len(channel_response))
                finishidx = int((AvgFreqband[1]/sampling_rate+0.5)*len(channel_response))
                channel_response[startidx:finishidx]

            real_median = np.median(channel_response.real)
            imag_median = np.median(channel_response.imag)
            median_complex = real_median + 1j * imag_median
            channel_response_matrix.append(median_complex)

            # mean_complex = np.mean(channel_response)
            # channel_response_matrix.append(mean_complex)


            valid_frames.append(frame)

    print(f"All frames processed for {len(rx_center_freqs)} frequency bands.")

    # 构造时间轴
    time_axis = np.array(valid_frames) * FFTLength / sampling_rate
    channel_response_matrix = np.array(channel_response_matrix)

    # 绘制振幅和相位
    # amplitude_output_file = os.path.join(Folder, "Channel_Response_Amplitude.png")
    # phase_output_file = os.path.join(Folder, "Channel_Response_Phase.png")

    # plt.figure(figsize=(10, 6))
    # plt.plot(time_axis, np.abs(channel_response_matrix), label="Amplitude")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("Channel Response Amplitude")
    # plt.grid()
    # plt.legend()
    # plt.savefig(amplitude_output_file)
    # plt.close()
    # print(f"Amplitude plot saved to {amplitude_output_file}")

    # plt.figure(figsize=(10, 6))
    # plt.plot(time_axis, np.angle(channel_response_matrix), label="Phase")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Phase (radians)")
    # plt.title("Channel Response Phase")
    # plt.grid()
    # plt.legend()
    # plt.savefig(phase_output_file)
    # plt.close()
    # print(f"Phase plot saved to {phase_output_file}")

    return channel_response_matrix, time_axis


def plot_3d_angle(angle_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file):
    # 展开矩阵，用于绘图
    angle_matrix_expanded = angle_matrix.transpose(1, 2, 0).reshape(len(freq_axis), FrameNumber)

    # 检查维度匹配
    if angle_matrix_expanded.shape != (len(freq_axis), len(time_axis)):
        raise ValueError(
            f"Matrix shape {angle_matrix_expanded.shape} does not match freq_axis ({len(freq_axis)}) "
            f"and time_axis ({len(time_axis)})."
        )

    freq_axis_file = "freq_axis_output.txt"
    with open(freq_axis_file, "w") as f:
        for freq in freq_axis:
            f.write(f"{freq}\n")
    print(f"Frequency axis saved to {freq_axis_file}")

    # 设置字体大小
    plt.rcParams.update({'font.size': 16})  # 全局字体大小设置

    # 绘制 3D 角度图像
    plt.figure(figsize=(10, 6))  # 调整图像尺寸
    plt.imshow(
        angle_matrix_expanded, aspect='auto', cmap='plasma', vmin=-1*np.pi, vmax=1*np.pi,
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    )
    plt.colorbar(label='Angle (radians)', pad=0.02, shrink=0.8)  # 调整颜色条的位置和大小
    plt.title('Angle Plot', fontsize=20)  # 修改标题
    plt.ylabel('Frequency (MHz)', fontsize=18)  # 加大Y轴标签字体
    plt.xlabel('Time (s)', fontsize=18)  # 加大X轴标签字体
    plt.xticks(fontsize=16)  # 调整X轴刻度字体
    plt.yticks(fontsize=16)  # 调整Y轴刻度字体

    # 调整图像布局以减少留白
    plt.tight_layout()

    # 保存绘图
    plt.savefig(output_file, bbox_inches='tight')  # 去掉多余留白
    print(f"Plot saved to {output_file}")
    plt.close()
def Average_freq_plot(folder_name, sample_rate):
    """
    Calculates and plots frequency offsets and variances across time for I/Q data files.

    Parameters:
    - folder_name: str, folder containing the I/Q data files
    - Start_point, Start_diff, Final_point: int, FFT windowing parameters
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each FFT segment
    - peaknumber: int, number of peaks to extract
    - output: list, placeholder for FFT output
    - framejumppoint: int, number of frames to skip during processing (default is 0, meaning process every frame)

    Returns:
    - frequency_offsets: list of calculated frequency offsets
    - variances: list of calculated variances
    """

    # 调用 process_filesIntimedomain 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # 初始化存储每帧频率偏移和方差的数组
    frequency = []  # 存储 FrequencyOffset 的数组

    # 获取文件夹的基本名称
    base_folder_name = os.path.basename(os.path.normpath(folder_name))

    # 创建目标目录
    output_dir = os.path.join("picture", "avgfreq")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个 bin 文件
    frame_count = 0  # 用于时间轴的计数
    with tqdm(total=len(sorted_files), desc="Processing files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files):
            # 如果 framejumppoint > 0，跳过指定的帧
            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)
            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data
            average_angel = np.mean(np.angle(complex_signal[1:-1]/complex_signal[0:-2]))
            average_frequency = average_angel*sample_rate/(2*np.pi)

            # 更新进度条和时间计数
            frequency.append(average_frequency)
            progress_bar.update(1)

    # 根据 FFTLength 和采样率计算新的时间轴（考虑跳跃）
    time_axis = np.arange(len(sorted_files)) * FFTLength / sample_rate

    # 绘制 FrequencyOffset 随时间的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, frequency, label="Average Frequency", marker="o", color="blue")
    plt.title("Average Frequency vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid()
    plt.legend()
    output_file_avg_freq = os.path.join(output_dir, f"{base_folder_name}_average_frequency.png")
    plt.savefig(output_file_avg_freq)
    print(f"Average Frequency plot saved to {output_file_avg_freq}")
    plt.close()

    return 

def Average_freq_plotminus(folder_name, sample_rate):
    """
    Calculates and plots frequency offsets and variances across time for I/Q data files.

    Parameters:
    - folder_name: str, folder containing the I/Q data files
    - Start_point, Start_diff, Final_point: int, FFT windowing parameters
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each FFT segment
    - peaknumber: int, number of peaks to extract
    - output: list, placeholder for FFT output
    - framejumppoint: int, number of frames to skip during processing (default is 0, meaning process every frame)

    Returns:
    - frequency_offsets: list of calculated frequency offsets
    - variances: list of calculated variances
    """

    # 调用 process_filesIntimedomain 获取文件信息
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(folder_name)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    adc_step = 0.000195313  # ADC 步长
    resistance = 50  # 50 ohms
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # 初始化存储每帧频率偏移和方差的数组
    frequency = []  # 存储 FrequencyOffset 的数组

    # 获取文件夹的基本名称
    base_folder_name = os.path.basename(os.path.normpath(folder_name))

    # 创建目标目录
    output_dir = os.path.join("picture", "avgfreq")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历每个 bin 文件
    frame_count = 0  # 用于时间轴的计数
    with tqdm(total=len(sorted_files), desc="Processing files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files[2:]):
            # 如果 framejumppoint > 0，跳过指定的帧
            # 读取 I/Q 数据
            I_data, Q_data = read_bin_file(bin_file_path)
            # 计算复数形式的信号
            complex_signal = I_data + 1j * Q_data - (-0.492472 -0.492484*1j)
            average_angel = (np.mean(np.unwrap(np.angle(complex_signal[1:-1]))-np.unwrap(np.angle(complex_signal[0:-2]))))%(2*np.pi)
            average_frequency = average_angel
            # 更新进度条和时间计数
            frequency.append(average_frequency)
            progress_bar.update(1)

    # 根据 FFTLength 和采样率计算新的时间轴（考虑跳跃）
    time_axis = np.arange(len(frequency)) * FFTLength / sample_rate

    # 绘制 FrequencyOffset 随时间的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, frequency, label="Average Frequency", marker="o", color="blue")
    plt.title("Average Frequency vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid()
    plt.legend()
    output_file_avg_freq = os.path.join(output_dir, f"{base_folder_name}_average_frequencyminusAfterDC.png")
    plt.savefig(output_file_avg_freq)
    print(f"Average Frequency plot saved to {output_file_avg_freq}")
    plt.close()
    return 0

def CalculateDopplerWithHamming(real_signal, Start_point, Start_diff, Final_point, sample_rate, resolution):
    """
    Perform Short-Time Fourier Transform (STFT) on a signal, save two plots:
    1. Raw FFT result (frequency from -f to f).
    2. Processed result with velocity calculation (limited to 0-100 Hz).

    Parameters:
    - real_signal: ndarray, the real signal (time-domain data)
    - Start_point: int, starting index of the analysis window
    - Start_diff: int, step size for moving the window
    - Final_point: int, end point of the signal analysis
    - sample_rate: float, sampling rate of the signal
    - resolution: int, size of each window for FFT

    Returns:
    - dbm_results: list of dBm values for each window
    """
    # Constants for dBm calculation
    adc_step = 0.000195313  # ADC step size (0.8V range, 12-bit ADC)
    resistance = 50         # Ohm
    FFTLength = resolution // 2  # Effective FFT length
    K_FFT = (adc_step / FFTLength) ** 2 / resistance / 0.001

    # Initialize results container
    dbm_results = []
    time_axis = []  # To track time points

    # Generate Hamming window
    hamming_window = windows.hamming(resolution)

    # Iteratively process each window
    i = 0
    while True:
        start_idx = Start_point + i * Start_diff
        end_idx = start_idx + resolution

        # Stop if the window exceeds Final_point
        if end_idx > Final_point:
            break

        # Extract the current window
        window_signal = real_signal[start_idx:end_idx]

        # Remove DC offset (mean)
        window_signal = window_signal - np.mean(window_signal)

        # Apply Hamming window
        window_signal = window_signal * hamming_window

        # Perform FFT and FFTShift
        fft_result = fft(window_signal)
        fft_result = fftshift(fft_result)
        fft_magnitude = np.abs(fft_result)

        # Calculate dBm values
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2)
        dbm_results.append(fft_dbm)

        # Record time point
        time_point = start_idx / sample_rate
        time_axis.append(time_point)

        # Move to the next window
        i += 1

    # Prepare data for plotting
    max_length = max(len(dbm) for dbm in dbm_results)
    dbm_results_padded = [np.pad(dbm, (0, max_length - len(dbm)), constant_values=np.nan) for dbm in dbm_results]
    freq_axis = np.linspace(-sample_rate/2, sample_rate/2, max_length)

    # Create output directory
    output_dir = os.path.join("picture", "doppler")
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Raw FFT result (-f to f)
    output_file_fft = os.path.join(output_dir, "stft_doppler_raw_fft.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.array(dbm_results_padded).T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )

    plt.colorbar(label='Power (dBm)')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Frequency (Hz)', fontsize=18)
    plt.title('Raw FFT (Full Frequency Range)', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(output_file_fft, bbox_inches='tight')
    plt.close()
    print(f"Raw FFT plot saved to {output_file_fft}")

    # Plot 2: Velocity calculation (0-100 Hz range)
    freq_mask = (freq_axis >= 0) & (freq_axis <= 80)
    dbm_results_filtered = np.array(dbm_results_padded)[:, freq_mask]
    freq_axis_filtered = freq_axis[freq_mask]

    output_file_velocity = os.path.join(output_dir, "stft_doppler_velocity_0_100Hz.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(
        dbm_results_filtered.T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], freq_axis_filtered[0]*0.012*np.pi, freq_axis_filtered[-1]*0.012*np.pi],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label='Power (dBm)')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Velocity (m/s)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.savefig(output_file_velocity, bbox_inches='tight')
    plt.close()
    print(f"Doppler plot saved to {output_file_velocity}")

    return dbm_results


def generate_micro_doppler(signal, fs, window='hann', nperseg=256, noverlap=128):
    """
    Generate a Micro-Doppler plot with STFT and phase-based analysis.

    Parameters:
    - signal: ndarray, input signal
    - fs: float, sampling frequency
    - window: str, window type for STFT
    - nperseg: int, window length
    - noverlap: int, overlap between windows

    Returns:
    - None (plots the Micro-Doppler result)
    """
    # Perform STFT
    f, t, Zxx = stft(signal, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)  # Magnitude spectrum
    phase = np.angle(Zxx)    # Phase spectrum

    # Compute instantaneous frequency
    inst_freq = np.diff(np.unwrap(phase, axis=0), axis=0) / (2 * np.pi * (t[1] - t[0]))

    # Map instantaneous frequency back to frequency bins
    freq_bin_size = f[1] - f[0]
    freq_centers = f[:-1] + freq_bin_size / 2
    inst_freq_mapped = inst_freq * fs / freq_bin_size  # Scale frequency

    # Smooth and interpolate for Micro-Doppler visualization
    micro_doppler = np.zeros_like(magnitude)
    for i in range(len(t) - 1):
        # Map instantaneous frequency to corresponding bins
        for j in range(len(inst_freq_mapped[:, i])):
            freq_idx = int((inst_freq_mapped[j, i] - f[0]) / freq_bin_size)
            if 0 <= freq_idx < len(f):
                micro_doppler[freq_idx, i] += magnitude[j, i]

    # Plot the Micro-Doppler result
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(micro_doppler + 1e-6), shading='gouraud', cmap='jet')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Micro-Doppler Plot')
    plt.show()
def SelectAvgdbm(folder_name): 
    # 处理 Channel0
    channel0_folder = os.path.join(folder_name, "Channel0")
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(channel0_folder)

    # 初始化 FFT 参数
    with open(sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2
    print(FFTLength)
    avg_dbm_values = []

    # 遍历每个 bin 文件
    with tqdm(total=len(sorted_files), desc="Processing Channel0 files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files):
            I_data, Q_data = read_bin_file(bin_file_path)
            complex_signal = I_data + 1j * Q_data
            fft_result = fftshift(fft(complex_signal))
            avg_dbm_values.append(np.mean(np.abs(fft_result)))
            progress_bar.update(1)

    overall_avg = np.mean(avg_dbm_values)
    frames_channel0 = [idx for idx, value in enumerate(avg_dbm_values) if value > overall_avg]

    # 保存 Channel0 的信号帧编号
    channel0_signal_file = os.path.join(folder_name, "Channel0haveSignal.txt")
    with open(channel0_signal_file, "w") as txt_file:
        for frame in frames_channel0:
            txt_file.write(f"{frame}\n")
    print(f"Channel0 frames saved to {channel0_signal_file}")

    # 处理 Channel1
    channel1_folder = os.path.join(folder_name, "Channel1")
    sorted_files, center_freqs, frame_length, frequency_frame_groups = process_filesIntimedomain(channel1_folder)

    avg_dbm_values2 = []
    with tqdm(total=len(sorted_files), desc="Processing Channel1 files") as progress_bar:
        for idx, bin_file_path in enumerate(sorted_files):
            I_data, Q_data = read_bin_file(bin_file_path)
            complex_signal = I_data + 1j * Q_data
            fft_result = fftshift(fft(complex_signal))
            avg_dbm_values2.append(np.mean(np.abs(fft_result)))
            progress_bar.update(1)

    overall_avg2 = np.mean(avg_dbm_values2)
    frames_channel1 = [idx for idx, value in enumerate(avg_dbm_values2) if value > overall_avg2]

    # 保存 Channel1 的信号帧编号
    channel1_signal_file = os.path.join(folder_name, "Channel1haveSignal.txt")
    with open(channel1_signal_file, "w") as txt_file:
        for frame in frames_channel1:
            txt_file.write(f"{frame}\n")
    print(f"Channel1 frames saved to {channel1_signal_file}")

    # 对比两个通道的信号帧，找到共同的帧编号
    common_frames = sorted(set(frames_channel0) & set(frames_channel1))

    # 保存共有帧编号到 BothSignal.txt
    both_signal_file = os.path.join(folder_name, "BothSignal.txt")
    with open(both_signal_file, "w") as txt_file:
        for frame in common_frames:
            txt_file.write(f"{frame}\n")
    print(f"Common frames saved to {both_signal_file}")

    return common_frames



def CompareChanneldetail(Folder,Start_diff,resolution, sampling_rate, signal_frame, DC_Calibration = [0,0]):

    TxFolder = os.path.join(Folder,"Channel0")
    RxFolder = os.path.join(Folder,"Channel1")
    
    # 获取 Rx 和 Tx 文件夹中的文件路径和元数据
    rx_sorted_files = process_files_fast(RxFolder)
    tx_sorted_files = process_files_fast(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)
    rx_complex_signal = np.array([], dtype=np.complex64)

    tx_complex_signal = np.array([], dtype=np.complex64)
    FFTLength = resolution  # 每个时间帧的 FFT 长度
    # 确保 Rx 和 Tx 文件结构一致
    # 初始化存储信道响应的矩阵

    for frame in signal_frame:#连接signal frame的所有时间帧

        # 获取 Rx 和 Tx 文件路径
        rx_file_path = rx_sorted_files[frame]
        tx_file_path = tx_sorted_files[frame]
        
        # 读取 Rx 和 Tx 的 I/Q 数据
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
        
        # 计算复数形式的 Rx 和 Tx 信号
        rx_complex_signal_frame = rx_I_data + 1j * rx_Q_data
        tx_complex_signal_frame = tx_I_data + 1j * tx_Q_data
        # 拼接当前帧的信号
        rx_complex_signal = np.concatenate((rx_complex_signal, rx_complex_signal_frame))#timedomain
        tx_complex_signal = np.concatenate((tx_complex_signal, tx_complex_signal_frame))
    
    total_frames = (len(rx_complex_signal)-resolution)//Start_diff #所有帧内跳数。
    channel_response_matrix = np.zeros((total_frames, 1, resolution), dtype=complex)

    # 遍历每个中心频率和时间帧，计算信道响应
    index = 0
    for i in range(total_frames):
        signal_segment_rx = rx_complex_signal[i*Start_diff:i*Start_diff+resolution]
        signal_segment_tx = tx_complex_signal[i*Start_diff:i*Start_diff+resolution]
        rx_fft_result = fftshift(fft(signal_segment_rx))
        tx_fft_result = fftshift(fft(signal_segment_tx))
        if DC_Calibration[0] != 0:
            rx_fft_result[0] = rx_fft_result[0] - DC_Calibration[0]*resolution
            tx_fft_result[0] = tx_fft_result[0] - DC_Calibration[1]*resolution

        # 计算信道响应
        channel_response = rx_fft_result / tx_fft_result
        channel_response_matrix[i, 0, :] = channel_response

    # 根据校准生成文件名
    base_folder_name = os.path.basename(os.path.dirname(RxFolder))

    valid_frames_time = np.arange(total_frames)
    time_axis = valid_frames_time * Start_diff / sampling_rate
    freq_axis = np.linspace(-sampling_rate / 2, sampling_rate / 2, resolution, endpoint=False)
    output_dir = 'picture/Channel/phase'
    os.makedirs(output_dir, exist_ok=True)
    file_suffix = "HighresolutionDC.png"
    output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")

    # 绘制绝对值信道响应
    abs_channel_response = np.angle(channel_response_matrix)
    plot_3d_angle(abs_channel_response, len(time_axis), time_axis, FFTLength, freq_axis[::-1], output_file)

    output_dir = 'picture/Channel/Amp'
    os.makedirs(output_dir, exist_ok=True)
    file_suffix = "HighresolutionDC.png"

    output_file = os.path.join(output_dir, f"{base_folder_name}_channel_response{file_suffix}")
    # 绘制绝对值信道响应
    abs_channel_response = np.abs(channel_response_matrix)
    plot_3d_amp(abs_channel_response, len(time_axis), time_axis, FFTLength, freq_axis[::-1], output_file)

    return channel_response_matrix, freq_axis, time_axis

def CompareChanneldetailSpecificFreqPlot(Folder, Start_diff, resolution, sampling_rate, selected_frequency, signal_frame,skipfirst=False,TakeAvg=False):
    # 获取 Rx 和 Tx 文件夹中的文件路径和元数据
    TxFolder = os.path.join(Folder,"Channel0")
    RxFolder = os.path.join(Folder,"Channel1")
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)

    # 构建频率轴并验证所选频率

    freq_axis = np.fft.fftfreq(resolution, d=1 / sampling_rate)

    if selected_frequency < freq_axis.min() or selected_frequency > freq_axis.max():
        raise ValueError(f"SelectedFrequency {selected_frequency} Hz is out of range [{freq_axis.min()}, {freq_axis.max()}] Hz.")

    # 找到目标频率对应的频率轴索引
    selected_index = np.argmin(np.abs(freq_axis - selected_frequency))
    print("selected_index=",selected_index)
    n = np.arange(resolution)  # 时域采样点索引
    e_term = np.exp(-1j * 2 * np.pi * selected_index * n / resolution)  # DFT 权值计算

    # 初始化 Rx 和 Tx 信号
    channel_response_matrix=[]
    framenum = 0
    total_frames = []
    segment_results = []  # 存储每段信号段的信道响应
    totalframenum = 0
    long1frame = 0
    # 拼接所有帧的 Rx 和 Tx 信号
    for signal_fragment in signal_frame:
        rx_complex_signal = np.array([], dtype=np.complex64)
        tx_complex_signal = np.array([], dtype=np.complex64)
        if skipfirst == True:
            if len(signal_fragment)<=2:
                long1frame += 1
                continue
            else:
                signal_fragment = signal_fragment[1:]

        for frame in signal_fragment:
            rx_file_path = rx_sorted_files[frame]
            tx_file_path = tx_sorted_files[frame]
            rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
            tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
            rx_complex_signal_frame = rx_I_data + 1j * rx_Q_data
            tx_complex_signal_frame = tx_I_data + 1j * tx_Q_data
            rx_complex_signal = np.concatenate((rx_complex_signal, rx_complex_signal_frame))
            tx_complex_signal = np.concatenate((tx_complex_signal, tx_complex_signal_frame))

        # 计算当前段的帧数
        framesnumberinthisSegment = (len(rx_complex_signal) - resolution) // Start_diff

        if framesnumberinthisSegment<0:
            continue
        total_frames.append(framesnumberinthisSegment)  # 每段信号段中的帧数
        totalframenum += framesnumberinthisSegment

        # 初始化当前段的信道响应结果
        current_segment_response = np.zeros(framesnumberinthisSegment, dtype=np.complex64)

        # 遍历每个时间帧，计算特定频率的信道响应
        for i in range(framesnumberinthisSegment):
            signal_segment_rx = rx_complex_signal[i * Start_diff:i * Start_diff + resolution]
            signal_segment_tx = tx_complex_signal[i * Start_diff:i * Start_diff + resolution]
            rx_fft_result = np.sum(signal_segment_rx * e_term)
            tx_fft_result = np.sum(signal_segment_tx * e_term)

            # 计算信道响应并存入当前段结果
            current_segment_response[i] = rx_fft_result / tx_fft_result

        # 将当前段结果存储到结果列表中
        if TakeAvg == True:
            segment_results.append(np.array([np.mean(current_segment_response)]))
        else:
            segment_results.append(current_segment_response)
            

    # 将所有段结果拼接成一维数组
    if TakeAvg == True:
        channel_response_matrix = np.array(segment_results)
        channel_response_matrix = channel_response_matrix.flatten()
    else:
        channel_response_matrix = np.concatenate(segment_results)
    
    with open(rx_sorted_files[0], "rb") as f:
        total_Data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_Data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    # 生成时间轴
    time_axis = []  # 初始化时间轴
    for segment_idx, segment_response in enumerate(segment_results):
        segment_start_time = signal_frame[segment_idx][0] * FFTLength / sampling_rate  # 当前段的起始时间
        frame_times = np.arange(len(segment_response)) * Start_diff / sampling_rate  # 当前段内每帧的时间
        segment_time_axis = segment_start_time + frame_times  # 当前段完整的时间轴
        time_axis.extend(segment_time_axis)  # 将当前段的时间轴加入总时间轴

    time_axis = np.array(time_axis)  # 转换为 NumPy 数组

    # 获取母文件夹路径
    parent_folder = os.path.dirname(RxFolder)
    print("len(channel_response_matrix)=",len(channel_response_matrix))
    # 绘图：幅度
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.abs(channel_response_matrix), label="Amplitude")
    plt.title(f"Channel Response Amplitude at {selected_frequency / 1e6:.2f} MHz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    output_file_amp = os.path.join(parent_folder, f"Longamp_{selected_frequency / 1e6:.2f}MHz.png")
    plt.savefig(output_file_amp)
    plt.close()
    #print(np.unwrap(np.angle(channel_response_matrix)))
    # 绘图：相位
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.unwrap(np.angle(channel_response_matrix)), label="Phase")
    plt.title(f"Channel Response Phase at {selected_frequency / 1e6:.2f} MHz")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.grid()
    plt.legend()
    output_file_phase = os.path.join(parent_folder, f"Longphase_{selected_frequency / 1e6:.2f}MHz.png")
    plt.savefig(output_file_phase)
    plt.close()

    # 输出文件路径信息
    plt.figure(figsize=(6, 6))  # 设置外部图像为正方形
    plt.scatter(channel_response_matrix.real, channel_response_matrix.imag, s=1, label="Complex Numbers")
    plt.title(f"Channel Response Complex Distribution at {selected_frequency / 1e6:.2f} MHz")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")

    # 强制设置 x 和 y 轴范围
    plt.xlim(-15, 15)  # x 轴范围 [-15, 15]
    plt.ylim(-15, 15)  # y 轴范围 [-15, 15]
    # 确保比例一致且范围不会被自动调整
    plt.gca().set_aspect('equal', adjustable='box')  # 强制内容正方形
    plt.grid()
    plt.legend()
    output_file_complex = os.path.join(parent_folder, f"Longcomplex_{selected_frequency / 1e6:.2f}MHz.png")
    plt.savefig(output_file_complex)
    plt.close()

    # 输出文件路径信息
    print(f"Amplitude plot saved to: {output_file_amp}")
    print(f"Phase plot saved to: {output_file_phase}")
    print(f"Complex plot saved to: {output_file_complex}")
    return channel_response_matrix, time_axis


def extract_frequency_response(channel_response_matrix, freq_axis, selected_frequency):
    """
    Extract the complex response of a specific frequency across all time frames.

    Parameters:
    - channel_response_matrix: ndarray, the channel response matrix (time x freq)
    - freq_axis: ndarray, the frequency axis corresponding to FFTLength
    - selected_frequency: float, the target frequency to extract

    Returns:
    - frequency_response: ndarray, complex response of the selected frequency over time
    """
    # 检查频率范围
    if selected_frequency < freq_axis.min() or selected_frequency > freq_axis.max():
        raise ValueError(f"Selected frequency {selected_frequency} Hz is out of range [{freq_axis.min()}, {freq_axis.max()}] Hz.")
    
    # 找到目标频率的索引
    selected_index = np.argmin(np.abs(freq_axis - selected_frequency))
    print(f"Selected frequency index: {selected_index}")

    # 提取目标频率的复数响应
    frequency_response = channel_response_matrix[:, 0, selected_index]  # 第二维 num_freq_bands 是 1
    return frequency_response

def ChannelSpecificFreq(Folder, sampling_rate, selected_frequency):
    """
    针对某一频率，对 Rx 和 Tx 文件夹中的每一帧信号计算信道响应并绘制散点图。
    
    Args:
        Folder (str): 数据文件夹路径
        sampling_rate (float): 采样率 (Hz)
        selected_frequency (float): 目标频率 (Hz)
    """

    # 获取 Tx 和 Rx 文件夹路径
    TxFolder = os.path.join(Folder, "Channel0")
    RxFolder = os.path.join(Folder, "Channel1")
    
    # 处理 Rx 和 Tx 文件夹中的文件
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)

    FFTLength = total_data_count // 2  # 每个时间帧的 FFT 长度
    print("FFTLeng",FFTLength)
    # 构建频率轴并验证所选频率
    freq_axis = np.fft.fftfreq(FFTLength, d=1 / sampling_rate)
    if selected_frequency < freq_axis.min() or selected_frequency > freq_axis.max():
        raise ValueError(f"Selected frequency {selected_frequency} Hz is out of range [{freq_axis.min()}, {freq_axis.max()}] Hz.")
    
    # 找到目标频率对应的频率轴索引
    selected_index = np.argmin(np.abs(freq_axis - selected_frequency))
    print(f"Selected frequency index: {selected_index}")

    # 计算 DFT 权重 (eterm)
    n = np.arange(FFTLength)  # 时域采样点索引
    e_term = np.exp(-1j * 2 * np.pi * selected_index * n / FFTLength)  # DFT 权值

    # 初始化复数结果列表
    channel_responses = []

    # 遍历文件夹中的每一帧数据
    for rx_file_path, tx_file_path in zip(rx_sorted_files, tx_sorted_files):
        # 读取 Rx 和 Tx 数据
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
        
        # 组合成复数信号
        rx_complex_signal = rx_I_data + 1j * rx_Q_data
        tx_complex_signal = tx_I_data + 1j * tx_Q_data

        # 检查数据长度是否一致
        if len(rx_complex_signal) != len(tx_complex_signal):
            print(f"Skipping file pair {rx_file_path} and {tx_file_path}: Rx and Tx data lengths do not match.")
            continue

        # 使用 DFT 权值计算目标频率的信号
        rx_fft_result = np.sum(rx_complex_signal * e_term)
        tx_fft_result = np.sum(tx_complex_signal * e_term)

        # 计算信道响应
        if tx_fft_result != 0:  # 避免除零
            channel_response = rx_fft_result / tx_fft_result
            channel_responses.append(channel_response)
        else:
            print(f"Skipping file pair {rx_file_path} and {tx_file_path}: Tx FFT result is zero.")
            continue

    # 转换为 NumPy 数组
    channel_responses = np.array(channel_responses)
    print(len(channel_responses))
    # 绘制复数信道响应的散点图
    # plt.figure(figsize=(6, 6))
    # plt.scatter(channel_responses.real, channel_responses.imag, s=5, label=f"Frequency: {selected_frequency/1e6:.2f} MHz")
    # plt.title(f"Complex Response Scatter Plot at {selected_frequency / 1e6:.2f} MHz")
    # plt.xlabel("Real Part")
    # plt.ylabel("Imaginary Part")
    # plt.grid()
    # plt.gca().set_aspect('equal', adjustable='box')  # 确保比例一致
    # plt.legend()
    # plt.show()
    return channel_responses

def readtxt_to_bin(foldername, output_filename="BothSignal.bin"):
    """
    从 BothSignal.txt 读取数字，并保存为二进制文件。
    
    Args:
        foldername (str): 包含 BothSignal.txt 文件的文件夹路径。
        output_filename (str): 输出二进制文件的文件名（默认是 BothSignal.bin）。
    
    Returns:
        str: 二进制文件的完整路径。
    """
    # 定位 BothSignal.txt 文件路径
    input_filepath = os.path.join(os.path.dirname(foldername), "BothSignal.txt")
    output_filepath = os.path.join(os.path.dirname(foldername), output_filename)
    
    # 读取 BothSignal.txt 文件中的数字
    with open(input_filepath, "r") as file:
        numbers = [int(line.strip()) for line in file if line.strip().isdigit()]
    
    if not numbers:
        raise ValueError("BothSignal.txt 文件中没有有效的数字！")
    
    # 根据最大值动态创建位图
    max_value = max(numbers)
    bitmap = np.zeros(max_value, dtype=np.uint8)
    
    for num in numbers:
        bitmap[num - 1] = 1  # 数字 1 对应索引 0

    # 保存位图为二进制文件
    with open(output_filepath, "wb") as f:
        f.write(bitmap)
    
    print(f"二进制文件已保存到: {output_filepath}")
    return output_filepath

def returnfirstsignalframe(arr):
    """
    返回数组中第一个值为 1 的位置。

    Args:
        arr (numpy.ndarray): 输入的数组。

    Returns:
        int: 第一个值为 1 的索引（从 1 开始）。如果没有值为 1，则返回 -1。
    """
    if not isinstance(arr, (list, np.ndarray)):
        raise ValueError("输入必须是一个列表或 numpy 数组！")

    # 找到第一个值为 1 的索引
    for idx, value in enumerate(arr):
        if value == 1:
            return idx+1  # 索引从 1 开始

    # 如果数组中没有 1，返回 -1
    return -1

def read_bin_to_array(foldername, input_filename="BothSignal.bin"):
    """
    从二进制文件读取内容并还原为位图数组。
    
    Args:
        foldername (str): 包含 BothSignal.bin 文件的文件夹路径。
        input_filename (str): 输入二进制文件的文件名（默认是 BothSignal.bin）。
    
    Returns:
        list: 解码后的位图数组。
    """
    # 定位二进制文件路径
    input_filepath = os.path.join(os.path.dirname(foldername), input_filename)
    
    # 读取二进制文件
    with open(input_filepath, "rb") as f:
        bitmap = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 转换为列表
    bitmap_list = bitmap.tolist()
    return bitmap_list

def longestframe(foldername):
    # 读取 BothSignal.txt 中的数字作为 filtered_frames
    with open(os.path.join(os.path.dirname(foldername), "BothSignal.txt"), "r") as file:
        filtered_frames = [int(line.strip()) for line in file if line.strip().isdigit()]
    
    print(f"Filtered frames: {filtered_frames}")

    # 查找连续最长的子数组
    if not filtered_frames:
        return []

    longest_sequence = []
    current_sequence = [filtered_frames[0]]

    for i in range(1, len(filtered_frames)):
        if filtered_frames[i] == filtered_frames[i - 1] + 1:
            current_sequence.append(filtered_frames[i])
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = [filtered_frames[i]]

    # 检查最后一段序列
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence
    return [longest_sequence]

def generate_complex_scatter_video(channel_matrix, time, output_file="./video/channel_scatter.mp4", slowdown_factor=10, fps=30):
    """
    生成视频展示 channel_matrix 的复数随时间变化
    Args:
        channel_matrix (np.ndarray): 一维复数数组，包含信道响应的复数值
        time (np.ndarray): 一维时间数组，对应每个复数值的时间戳
        output_file (str): 输出视频文件路径，默认 ./video/channel_scatter.mp4
        slowdown_factor (float): 慢放倍数，时间轴会被拉伸 slowdown_factor 倍
        fps (int): 视频的帧率（frames per second）
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 时间区间（慢放后每帧时间对应的实际时间范围）
    frame_duration = 1 / fps  # 每帧的时间间隔（秒）
    playback_frame_duration = frame_duration / slowdown_factor  # 慢放后每帧覆盖的时间范围
    print("playback_frame_duration=",playback_frame_duration)
    # 创建绘图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(
        min(channel_matrix.real) - 1, 
        max(channel_matrix.real) + 1
    )
    ax.set_ylim(
        min(channel_matrix.imag) - 1, 
        max(channel_matrix.imag) + 1
    )
    ax.set_title("Complex Scatter Plot Over Time")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.grid()

    # 初始化空散点
    scatter = ax.scatter([], [], s=5, label="Complex Numbers")
    ax.legend()

    # 初始化动画帧
    def init():
        scatter.set_offsets([])
        return scatter,

    # 更新每一帧的内容
    def update(frame):
        # 当前帧的时间范围（慢放后）
        start_time = frame * playback_frame_duration
        end_time = (frame + 1) * playback_frame_duration

        # 获取属于当前时间范围的索引
        indices = np.where((time >= start_time) & (time < end_time))[0]

        # 提取对应的复数数据
        current_real = channel_matrix.real[indices]
        current_imag = channel_matrix.imag[indices]

        # 更新散点
        scatter.set_offsets(np.c_[current_real, current_imag])
        return scatter,

    # 总帧数根据慢放后的视频总时间计算
    total_frames = int(np.ceil((np.max(time) * slowdown_factor) / frame_duration))

    # 创建动画
    ani = FuncAnimation(
        fig, update, frames=total_frames, init_func=init,
        blit=True, interval=1000 / fps  # 每帧间隔（毫秒）
    )

    # 保存动画到文件
    ani.save(output_file, fps=fps, writer="ffmpeg")
    plt.close(fig)

    print(f"Video saved to {output_file}")

def readFilter(foldername):
    """
    从指定路径的 BothSignal.txt 文件中读取所有数字，并将连续的数字分组为二维数组。

    参数：
        foldername (str): 文件所在文件夹的路径。

    返回：
        list[list[int]]: 包含连续数字的二维数组。
    """
    # 获取文件路径
    file_path = os.path.join(os.path.dirname(foldername), "BothSignal.txt")

    # 读取文件并提取所有数字
    with open(file_path, "r") as file:
        filtered_frames = [int(line.strip()) for line in file if line.strip().isdigit()]


    # 如果没有数字，返回空数组
    if not filtered_frames:
        return []

    # 分组连续的数字
    grouped_sequences = []
    current_sequence = [filtered_frames[0]]

    for i in range(1, len(filtered_frames)):
        if filtered_frames[i] == filtered_frames[i - 1] + 1:
            current_sequence.append(filtered_frames[i])
        else:
            grouped_sequences.append(current_sequence)
            current_sequence = [filtered_frames[i]]

    # 添加最后一段序列
    grouped_sequences.append(current_sequence)

    return grouped_sequences
def plot_complex_scatter(channel_matrix):
    
    """
    绘制复数数组的散点图
    Args:
        channel_matrix (np.ndarray): 复数数组
    """
    plt.figure(figsize=(6, 6))  # 设置外部图像为正方形
    plt.scatter(channel_matrix.real, channel_matrix.imag, s=1, label="Complex Numbers")
    plt.title("Channel Response Complex Distribution")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")

    # 强制设置 x 和 y 轴范围
    plt.xlim(-1.5, 1.5)  # x 轴范围 [-1.5, 1.5]
    plt.ylim(-1.5, 1.5)  # y 轴范围 [-1.5, 1.5]
    # 确保比例一致且范围不会被自动调整
    plt.gca().set_aspect('equal', adjustable='box')  # 强制内容正方形
    plt.grid()
    plt.legend()
    plt.show()  # 直


def plot_complex_scatter_savefile(channel_matrix, output_file):
    """
    绘制复数数组的散点图
    Args:
        channel_matrix (np.ndarray): 复数数组
        output_file (str): 输出图片路径
    """
    plt.figure(figsize=(6, 6))  # 设置外部图像为正方形
    plt.scatter(channel_matrix.real, channel_matrix.imag, s=1, label="Complex Numbers")
    plt.title("Channel Response Complex Distribution")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")

    # 强制设置 x 和 y 轴范围
    plt.xlim(-1.5, 1.5)  # x 轴范围 [-15, 15]
    plt.ylim(-1.5, 1.5)  # y 轴范围 [-15, 15]
    # 确保比例一致且范围不会被自动调整
    plt.gca().set_aspect('equal', adjustable='box')  # 强制内容正方形
    plt.grid()
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Complex scatter plot saved to {output_file}")

def save_channel_data(file_path, channel_matrix, time):
    """
    保存 channel_matrix 和 time 到 .npz 文件
    Args:
        file_path (str): 保存的文件路径（建议以 .npz 结尾）
        channel_matrix (np.ndarray): 要保存的复数数组
        time (np.ndarray): 要保存的时间数组
    """
    np.savez(file_path, channel_matrix=channel_matrix, time=time)
    print(f"Data saved to {file_path}")

def load_channel_data(file_path):
    """
    从 .npz 文件中加载 channel_matrix 和 time
    Args:
        file_path (str): .npz 文件路径
    Returns:
        tuple: (channel_matrix, time)
    """
    data = np.load(file_path)
    channel_matrix = data['channel_matrix']
    time = data['time']
    print(f"Data loaded from {file_path}")
    return channel_matrix, time

def Channel_Cali(channel_matrix):
    """
    将 channel_matrix 中心化，使其中心坐标为 0
    Args:
        channel_matrix (np.ndarray): 复数数组
        time (np.ndarray): 时间数组
    Returns:
        channel_matrix (np.ndarray): 中心化后的复数数组
        time (np.ndarray): 原时间数组
    """
    mean_value = np.mean(channel_matrix)
    channel_matrix -= mean_value  # 中心化复数数组
    print(mean_value)
    return channel_matrix
def Channel_Cali2(channel_matrix):
    """
    校正 channel_matrix，将振幅大于 2 的值替换为前一个点和后一个点的均值。

    Args:
        channel_matrix (np.ndarray): 复数数组

    Returns:
        channel_matrix (np.ndarray): 校正后的复数数组
    """
    # 确保输入是复数数组
    if not np.iscomplexobj(channel_matrix):
        raise ValueError("Input channel_matrix must be a complex array.")
    
    # 遍历数组，校正振幅大于 2 的值
    for i in range(len(channel_matrix)):
        amplitude = np.abs(channel_matrix[i])
        if amplitude > 2:
            if i == 0:  # 如果是第一个点，使用后一个点的值
                channel_matrix[i] = channel_matrix[i + 1]
            elif i == len(channel_matrix) - 1:  # 如果是最后一个点，使用前一个点的值
                channel_matrix[i] = channel_matrix[i - 1]
            else:  # 否则，使用前后点的均值
                channel_matrix[i] = (channel_matrix[i - 1] + channel_matrix[i + 1]) / 2
    
    # 中心化复数数组
    mean_value = np.mean(channel_matrix)
    channel_matrix -= mean_value
    
    return channel_matrix

def plot_amplitude_and_phase(channel_matrix, time, amplitude_file, phase_file):
    """
    绘制信道响应的振幅和相位随时间变化
    Args:
        channel_matrix (np.ndarray): 复数数组
        time (np.ndarray): 时间数组
        amplitude_file (str): 振幅图输出路径
        phase_file (str): 相位图输出路径
    """
    # 绘制振幅随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.abs(channel_matrix), label="Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Channel Response Amplitude Over Time")
    plt.grid()
    plt.legend()
    plt.savefig(amplitude_file)
    plt.close()
    print(f"Amplitude plot saved to {amplitude_file}")
    print(np.unwrap(np.angle(channel_matrix))[-1])
    k = np.unwrap(np.angle(channel_matrix))[-1]/time[-1]
    print(k)
    # 绘制相位随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(time, np.unwrap(np.angle(channel_matrix))-k*time, label="Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.title("Channel Response Phase Over Time")
    plt.grid()
    plt.legend()
    plt.savefig(phase_file)
    plt.close()
    print(f"Phase plot saved to {phase_file}")
def combine_selected_signals(Folder, time_frame):
    rx_sorted_files= process_files_fast(Folder)
    # 初始化复数信号
    combined_signal = np.array([], dtype=np.complex64)
    # 遍历每个指定的帧号，拼接信号
    for frame in time_frame:
        rx_file_path = rx_sorted_files[frame]

        # 读取当前帧的 I 和 Q 数据
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)

        # 将 I 和 Q 数据转换为复数信号
        rx_complex_signal_frame = rx_I_data + 1j * rx_Q_data
        # 拼接当前帧信号到总信号
        combined_signal = np.concatenate((combined_signal, rx_complex_signal_frame))

    #print(f"Combined signal length: {len(combined_signal)} samples")

    # 返回组合后的信号
    return combined_signal


def find_signal_delay(complex_signal, long_signal, plot_file=None):
    """
    Find the starting index in `long_signal` where `complex_signal` matches best based on correlation
    and plot the correlation values against match indices.

    Args:
        complex_signal (np.ndarray): The shorter signal to test.
        long_signal (np.ndarray): The longer signal in which to find the delay.
        plot_file (str, optional): File path to save the plot. If None, the plot will be displayed.

    Returns:
        int: The starting index in `long_signal` where `complex_signal` matches best.
        float: The maximum correlation value.
    """
    # Ensure inputs are numpy arrays
    complex_signal = np.array(complex_signal)
    long_signal = np.array(long_signal)
    
    # Compute cross-correlation
    correlation = np.correlate(long_signal, complex_signal, mode='valid')
    
    # Find the index of maximum correlation
    best_match_index = np.argmax(correlation)
    max_correlation = correlation[best_match_index]
    
    # Plot correlation vs match index
    match_indices = np.arange(len(correlation))
    plt.figure(figsize=(10, 6))
    plt.plot(match_indices, correlation, label="Correlation")
    plt.axvline(best_match_index, color='red', linestyle='--', label=f"Best Match: {best_match_index}")
    plt.xlabel("Match Index")
    plt.ylabel("Correlation")
    plt.title("Correlation vs Match Index")
    plt.legend()
    plt.grid()

    # Save or display the plot
    if plot_file:
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")
    else:
        plt.show()

    # Return best match index and maximum correlation value
    return best_match_index, max_correlation

def optimize_selection(arr, max_zeros=200):
    """
    优化选择数组中尽可能多的 1，同时确保 0 的数量不超过 max_zeros。
    
    Args:
        arr (list): 输入数组，元素为 0 或 1。
        max_zeros (int): 允许的最大 0 的数量。
    
    Returns:
        dict: 包含最佳 z, x, y 的结果，以及选取的 1 和 0 的数量。
    """
    best_z, best_x, best_y = 0, 0, 0
    max_ones = 0
    best_zero_count = 0
    
    n = len(arr)
    
    # 遍历可能的 z, x, y 组合
    for z in range(20):
        for x in range(1, 20):  # 步长至少为 1
            for y in range(1, 30):  # y 不能超过剩余长度
                ones_count = 0
                zeros_count = 0
                
                # 从 z 开始，每间隔 x 个位置，选取连续 y 个元素
                for start in range(z, n, x):
                    segment = arr[start:start + y]
                    ones_count += segment.count(1)
                    zeros_count += segment.count(0)
                    
                    # 提前停止计算，如果 0 的数量超出限制
                    if zeros_count > max_zeros:
                        break
                
                # 更新最佳结果
                if zeros_count <= max_zeros and ones_count > max_ones:
                    best_z, best_x, best_y = z, x, y
                    max_ones = ones_count
                    best_zero_count = zeros_count
    
    return {
        "z": best_z,
        "x": best_x,
        "y": best_y,
        "ones_count": max_ones,
        "zeros_count": best_zero_count,
    }

def timedomainpic(folder, frame):
    # 读取二进制文件路径 (假设您有 `process_files_fast` 和 `read_bin_file` 函数)
    bin_file_path = process_files_fast(folder)[frame]
    I_data, Q_data = read_bin_file(bin_file_path)
    print(I_data)
    print(Q_data)
    I_data = I_data.astype(np.float32)
    Q_data = Q_data.astype(np.float32)
    # 计算能量
    Energy = I_data**2 + Q_data**2

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(Energy, label="Energy", color="blue")
    plt.xlabel("Data Points")
    plt.ylabel("Energy")
    plt.title("Energy vs. Data Points")
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_time_complexamp(Complex):
    magnitude = np.abs(Complex)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(magnitude, label="Magnitude", color="blue")
    plt.xlabel("Data Points")
    plt.ylabel("Amplitude")
    plt.title("Amplitude vs. Data Points")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_time_complexpha(Complex):
    magnitude = np.unwrap(np.angle(Complex))

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(magnitude, label="phase", color="blue")
    plt.xlabel("Data Points")
    plt.ylabel("phase")
    plt.title("phase vs. Data Points")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_bin_file(file_path, data):
    """
    将复数信号数据保存到二进制文件中，存储为 int16 类型。
    """
    # 确保数据为 int16 类型
    interleaved_data = np.empty((data.size * 2,), dtype=np.int16)
    interleaved_data[0::2] = np.round(data.real).astype(np.int16)  # 保存实部
    interleaved_data[1::2] = np.round(data.imag).astype(np.int16)  # 保存虚部

    # 保存到文件
    with open(file_path, "wb") as f:
        interleaved_data.tofile(f)

def create_new_signal_Folder(Folder, firstframe, newfoldername,firstjump = 0):
    # 设置路径
    TxFolder = os.path.join(Folder, "Channel0")  # 不变的
    RxFolder = os.path.join(Folder, "Channel1")  # 人动的
    rx_sorted_files = process_files_fast(RxFolder)
    tx_sorted_files = process_files_fast(TxFolder)
    residual_error = 0
    # 初始化
    TxsaveCounter = 0
    RxsaveCounter = 0
    TotleSignallength = 15000  # 总信号长度动态更新
    combined_signal_rx = np.array([], dtype=np.complex64)
    combined_signal_tx = np.array([], dtype=np.complex64)

    # 创建输出文件夹
    if not os.path.exists(newfoldername):
        os.makedirs(newfoldername)

    Rxfoldernewname = os.path.join(newfoldername, "Channel1")
    Txfoldernewname = os.path.join(newfoldername, "Channel0")

    if not os.path.exists(Rxfoldernewname):
        os.makedirs(Rxfoldernewname)
    if not os.path.exists(Txfoldernewname):
        os.makedirs(Txfoldernewname)

    # 遍历每个指定的帧号，拼接信号
    for rx_file_path, tx_file_path in zip(rx_sorted_files[firstframe:], tx_sorted_files[firstframe:]):
        # 读取当前帧的 I 和 Q 数据
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

        # 将 I 和 Q 数据转换为复数信号
        rx_complex_signal_frame = rx_I_data + 1j * rx_Q_data
        tx_complex_signal_frame = tx_I_data + 1j * tx_Q_data

        # 拼接当前帧信号到总信号
        combined_signal_rx = np.concatenate((combined_signal_rx, rx_complex_signal_frame))
        combined_signal_tx = np.concatenate((combined_signal_tx, tx_complex_signal_frame))

        # 如果总长度达到或超过 15000
        if len(combined_signal_rx) >= TotleSignallength:
        # 动态调整 TotleSignallength
            adjusted_length = round(14999.7367 + residual_error)
            residual_error += 14999.7367 - adjusted_length  # 更新误差累积

            # 更新保存计数器
            RxsaveCounter += 1
            TxsaveCounter += 1

            # 提取前 10240 个数据并保存
            rx_save = combined_signal_rx[:10240]
            tx_save = combined_signal_tx[:10240]

            # 检查保存数据长度是否正确
            if len(rx_save) != 10240:
                print(f"错误: rx_save 长度不正确，RxsaveCounter: {RxsaveCounter}")

            # 更新 combined_signal，移除前 adjusted_length 个复数数据
            combined_signal_rx = combined_signal_rx[adjusted_length:]
            combined_signal_tx = combined_signal_tx[adjusted_length:]

            # 保存到新文件夹
            rx_save_file_name = os.path.join(
                Rxfoldernewname, f"center900frame{RxsaveCounter - 1}.bin"
            )
            tx_save_file_name = os.path.join(
                Txfoldernewname, f"center900frame{TxsaveCounter - 1}.bin"
            )

            save_bin_file(rx_save_file_name, rx_save)
            save_bin_file(tx_save_file_name, tx_save)

    print(f"数据处理完成，新文件已保存到文件夹: {newfoldername}")


def create_new_signal_Folder_Big(Folder, firstframe, newfoldername, firstjump=0):
    # 设置路径
    TxFolder = os.path.join(Folder, "Channel0")  # 不变的
    RxFolder = os.path.join(Folder, "Channel1")  # 人动的
    rx_sorted_files = process_files_fast(RxFolder)
    tx_sorted_files = process_files_fast(TxFolder)
    residual_error = 0

    # 初始化
    TxsaveCounter = 0
    RxsaveCounter = 0
    TotleSignallength = 15000  # 总信号长度动态更新
    combined_signal_rx = np.array([], dtype=np.complex64)
    combined_signal_tx = np.array([], dtype=np.complex64)

    # 创建输出文件夹
    if not os.path.exists(newfoldername):
        os.makedirs(newfoldername)

    Rxfoldernewname = os.path.join(newfoldername, "Channel1")
    Txfoldernewname = os.path.join(newfoldername, "Channel0")

    if not os.path.exists(Rxfoldernewname):
        os.makedirs(Rxfoldernewname)
    if not os.path.exists(Txfoldernewname):
        os.makedirs(Txfoldernewname)

    # 遍历每个指定的帧号，拼接信号
    for idx, (rx_file_path, tx_file_path) in enumerate(zip(rx_sorted_files[firstframe:], tx_sorted_files[firstframe:])):
        # 读取当前帧的 I 和 Q 数据
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

        # 将 I 和 Q 数据转换为复数信号
        rx_complex_signal_frame = rx_I_data + 1j * rx_Q_data
        tx_complex_signal_frame = tx_I_data + 1j * tx_Q_data

        # **对于 firstframe，跳过 firstjump 之前的数据**
        if idx == 0:  
            rx_complex_signal_frame = rx_complex_signal_frame[firstjump:]
            tx_complex_signal_frame = tx_complex_signal_frame[firstjump:]

        # 拼接当前帧信号到总信号
        combined_signal_rx = np.concatenate((combined_signal_rx, rx_complex_signal_frame))
        combined_signal_tx = np.concatenate((combined_signal_tx, tx_complex_signal_frame))

        # **如果长度大于等于 15000，循环去掉前面的数据直到长度小于 15000**
        while len(combined_signal_rx) >= TotleSignallength:
            # 动态调整 TotleSignallength
            adjusted_length = round(14999.7367 + residual_error)
            residual_error += 14999.7367 - adjusted_length  # 更新误差累积

            # 更新保存计数器
            RxsaveCounter += 1
            TxsaveCounter += 1

            # 提取前 10240 个数据并保存
            rx_save = combined_signal_rx[:10240]
            tx_save = combined_signal_tx[:10240]

            # **检查保存数据长度是否正确**
            if len(rx_save) != 10240:
                print(f"错误: rx_save 长度不正确，RxsaveCounter: {RxsaveCounter}")

            # **循环去掉 adjusted_length 直到长度小于 15000**
            combined_signal_rx = combined_signal_rx[adjusted_length:]
            combined_signal_tx = combined_signal_tx[adjusted_length:]

            # 保存到新文件夹
            rx_save_file_name = os.path.join(Rxfoldernewname, f"center900frame{RxsaveCounter - 1}.bin")
            tx_save_file_name = os.path.join(Txfoldernewname, f"center900frame{TxsaveCounter - 1}.bin")

            save_bin_file(rx_save_file_name, rx_save)
            save_bin_file(tx_save_file_name, tx_save)

    print(f"数据处理完成，新文件已保存到文件夹: {newfoldername}")



def Complex_new_signal_Folder(ComplexSignal, newfoldername, firstjump=0):
    # 设置路径
    RxsaveCounter = 0
    TotleSignallength = 15000  # 总信号长度动态更新
    combined_signal_rx = np.array([], dtype=np.complex64)

    # 创建输出文件夹
    if not os.path.exists(newfoldername):
        os.makedirs(newfoldername)

    Rxfoldernewname = os.path.join(newfoldername, "Channel1")

    if not os.path.exists(Rxfoldernewname):
        os.makedirs(Rxfoldernewname)

    # **直接从 ComplexSignal 获取数据**
    rx_complex_signal_frame = ComplexSignal[firstjump:]

    # 拼接信号
    combined_signal_rx = np.concatenate((combined_signal_rx, rx_complex_signal_frame))

    # **如果长度大于等于 15000，循环去掉前面的数据直到长度小于 15000**
    while len(combined_signal_rx) >= TotleSignallength:
        # 动态调整 TotleSignallength
        adjusted_length = round(14999.7367 + residual_error)
        residual_error += 14999.7367 - adjusted_length  # 更新误差累积

        # 更新保存计数器
        RxsaveCounter += 1

        # 提取前 10240 个数据并保存
        rx_save = combined_signal_rx[:10240]

        # **检查保存数据长度是否正确**
        if len(rx_save) != 10240:
            print(f"错误: rx_save 长度不正确，RxsaveCounter: {RxsaveCounter}")

        # **循环去掉 adjusted_length 直到长度小于 15000**
        combined_signal_rx = combined_signal_rx[adjusted_length:]

        # 保存到新文件夹
        rx_save_file_name = os.path.join(Rxfoldernewname, f"center900frame{RxsaveCounter - 1}.bin")
        save_bin_file(rx_save_file_name, rx_save)

    print(f"数据处理完成，新文件已保存到文件夹: {newfoldername}")

def H_T(Folder, frame):
    TxFolder = os.path.join(Folder, "Channel0")  # 发射信号目录
    RxFolder = os.path.join(Folder, "Channel1")  # 接收信号目录

    # 初始化存储复数信号的列表
    tx_complex_signal = []
    rx_complex_signal = []

    # 遍历 frame 数组，读取每个帧的数据
    for i in frame:
        tx_complex_signal.append(read_data(TxFolder, i))  # 每帧的发射信号
        rx_complex_signal.append(read_data(RxFolder, i))  # 每帧的接收信号

    # 将列表转换为 NumPy 数组，方便后续计算
    tx_complex_signal = np.array(tx_complex_signal)
    rx_complex_signal = np.array(rx_complex_signal)

    # 频域计算
    rx_fft_result = fft(rx_complex_signal, axis=-1)  # 对每帧数据计算 FFT
    tx_fft_result = fft(tx_complex_signal, axis=-1)

    # 计算信道频域响应 H(f)，逐帧计算
    Channel_f = rx_fft_result / tx_fft_result  # 防止除以零

    # 将频域信道响应转换为时间域信道响应
    Channel_t = ifft(Channel_f, axis=-1)  # 对每帧数据进行 IFFT
    combined_Channel_t = Channel_t.flatten()  # 展平为一维数组

    return combined_Channel_t



def read_data(folder,frame_number):
    bin_file_path = process_files_fast(folder)[frame_number]
    I_data, Q_data = read_bin_file(bin_file_path)
    complexnumber = I_data+1j*Q_data
    return complexnumber
def OneFreqAvgTimeChannel(complex_signal, Start_diff, resolution, sampling_rate, selected_frequency):
    
    freq_axis = np.fft.fftfreq(resolution, d=1 / sampling_rate)
    if selected_frequency < freq_axis.min() or selected_frequency > freq_axis.max():
        raise ValueError(f"SelectedFrequency {selected_frequency} Hz is out of range [{freq_axis.min()}, {freq_axis.max()}] Hz.")

    # 找到目标频率对应的频率轴索引
    selected_index = np.argmin(np.abs(freq_axis - selected_frequency))
    n = np.arange(resolution)  # 时域采样点索引
    e_term = np.exp(-1j * 2 * np.pi * selected_index * n / resolution)  # DFT 权值计算
    
    framesnumberinthisSegment = (len(complex_signal) - resolution) // Start_diff

    # **修改 dtype 以存储复数**
    current_segment_response = np.zeros(framesnumberinthisSegment, dtype=np.complex128)

    for i in range(framesnumberinthisSegment):
        signal_segment_rx = complex_signal[i * Start_diff : i * Start_diff + resolution]
        rx_fft_result = np.sum(signal_segment_rx * e_term)

        # 计算信道响应并存入当前段结果
        current_segment_response[i] = rx_fft_result

    return current_segment_response  # 直接返回整个数组

def SpecificFreqTimeDomainAvgChannel(Folder, Start_diff, resolution, sampling_rate, selected_frequency, save_filename="H_T_index.npz"):
    """
    Compute the time-averaged channel response at a specific frequency.

    Parameters:
    - Folder: str, path to the folder containing Channel0 and Channel1 data.
    - Start_diff: int, step size for windowing.
    - resolution: int, FFT window size.
    - sampling_rate: float, sampling rate of the signal.
    - selected_frequency: float, frequency of interest.
    - save_filename: str, name of the file to save the computed channel response.

    Returns:
    - H_T_index: list of complex values representing the averaged channel response.
    """

    # 获取 Rx 和 Tx 文件夹中的文件路径
    TxFolder = os.path.join(Folder, "Channel0")
    RxFolder = os.path.join(Folder, "Channel1")
    rx_sorted_files = process_files_fast(RxFolder)
    tx_sorted_files = process_files_fast(TxFolder)

    H_T_index = []

    # 遍历所有帧，计算指定频率的时间平均通道
    for frame in range(len(rx_sorted_files)):
        rx_file_path = rx_sorted_files[frame]
        tx_file_path = tx_sorted_files[frame]
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

        # 复数信号
        rx_complex = rx_I_data + 1j * rx_Q_data
        tx_complex = tx_I_data + 1j * tx_Q_data

        # 计算某一频率的 H(f)
        rx_ChannelSpecfreq = OneFreqAvgTimeChannel(rx_complex, Start_diff, resolution, sampling_rate, selected_frequency)
        tx_ChannelSpecfreq = OneFreqAvgTimeChannel(tx_complex, Start_diff, resolution, sampling_rate, selected_frequency)

        # 计算 H_T_index（元素相除）
        H_T_frame = rx_ChannelSpecfreq / tx_ChannelSpecfreq

        # 计算实部和虚部的中位数
        real_median = np.median(np.real(H_T_frame))
        imag_median = np.median(np.imag(H_T_frame))
        if np.abs(real_median + 1j * imag_median)>1:
            print(frame)
            plot_complex_scatter(H_T_frame)

        # 存入 H_T_index（作为复数）
        H_T_index.append(real_median + 1j * imag_median)

    # 保存数据
    print("in def",H_T_index)
    save_path = os.path.join(Folder, save_filename)
    np.savez_compressed(save_path, H_T_index=np.array(H_T_index, dtype=np.complex128))
    print(f"Saved H_T_index to {save_path}")

    return H_T_index


def saveChannelResponse(folder, filename, channel_response_matrix):
    """
    将 channel_response_matrix（复数数组）存储到指定文件夹下的 .npz 文件中，key 设为 'H_T_index'
    
    Args:
        folder (str): 存储文件的文件夹路径
        filename (str): 保存的文件名，例如 "your_custom_filename.npz"
        channel_response_matrix (np.ndarray): 要存储的复数数组
    """
    # 拼接文件路径
    file_path = os.path.join(folder, filename)
    
    # 如果文件夹不存在，创建文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 保存数据到 .npz 文件，key 设为 'H_T_index'
    np.savez(file_path, H_T_index=channel_response_matrix)
    print(f"数据已保存到 {file_path}，key='H_T_index'")

# 示例使用方法：

def readH_T_index(folder, filename="H_T_index.npz"):
    """
    读取存储的 H_T_index 复数数组
    
    Args:
        folder (str): 存放 H_T_index 的文件夹
        filename (str): 读取的文件名，默认为 "H_T_index.npz"
    
    Returns:
        np.ndarray: 复数数组
    """
    file_path = os.path.join(folder, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    data = np.load(file_path)
    H_T_index = data["H_T_index"]
    
    return H_T_index  # 返回复数数组

def smooth_magnitude(H_T_index, window_size=5):
    """
    Perform a moving average smoothing over the magnitude of H_T_index.
    
    Parameters:
    - H_T_index: ndarray, complex-valued input array
    - window_size: int, size of the smoothing window (default=5)

    Returns:
    - smoothed_magnitude: ndarray, smoothed amplitude with reduced size (original_size - window_size + 1)
    """
    amplitude = np.abs(H_T_index)  # 计算振幅
    smoothed_magnitude = np.convolve(amplitude, np.ones(window_size)/window_size, mode='valid')  # 滑动平均
    return smoothed_magnitude

def plot2d(matrix):
    """
    仅查看矩阵形状的 2D 可视化
    
    Args:
        matrix (np.ndarray): 需要绘制的矩阵
        output_file (str): 输出图像文件名
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(matrix, aspect='auto', cmap='plasma', origin='lower')

    # 添加颜色条
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Amplitude")

    # 设置横纵轴范围
    ax.set_xlabel(f"Time Axis (shape={matrix.shape[1]})")
    ax.set_ylabel(f"Frequency Axis (shape={matrix.shape[0]})")
    ax.set_title("2D Matrix Shape Visualization")

    plt.show()

def analyze_signal_durations_print(complex_signal, end_threshold=80, end_count=10, start_threshold=200):
    """
    计算信号的宽度（每段信号的时间间隔）和信号开始间隔（相邻信号的起始点间隔）。
    
    参数：
    - complex_signal: 复数数组
    - end_threshold: 低于该阈值连续 `end_count` 次即认为信号结束 (默认 100)
    - end_count: 低于 `end_threshold` 的连续点数阈值 (默认 10)
    - start_threshold: 在无信号状态下，超过此阈值的点认为是信号开始 (默认 500)
    
    输出：
    - 打印信号宽度统计和信号开始间隔统计
    """

    # 计算振幅

    # 计算振幅
    amplitude = np.abs(complex_signal)

    # 状态标志位
    in_signal = False  
    zero_count = 0  
    start_time = None
    prev_start_time = None

    # 统计字典 & 信号开始时间数组
    signal_width_dict = defaultdict(int)
    signal_start_interval_dict = defaultdict(int)
    start_times = []  # 存储所有信号的开始时间

    for i, amp in enumerate(amplitude):
        if in_signal:
            # 检测信号结束条件：连续 end_count 次振幅小于 end_threshold
            if amp < end_threshold:
                zero_count += 1
            else:
                zero_count = 0  # 只要出现大于 end_threshold 的值，就重置计数

            if zero_count >= end_count:  # 触发信号结束
                end_time = i - end_count + 1  # 记录信号结束时间
                signal_width = end_time - start_time  # 计算宽度
                signal_width_dict[signal_width] += 1  # 统计宽度
                in_signal = False  # 标记为无信号状态
                prev_start_time = start_time  # 记录上一个信号开始时间
                start_time = None
        else:
            # 无信号状态下，检测信号开始
            if amp > start_threshold:
                start_time = i
                start_times.append(start_time)  # 记录信号开始时间
                in_signal = True
                zero_count = 0  # 重置连续小值计数

                if prev_start_time is not None:
                    interval = start_time - prev_start_time  # 计算间隔
                    signal_start_interval_dict[interval] += 1  # 统计间隔

    # 打印结果
    # print("信号宽度统计（单位：点）:")
    # for width, count in sorted(signal_width_dict.items()):
    #     print(f"宽度: {width}, 次数: {count}")

    # print("\n信号开始间隔统计（单位：点）:")
    # for interval, count in sorted(signal_start_interval_dict.items()):
    #     print(f"间隔: {interval}, 次数: {count}")

    print("\n信号开始时间（单位：点）:", start_times)
    if len(start_times) > 1:
        start_intervals = np.diff(start_times)
        print("\n信号开始时间间隔（单位：点）:", start_intervals)

    return signal_width_dict, signal_start_interval_dict, start_times



def create_Folder_different_signal_length(complex_signal, foldername, start_times, third_shortest_width,Savefilecounter):
    """
    从 complex_signal 中提取信号段，并按原始文件夹结构存储到新的文件夹。

    参数：
    - complex_signal: 长的复数数组
    - foldername: 目标文件夹路径，例如 "./data/xxxx/Channel0/" 或 "./data/xxxx/Channel1/"
    - start_times: 处理后保留的信号开始时间（去掉首尾）
    - third_shortest_width: 统一保存的信号长度

    输出：
    - 数据保存到 "./signaldata/xxxx/Channel0/" 或 "./signaldata/xxxx/Channel1/"
    """

    # 替换 "./data/" 为 "./signaldata/"，保证路径正确
    newfoldername = foldername.replace("./data/", "./signaldata/")

    # 确保新文件夹存在
    os.makedirs(newfoldername, exist_ok=True)

    # 遍历 start_times，提取信号
    first = True

    for start in start_times:
        end = start + third_shortest_width  # 计算结束索引

        # 确保索引不超出范围
        if end > len(complex_signal):
            print(f"Warning: 信号片段 {Savefilecounter} 超出范围，跳过。")
            break

        # 提取信号
        signal_segment = complex_signal[start:end]

        # 生成文件路径（保持 center1890frame{Savefilecounter}.bin 格式）
        # if Savefilecounter==4329:
        #     print("NowSavefilecounter",Savefilecounter)
        #     plot_time_complexamp(signal_segment)
        #     first = False
        save_file_name = os.path.join(newfoldername, f"center1890frame{Savefilecounter}.bin")

        # 保存文件
        save_bin_file(save_file_name, signal_segment)

        # 更新计数器
        Savefilecounter += 1

    #print(f"数据处理完成，新文件已保存到: {newfoldername}")

    return Savefilecounter




def analyze_signal_durations(complex_signal, end_threshold=80, end_count=10, start_threshold=200):
    """
    计算信号的开始时间 和 信号宽度的第5百分位值。
    
    参数：
    - complex_signal: 复数数组
    - end_threshold: 低于该阈值连续 `end_count` 次即认为信号结束 (默认 80)
    - end_count: 低于 `end_threshold` 的连续点数阈值 (默认 10)
    - start_threshold: 在无信号状态下，超过此阈值的点认为是信号开始 (默认 200)
    
    返回：
    - start_times: 记录所有信号开始时间
    - percentile_width: 记录所有信号宽度的第5百分位值
    """

    # 计算振幅
    amplitude = np.abs(complex_signal)

    # --- 这部分逻辑完全保持不变 ---
    in_signal = False
    zero_count = 0
    start_time = None
    signal_widths = []
    start_times = []

    for i, amp in enumerate(amplitude):
        if in_signal:
            if amp < end_threshold:
                zero_count += 1
            else:
                zero_count = 0
            if zero_count >= end_count:
                end_time = i - end_count + 1
                signal_widths.append(end_time - start_time)
                in_signal = False
                start_time = None
        else:
            if amp > start_threshold:
                start_time = i
                start_times.append(start_time)
                in_signal = True
                zero_count = 0

    # --- 这是唯一的修改点 ---
    # 原逻辑:
    # if len(signal_widths) >= 3:
    #     sorted_widths = sorted(signal_widths)
    #     third_shortest_width = sorted_widths[2]
    # else:
    #     third_shortest_width = None
    # return start_times, third_shortest_width

    # 新逻辑：计算第5百分位宽度
    if signal_widths:  # 只要列表不为空即可计算
        # 使用 numpy 计算第5百分位值，并转换为整数
        percentile_width = int(np.percentile(signal_widths, 5))
    else:
        # 如果没有检测到任何完整的信号，返回一个安全的默认值 0
        # 这比返回 None 更安全，可以防止后续计算出错
        percentile_width = 0

    # 返回所有开始时间和计算出的百分位宽度
    return start_times, percentile_width


def lazyrawprocess(rawdatafolder):
    TxFolder = os.path.join(rawdatafolder, "Channel0")
    RxFolder = os.path.join(rawdatafolder, "Channel1")
    tx_sorted_files = process_files_fast(TxFolder)
    total_frames = len(tx_sorted_files)
    time_axis = []
    tempcomplex_tx = np.array([], dtype=np.complex64)
    tempcomplex_rx = np.array([], dtype=np.complex64)

    Savefilecounter = 0
    newfoldername = rawdatafolder.replace("./data/", "./signaldata/")
    os.makedirs(newfoldername, exist_ok=True)
    time_axis_path = os.path.join(newfoldername, "time_axis.npy")

    # 使用一个更具描述性的变量名，例如 p5_width (percentile 5 width)
    p5_width = None

    for i in range(0, total_frames, 200):
        print("i=", i)
        frame_range = list(range(i, min(i + 200, total_frames)))

        rawcomplexsignal_tx = combine_selected_signals(TxFolder, frame_range)
        rawcomplexsignal_rx = combine_selected_signals(RxFolder, frame_range)

        if i == 0:
            # 首次迭代，调用修改后的函数计算第5百分位的宽度
            start_times, p5_width = analyze_signal_durations(rawcomplexsignal_tx) # 返回的是 p5_width
            print(f"Calculated Standard Width (5th Percentile): {p5_width} samples")

            create_Folder_different_signal_length(
                rawcomplexsignal_rx, RxFolder, start_times[2:-1], p5_width, Savefilecounter
            )
            Savefilecounter = create_Folder_different_signal_length(
                rawcomplexsignal_tx, TxFolder, start_times[2:-1], p5_width, Savefilecounter
            )

            nextstart = start_times[-1]
            tempcomplex_tx = rawcomplexsignal_tx[nextstart:]
            tempcomplex_rx = rawcomplexsignal_rx[nextstart:]
            time_axis.extend(start_times[2:-1])

        else:
            rawcomplexsignal_tx = np.concatenate((tempcomplex_tx, rawcomplexsignal_tx))
            rawcomplexsignal_rx = np.concatenate((tempcomplex_rx, rawcomplexsignal_rx))

            # 后续迭代，只需找到 start_times，并继续使用之前计算出的 p5_width
            start_times, _ = analyze_signal_durations(rawcomplexsignal_tx)
            
            create_Folder_different_signal_length(
                rawcomplexsignal_rx, RxFolder, np.array(start_times[0:-1]), p5_width, Savefilecounter
            )
            print("tempcomplex_tx", len(tempcomplex_tx))
            print("Savefilecounter", Savefilecounter)
            
            Savefilecounter = create_Folder_different_signal_length(
                rawcomplexsignal_tx, TxFolder, np.array(start_times[0:-1]), p5_width, Savefilecounter
            )

            # 检查 start_times 是否为空，避免索引错误
            if start_times:
                time_axis.extend(np.array(start_times[0:-1]) + nextstart)
                nextstart += start_times[-1]
                tempcomplex_tx = rawcomplexsignal_tx[start_times[-1]:]
                tempcomplex_rx = rawcomplexsignal_rx[start_times[-1]:]
            else: # 如果没有找到新的信号，则整个信号块都是剩余部分
                tempcomplex_tx = rawcomplexsignal_tx
                tempcomplex_rx = rawcomplexsignal_rx


    np.save(time_axis_path, np.array(time_axis, dtype=np.int64))
    return 0



def Hft_to_Hfr(channel_response_matrix, freq_range=(-7.5e6, 7.5e6), doppler_range=(-1000, 1000), plot=True, Removemedian=True):
    """
    将时变信道 H(f, t) 变换为 H(f, r)，其中 r 是多普勒频率
    仅去除 DC 分量（r=0 处的恒定偏移），确保实部和虚部的中值都正确处理
    """

    H_fr_matrix = np.fft.fft(channel_response_matrix, axis=1)  # 对时间轴 (t) 做 FFT


    if Removemedian:
        dc_index = 0  # FFT 结果的第一个频率 bin 代表 r=0
        median_real = np.median(H_fr_matrix[:, dc_index].real, keepdims=True)  # 计算实部的中值
        median_imag = np.median(H_fr_matrix[:, dc_index].imag, keepdims=True)  # 计算虚部的中值
        median_val = median_real + 1j * median_imag  # 构造复数的中值偏移量
        print(median_val)
        print(H_fr_matrix[:, dc_index])
        # 仅对 DC 分量减去偏移
        H_fr_matrix[:, dc_index] -= median_val
        print(H_fr_matrix[:, dc_index])

    # 3️⃣ 进行 fftshift，将 DC 分量移动到中心
    H_fr_matrix = np.fft.fftshift(H_fr_matrix, axes=1)

    # 4️⃣ 绘图
    if plot:
        H_fr_magnitude = 20 * np.log10(np.abs(H_fr_matrix))  # dB scale
        
        # 生成频率轴（纵轴） & 多普勒轴（横轴）
        freq_axis = np.linspace(freq_range[0], freq_range[1], H_fr_matrix.shape[0])  # 频率 f
        doppler_axis = np.linspace(doppler_range[0], doppler_range[1], H_fr_matrix.shape[1])  # 多普勒 r

        # 画图
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(doppler_axis, freq_axis, H_fr_magnitude, cmap="jet", shading='auto')  # 使用 pcolormesh
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Doppler Frequency (Hz)", fontsize=14)  # 横轴是多普勒
        plt.ylabel("Frequency (Hz)", fontsize=14)  # 纵轴是频率
        plt.title("Doppler Spectrum H(f, r)", fontsize=16)
        
        plt.xticks(np.linspace(doppler_axis[0], doppler_axis[-1], 5))  # 设置 x 轴刻度
        plt.yticks(np.linspace(freq_axis[0], freq_axis[-1], 5))  # 设置 y 轴刻度
        
        plt.show()

    return H_fr_matrix  # 返回 H(f, r)

def SumHfrEnergy(H_fr_matrix, doppler_range=(-1000, 1000), plot=True):
    H_fr_energy = np.sum(np.abs(H_fr_matrix) ** 2, axis=0)  # 计算多普勒方向上的能量\
    H_fr_energy = 20 * np.log10(np.abs(H_fr_energy))
    #print(len(H_fr_energy))
    if plot:
        doppler_freqs = np.linspace(doppler_range[0], doppler_range[1], H_fr_matrix.shape[1])  # 生成多普勒轴
        plt.figure(figsize=(10, 6))
        plt.plot(doppler_freqs, H_fr_energy, linewidth=2)
        plt.xlabel("Doppler Frequency (Hz)", fontsize=14)
        plt.ylabel("Summed Energy", fontsize=14)
        plt.title("Summed Doppler Spectrum Energy", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.show()
    return H_fr_energy  # 返回多普勒频谱的能量

def load_time_axis(newfoldername):
    """
    读取存储的 time_axis.npy 并返回 NumPy 数组。

    参数：
    - rawdatafolder: 原始数据文件夹，例如 "./data/xxxx/"

    返回：
    - time_axis: NumPy 数组
    """
    # 确定 `time_axis.npy` 文件路径
    time_axis_path = os.path.join(newfoldername, "time_axis.npy")

    if not os.path.exists(time_axis_path):
        raise FileNotFoundError(f"时间轴文件未找到: {time_axis_path}")

    return np.load(time_axis_path)

    
def lazydoppler(newfoldername, sampling_rate, selectfreq, Start_diff, STFTsampling_rate, resolution, avgcenterfreq=False, nolinear=False, centerfreqband=(-2e6, 2e6), STFTfreqrange=None):
    if avgcenterfreq:
        combinedsignalstable, _ = CompareChannelAvgFreq(newfoldername, [0, 0], sampling_rate, signalfilter=False, AvgFreqband=centerfreqband)
    else:
        combinedsignalstable = ChannelSpecificFreq(newfoldername, sampling_rate, selectfreq)
    combinedsignalstable = Channel_Cali(combinedsignalstable)

    if nolinear:
        StartSamplepoint = load_time_axis(newfoldername)
        time = StartSamplepoint / 15e6
        
        # 步骤 2: 从 newfoldername 提取基础文件名
        # 例如, 从 "./signaldata/10_Doublehumanmoving_diff/" 提取 "10_Doublehumanmoving_diff"
        base_filename = os.path.basename(newfoldername.strip('/'))
        
        # 将提取出的文件名传递给 nolinearSTFT 函数
        nolinearSTFT(time, combinedsignalstable, resolution, freq_range=STFTfreqrange, output_filename_base=base_filename)
        calculate_global_doppler_spectrum(
            timeaxis=time, 
            complexsignal=combinedsignalstable, 
            freq_range=STFTfreqrange, 
            output_filename_base=base_filename
        )
    else:
        STFT(combinedsignalstable, 0, Start_diff, len(combinedsignalstable), STFTsampling_rate, resolution, freq_range=STFTfreqrange)

        # H_fr_matrix = np.fft.fftshift(np.fft.fft(combinedsignalstable))  # 对时间轴 (t) 做 FFT
        # H_fr_energy = np.abs(H_fr_matrix) ** 2
        # H_fr_energy = 20 * np.log10(np.abs(H_fr_energy))
        # plt.figure(figsize=(10, 6))
        # plt.plot(H_fr_energy)
        # plt.xlabel("Doppler Frequency (Hz)", fontsize=14)
        # plt.ylabel("Summed Energy", fontsize=14)
        # plt.title("Summed Doppler Spectrum Energy", fontsize=16)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(True)
        # plt.show()
    return 0

def lazyChannel(newfoldername,sampling_rate):
    StartSamplepoint = load_time_axis(newfoldername)
    time = StartSamplepoint/15e6
    CompareChannelwithTime(newfoldername, sampling_rate, time, avgminus=False)

def lazydopplerenergy(newfoldername,sampling_rate,selectfreq,Start_diff,STFTsampling_rate,resolution,avgcenterfreq=False,nolinear=False,centerfreqband=(-2e6,2e6),STFTfreqrange=None):
    Htf = OnlyCompareChannel(Channelfolder_name,[0,0],15e6,avgminus=True)
    print(np.shape(Htf))
    print(len(Htf[:,0]))
    Htr_total = []
    for i in range(len(Htf[:,0])):
        start_idx =  i * Start_diff
        end_idx = start_idx + resolution
        # Stop if the window exceeds Final_point
        if end_idx > len(Htf[0, :]):
            break
        # Extract the current window from all rows (first dimension) and the specified columns (second dimension)
        window_signal = Htf[:, start_idx:end_idx].copy()
        Htr = Hft_to_Hfr(window_signal,plot=False,Removemedian=False)
        Htr_total.append(SumHfrEnergy(Htr,plot=False))

    print(np.shape(Htr_total))
    Htr_total = np.array(Htr_total).T  # **确保形状为 频率 × 时间**
    Htr_total = Htr_total[::-1, :]  # 颠倒频率轴，使低频在下，高频在上

    # 生成时间和频率轴
    time_axis = np.linspace(0, 10, Htr_total.shape[1])  # 这里假设 0-10s 的时间
    freq_axis = np.linspace(-15e6, 15e6, Htr_total.shape[0])  # 假设 -15MHz 到 15MHz 的频率范围

    # 绘制 STFT 结果
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(time_axis, freq_axis, Htr_total, shading='auto', cmap='jet', vmin=-100, vmax=-20)

    # 设置 colorbar 并调整字体大小
    cbar = plt.colorbar(label='Channel Strength (dB)')
    cbar.ax.tick_params(labelsize=24)  # 调整 colorbar 数值的字体大小

    # 设置 x 轴、y 轴标签和标题
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('Doppler Frequency (Hz)', fontsize=20)
    plt.title('Non-Uniform Short-Time Fourier Transform (NU-STFT)', fontsize=18)

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.grid()
    plt.show()

    # combinedsignalstable,_ = CompareChannelAvgFreq(newfoldername, [0,0], sampling_rate, signalfilter=False,AvgFreqband = centerfreqband)

    # combinedsignalstable = Channel_Cali(combinedsignalstable)
    # plot_time_complexamp(combinedsignalstable)
    # if nolinear:
    #     StartSamplepoint = load_time_axis(newfoldername)
    #     time = StartSamplepoint/15e6
    #     nolinearSTFT(time,combinedsignalstable,resolution,freq_range=STFTfreqrange)
    #     #STFT(combinedsignalstable, 0, Start_diff, len(combinedsignalstable), STFTsampling_rate, resolution,freq_range = STFTfreqrange)

def get_H_f_t_matrix(Folder, sampling_rate):
    """
    从原始IQ文件计算并返回完整的 H(f,t) 信道响应矩阵。

    Args:
        Folder (str): 包含 Channel0 和 Channel1 的主文件夹路径。
        sampling_rate (float): ADC 采样率 (Hz)。

    Returns:
        H_f_t (np.ndarray): 复数二维矩阵，形状为 (频率点数, 时间帧数)。
        frame_rate (float): 帧率 (Hz)，用于多普勒轴计算。
        fft_length (int): FFT点数，用于距离轴计算。
    """
    print("Step 1: Extracting H(f,t) matrix from raw data...")
    TxFolder = os.path.join(Folder, "Channel0")
    RxFolder = os.path.join(Folder, "Channel1")

    # 假设 process_files 和 read_bin_file 函数已定义
    try:
        rx_sorted_files, _, _, _ = process_files(RxFolder)
        tx_sorted_files, _, _, _ = process_files(TxFolder)
    except NameError:
        print("错误：请确保 'process_files' 函数已在您的代码中定义。")
        return None, None, None

    if not rx_sorted_files:
        print(f"错误：在 {RxFolder} 中没有找到文件。")
        return None, None, None
    if not tx_sorted_files:
        print(f"错误：在 {TxFolder} 中没有找到文件。")
        return None, None, None

    # 从第一个文件中获取FFT长度
    try:
        with open(rx_sorted_files[0], "rb") as f:
            total_data = np.fromfile(f, dtype=np.int16)
        fft_length = len(total_data) // 2
    except (IOError, NameError) as e:
        print(f"错误：读取文件或 'read_bin_file' 函数有问题。 {e}")
        return None, None, None
        
    print(f"Detected FFT Length (number of frequency bins): {fft_length}")
    
    H_f_t_columns = []
    
    for frame_idx, (rx_file, tx_file) in enumerate(zip(rx_sorted_files, tx_sorted_files)):
        rx_I, rx_Q = read_bin_file(rx_file)
        tx_I, tx_Q = read_bin_file(tx_file)
        
        rx_complex = rx_I + 1j * rx_Q
        tx_complex = tx_I + 1j * tx_Q
        
        # 对当前时间帧的信号做FFT
        rx_fft = fftshift(fft(rx_complex))
        tx_fft = fftshift(fft(tx_complex))
        
        # 为避免除以零，给分母加一个极小值
        epsilon = 1e-12
        
        # 计算该时间帧的信道频率响应 H(f)
        channel_response_one_frame = rx_fft / (tx_fft + epsilon)
        
        H_f_t_columns.append(channel_response_one_frame)

    if not H_f_t_columns:
        print("错误：未能处理任何帧，无法生成 H(f,t) 矩阵。")
        return None, None, None

    # 将所有时间帧的 H(f) 向量堆叠成 H(f,t) 矩阵
    # 使用 np.stack，列方向是时间(t)，行方向是频率(f)
    H_f_t = np.stack(H_f_t_columns, axis=1)
    
    # 计算帧率 (每秒钟处理多少个时间帧)
    # 帧率 = ADC采样率 / 每个I或Q样本序列的长度
    frame_rate = sampling_rate / fft_length
    print(f"Calculated frame rate: {frame_rate:.2f} Hz")
    
    print(f"Successfully created H(f,t) matrix with shape: {H_f_t.shape}")
    print("      (Rows: Frequency Bins, Columns: Time Frames)")
    
    return H_f_t, frame_rate, fft_length


# ==============================================================================
# 步骤 2: 核心函数，用于生成和绘制距离-多普勒图
# ==============================================================================
def generate_and_plot_ddm(H_f_t, adc_sampling_rate, frame_rate, fft_length):
    """
    接收 H(f,t) 矩阵，生成并绘制距离-多普勒图 (修正版)。

    Args:
        H_f_t (np.ndarray): H(f,t) 矩阵，形状 (频率点数, 时间帧数)。
        adc_sampling_rate (float): ADC 采样率 (Hz), e.g., 15e6。
        frame_rate (float): 帧率 (Hz)。
        fft_length (int): FFT点数。
    """
    if H_f_t is None:
        print("输入 H_f_t 矩阵为空，无法继续。")
        return

    print("\nStep 2: Generating Distance-Doppler Map (Corrected Version)...")
    SPEED_OF_LIGHT = 3 * 10**8  # 光速 (m/s)

    # --- 核心计算 (与之前相同) ---
    H_f_fd = fftshift(fft(H_f_t, axis=1), axes=1)
    h_tau_fd = fftshift(ifft(H_f_fd, axis=0), axes=0)
    ddm_energy_db = 10 * np.log10(np.abs(h_tau_fd)**2)

    # --- 准备绘图坐标轴 (修正部分) ---
    num_freq_bins, num_time_frames = H_f_t.shape
    
    # X轴: 多普勒频移 (Hz) - 计算方式正确，无需修改
    doppler_axis = fftshift(fftfreq(num_time_frames, d=1/frame_rate))
    
    # Y轴: 距离 (m) - *** 这里是关键的修正 ***
    # 1. 计算频率分辨率 (两个相邻频率点之间的间隔)
    #    总带宽为 adc_sampling_rate，有 num_freq_bins (即 fft_length) 个频点
    delta_f = adc_sampling_rate / num_freq_bins
    
    # 2. 使用频率分辨率作为 d 参数来计算时延轴
    #    fftfreq(点数, 采样间隔) -> 这里的“采样间隔”是频域的间隔 delta_f
    delay_axis_s = fftshift(fftfreq(num_freq_bins, d=delta_f))
    
    # 3. 将时延转换为距离
    distance_axis_m = delay_axis_s * SPEED_OF_LIGHT / 2 # 除以2，考虑信号往返
    
    print(f"Distance axis calculated. Range: {distance_axis_m.min():.2f} m to {distance_axis_m.max():.2f} m")


    # --- 绘图 (修正部分) ---
    plt.figure(figsize=(12, 8))
    
    # 为解决“颜色单一”问题，我们需要处理动态范围
    # 方法1: 找到能量的99.9%分位点作为最大值，忽略极强的直达波
    # 我们先排除无穷大的值
    finite_energy_db = ddm_energy_db[np.isfinite(ddm_energy_db)]
    if len(finite_energy_db) == 0:
        print("错误：能量矩阵中没有有效数值，无法绘图。")
        return

    v_max = np.percentile(finite_energy_db, 99.9) 
    # v_min可以设置为中位数或者比v_max低一个固定值，比如40dB
    v_min = v_max - 40 

    # 使用 imshow 绘制热力图, 加入 vmax 和 vmin
    plt.imshow(ddm_energy_db, aspect='auto', origin='lower',
               extent=[doppler_axis[0], doppler_axis[-1], distance_axis_m[0], distance_axis_m[-1]],
               vmax=v_max, vmin=v_min)
    
    plt.colorbar(label='Energy (dB)')
    plt.xlabel('Doppler (Hz)', fontsize=14)
    plt.ylabel('Distance (m)', fontsize=14)
    plt.title('Distance-Doppler Map (Corrected)', fontsize=16)
    
    max_dist_display = 5000 
    plt.ylim(-max_dist_display, max_dist_display)
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()
    print("Distance-Doppler Map plotted.")



def generate_ddm_with_zeropadding(H_f_t, adc_sampling_rate, frame_rate, fft_length, padding_factor=10):
    """
    通过在频率轴上补零来生成高精细度的延迟-多普勒图。

    Args:
        H_f_t (np.ndarray): H(f,t) 矩阵。
        adc_sampling_rate (float): ADC 采样率 (Hz)。
        frame_rate (float): 帧率 (Hz)。
        fft_length (int): 原始FFT点数。
        padding_factor (int): 补零的倍数，例如10。
    """
    if H_f_t is None:
        print("输入 H_f_t 矩阵为空，无法继续。")
        return

    print(f"\nGenerating DDM with {padding_factor}x Zero-Padding on Frequency Axis...")
    
    # 步骤 1: 沿时间轴做FFT，得到多普勒信息 (和之前一样)
    H_f_fd = fftshift(fft(H_f_t, axis=1), axes=1)

    # ==================== 核心步骤：频率轴补零 ====================
    num_freq_bins, num_time_frames = H_f_fd.shape
    
    # 计算补零后的新FFT点数
    new_num_freq_bins = num_freq_bins * padding_factor
    
    # 创建一个用于存放补零后频谱的、填满0的新矩阵
    H_padded = np.zeros((new_num_freq_bins, num_time_frames), dtype=np.complex128)
    
    # 因为频谱是经过 fftshift 的，低频在两边，高频在中间。
    # 我们需要将0插在频谱的中心（即高频部分）。
    center_index = num_freq_bins // 2
    
    # 复制频谱的前半部分（负频率到0）
    H_padded[0:center_index, :] = H_f_fd[0:center_index, :]
    
    # 复制频谱的后半部分（0到正频率），放到新矩阵的末尾
    # 这样就在中间留下了一大块0
    H_padded[new_num_freq_bins - (num_freq_bins - center_index):, :] = H_f_fd[center_index:, :]
    
    # =================================================================

    # 步骤 2: 对补零后的频谱，沿频率轴做IFFT，得到高精细度的时延信息
    # 注意：在做IFFT之前，需要先 un-shift，IFFT之后再 shift回来
    # 这是因为补零操作破坏了fftshift的对称性
    h_tau_fd_padded = fftshift(ifft(fftshift(H_padded, axes=0), axis=0), axes=0)

    # 步骤 3: 计算能量并转换为dB
    ddm_energy_db = 10 * np.log10(np.abs(h_tau_fd_padded)**2)

    # --- 准备绘图坐标轴 (使用新的点数) ---
    # 多普勒轴不变
    doppler_axis = fftshift(fftfreq(num_time_frames, d=1/frame_rate))
    
    # 距离轴需要用新的点数重新计算，以反映插值后的精细度
    delta_f = adc_sampling_rate / num_freq_bins # 频率分辨率保持不变
    delay_axis_s = fftshift(fftfreq(new_num_freq_bins, d=delta_f))
    distance_axis_m = delay_axis_s * 3e8 / 2

    # --- 绘图 ---
    plt.figure(figsize=(12, 8))
    finite_energy_db = ddm_energy_db[np.isfinite(ddm_energy_db)]
    if len(finite_energy_db) == 0:
        print("错误：能量矩阵中没有有效数值，无法绘图。")
        return
    v_max = np.percentile(finite_energy_db, 99.9) 
    v_min = v_max - 50

    # 使用 interpolation='none' 来忠实地显示所有计算出的点
    plt.imshow(ddm_energy_db, aspect='auto', origin='lower',
               extent=[doppler_axis[0], doppler_axis[-1], distance_axis_m[0], distance_axis_m[-1]],
               vmax=v_max, vmin=v_min, interpolation='none')
    
    plt.colorbar(label='Energy (dB)')
    plt.xlabel('Doppler (Hz)', fontsize=14)
    plt.ylabel('Distance (m) [Padded]', fontsize=14)
    plt.title(f'DDM with {padding_factor}x Freq-Axis Zero-Padding', fontsize=16)
    
    # 限制显示范围以便观察
    plt.ylim(-100, 100) 
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()

def capon_cir_estimator(h_f_vector, delta_f, subarray_L, num_scan_points=1000, max_delay_ns=200):
    """
    使用Capon算法和空间平滑法，从单快照信道频率响应估计高分辨率的信道冲激响应。

    Args:
        h_f_vector (np.ndarray): 单个快照的信道频率响应向量 H(f)。
        delta_f (float): 子载波频率间隔 (Hz)。
        subarray_L (int): 空间平滑的子阵列长度。这是关键的调节参数。
        num_scan_points (int): 在延迟轴上扫描的点的数量。
        max_delay_ns (float): 扫描的最大延迟（纳秒）。

    Returns:
        tau_grid (np.ndarray): 扫描的延迟轴 (秒)。
        capon_spectrum (np.ndarray): 对应的Capon谱能量。
    """
    M = len(h_f_vector)
    if subarray_L >= M:
        raise ValueError("Subarray length L must be smaller than the vector length M.")

    K = M - subarray_L + 1  # 虚拟快照的数量

    # 1. 空间平滑法：构建协方差矩阵 R
    R = np.zeros((subarray_L, subarray_L), dtype=np.complex128)
    for i in range(K):
        snapshot = h_f_vector[i:i + subarray_L].reshape(subarray_L, 1)
        R += snapshot @ snapshot.conj().T
    R /= K

    # 2. 对R求逆。为增加数值稳定性，可以加入微小的对角加载。
    try:
        R_inv = inv(R + np.eye(subarray_L) * 1e-7)
    except LinAlgError:
        print("Warning: Covariance matrix is singular. Capon failed for this slice.")
        return None, None

    # 3. 在延迟范围内扫描，计算Capon谱
    tau_grid = np.linspace(-max_delay_ns * 1e-9, max_delay_ns * 1e-9, num_scan_points)
    capon_spectrum = np.zeros(num_scan_points)
    
    # 预先计算频率向量，避免在循环中重复计算
    freq_indices = np.arange(subarray_L).reshape(subarray_L, 1)
    
    for i, tau in enumerate(tau_grid):
        # a. 构建导向矢量 a(τ)
        steering_vector = np.exp(-1j * 2 * np.pi * tau * delta_f * freq_indices)
        
        # b. 计算Capon能量
        denominator = steering_vector.conj().T @ R_inv @ steering_vector
        capon_spectrum[i] = 1.0 / np.abs(denominator)
        
    return tau_grid, capon_spectrum


def generate_ddm_with_capon(H_f_t, adc_sampling_rate, frame_rate, fft_length, subarray_L=100):
    """
    使用Capon算法生成高分辨率的DDM。
    警告：计算会非常缓慢！
    """
    print(f"\nGenerating DDM with Capon Estimator (Subarray L={subarray_L})...")
    print("This will be VERY SLOW. Please be patient.")
    
    # 步骤 1: 沿时间轴做FFT
    H_f_fd = fftshift(fft(H_f_t, axis=1), axes=1)
    num_freq_bins, num_time_frames = H_f_fd.shape

    # 准备一个足够精细的延迟/距离轴用于最终绘图
    # 这里可以根据capon_cir_estimator的输出来动态确定
    num_scan_points = 1000
    max_delay_ns = 200 # 扫描 +/- 200ns
    
    # 初始化最终的DDM矩阵
    ddm_capon = np.zeros((num_scan_points, num_time_frames))
    
    # 计算子载波间隔
    delta_f = adc_sampling_rate / fft_length

    # 步骤 2: 逐个多普勒通道应用Capon算法
    for i in range(num_time_frames):
        if (i + 1) % 50 == 0:
            print(f"Processing Doppler bin {i + 1}/{num_time_frames}...")
            
        h_f_vector = H_f_fd[:, i]
        
        # 调用Capon估计器
        tau_axis, cir_spectrum = capon_cir_estimator(
            h_f_vector, delta_f, subarray_L=subarray_L,
            num_scan_points=num_scan_points, max_delay_ns=max_delay_ns
        )
        
        if cir_spectrum is not None:
            ddm_capon[:, i] = cir_spectrum

    # --- 绘图 ---
    doppler_axis = fftshift(fftfreq(num_time_frames, d=1/frame_rate))
    distance_axis_m = tau_axis * 3e8 / 2 # 将延迟转换为距离

    ddm_energy_db = 10 * np.log10(ddm_capon)
    
    plt.figure(figsize=(12, 8))
    # ... (此处省略与之前类似的绘图代码，包括vmax/vmin的计算) ...
    v_max = np.percentile(ddm_energy_db[np.isfinite(ddm_energy_db)], 99.9)
    v_min = v_max - 50
    plt.imshow(ddm_energy_db, aspect='auto', origin='lower',
               extent=[doppler_axis[0], doppler_axis[-1], distance_axis_m[0], distance_axis_m[-1]],
               vmax=v_max, vmin=v_min, interpolation='bilinear')
    
    plt.colorbar(label='Capon Power (dB)')
    plt.xlabel('Doppler (Hz)')
    plt.ylabel('Distance (m) [Capon Estimated]')
    plt.title(f'DDM with Capon Estimator (L={subarray_L})')
    plt.ylim(-30, 30) # 限制在合理的室内距离
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()
def calculate_caf_for_window(ref_data, sur_data, time_axis_sec, doppler_axis, delay_axis_sec, fs):
    """为单个窗口计算完整的二维CAF矩阵（此函数无需改变）"""
    caf_matrix = np.zeros((len(doppler_axis), len(delay_axis_sec)), dtype=np.complex128)
    for j, delay_sec in enumerate(delay_axis_sec):  # 外层循环，迭代次数少 (2次)
        # 它们只依赖外层循环的 delay_sec，所以可以放在这里
        delay_samples = int(delay_sec * fs)
        ref_data_delayed = np.roll(ref_data, delay_samples)
        product = sur_data * np.conj(ref_data_delayed) 
        for i, fd in enumerate(doppler_axis):  # 内层循环，迭代次数多 (1000次)
            # (A) 现在放到了内层
            doppler_phasor = np.exp(-1j * 2 * np.pi * fd * time_axis_sec)
            # (D) 依赖内层的 A 和 外层的 C
            caf_point = np.sum(product * doppler_phasor)
            
            # 存储位置要注意，i对应行，j对应列
            caf_matrix[i, j] = caf_point
    return caf_matrix

def visualize_product_3d(product_vector, title_info=""):
    """
    将一个长的复数向量在3D空间中可视化 (Index, I, Q)。
    会自动进行降采样以处理大量数据。
    """
    n_points = len(product_vector)
    print(f"开始可视化 '{title_info}'... 向量长度: {n_points}")

    # --- 降采样 ---
    # 如果点数太多，则进行降采样，目标是只绘制几千个点
    max_points_to_plot = 5000  # 可以根据电脑性能调整
    if n_points > max_points_to_plot:
        step = n_points // max_points_to_plot
        print(f"数据量过大 ({n_points} 点)，进行降采样，每 {step} 个点取一个。")
        vis_data = product_vector[::step]
        # 创建对应的原始x轴坐标
        x_axis = np.arange(0, n_points, step)
    else:
        print("数据量适中，绘制所有点。")
        vis_data = product_vector
        x_axis = np.arange(n_points)
    
    # 提取I和Q分量
    i_data = vis_data.real
    q_data = vis_data.imag

    # --- 绘图 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 使用plot绘制3D轨迹线
    ax.plot(x_axis, i_data, q_data, lw=0.8) # lw是线宽

    # 如果点不多，也可以同时绘制散点
    if len(x_axis) < 2000:
        ax.scatter(x_axis, i_data, q_data, c=q_data, cmap='viridis', s=5, alpha=0.6)

    ax.set_title(f'Product 向量的三维(Index-I-Q)可视化\n{title_info}')
    ax.set_xlabel('样本序号 (Sample Index)')
    ax.set_ylabel('I (实部)')
    ax.set_zlabel('Q (虚部)')
    ax.grid(True)
    
    print("绘图完成。您可以交互式地旋转图形。")
    plt.show()

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# 假设这些函数已经定义在您的代码的其他地方
# def process_files(folder): ...
# def read_bin_file(path): ...
# def calculate_caf_for_window(ref, sur, time_axis, ...): ...
# def load_time_axis(folder): ...

def lazy_caf_doppler_from_bin(folder_name,
                              sampling_rate,
                              window_size=400,
                              slide_step=100,
                              doppler_range=(-500, 500),
                              doppler_resolution=2.0,
                              time_stretch_factor=1.7):
    """
    全新版本：直接从原始.bin文件读取数据并执行CAF分析。
    (V5 - 根据最终要求修正时间戳加载和校准逻辑)
    """
    print("--- 开始执行基于交叉相关(CAF)的多普勒分析 (V5 - 最终时间校准) ---")

    # --- 1. 加载文件列表和绝对时间戳 ---
    tx_folder = os.path.join(folder_name, "Channel0")
    rx_folder = os.path.join(folder_name, "Channel1")

    try:
        tx_sorted_files, _, _, _ = process_files(tx_folder)
        rx_sorted_files, _, _, _ = process_files(rx_folder)
    except (NameError, TypeError):
        print("错误: process_files 函数未定义或返回None。")
        return

    total_frames = min(len(tx_sorted_files), len(rx_sorted_files))
    print(f"文件列表加载完成，共找到 {total_frames} 个数据帧。")

    try:
        # <<< FIX 1: 修正时间戳的加载和校准方式 >>>
        # 1. 直接加载，因为已经是秒
        start_times_raw = load_time_axis(folder_name)[:total_frames]
        # 2. 乘以校准/拉伸因子
        frame_start_times = start_times_raw * time_stretch_factor/15e6
        print(f"已加载并校准 {len(frame_start_times)} 个帧起始时间点 (乘以 {time_stretch_factor})。")
    except (NameError, TypeError):
        print("错误: load_time_axis 函数未定义或返回None。无法加载精确时间戳。")
        return

    # 确定单个文件的FFT长度
    I, Q = read_bin_file(tx_sorted_files[0])
    FFTLength = len(I)

    # --- 2. CAF参数定义 ---
    doppler_axis = np.arange(doppler_range[0], doppler_range[1] + doppler_resolution, doppler_resolution)
    delay_axis_sec = np.array([0.0])

    # --- 3. 滑动窗口处理 ---
    spectrogram_data = []
    spectrogram_time_axis = []

    for i in tqdm(range(0, total_frames - window_size + 1, slide_step), desc="Processing Windows with CAF"):

        # a. 初始化列表 (V4的逻辑，正确且不变)
        window_ref_data_list = []
        window_sur_data_list = []
        window_time_axis_list = []

        # b. 遍历当前窗口内的每一帧 (V4的逻辑，正确且不变)
        for frame_idx_in_window in range(window_size):
            absolute_frame_idx = i + frame_idx_in_window

            tx_path = tx_sorted_files[absolute_frame_idx]
            rx_path = rx_sorted_files[absolute_frame_idx]
            tx_i, tx_q = read_bin_file(tx_path)
            rx_i, rx_q = read_bin_file(rx_path)

            current_ref_frame_data = tx_i + 1j * tx_q
            current_sur_frame_data = rx_i + 1j * rx_q
            window_ref_data_list.append(current_ref_frame_data)
            window_sur_data_list.append(current_sur_frame_data)

            # 为这一帧创建它自己的、精确的、绝对时间轴
            # 这里的 frame_start_times 已经是校准过的值了
            frame_start_time = frame_start_times[absolute_frame_idx]
            num_samples_in_frame = len(current_ref_frame_data)
            # 注意：内部相对时间不应该被拉伸，只有起点被拉伸了
            # 所以这里的 sampling_rate 仍然是原始的采样率
            relative_time_in_frame = np.arange(num_samples_in_frame) / sampling_rate
            single_frame_time_axis = frame_start_time + relative_time_in_frame

            window_time_axis_list.append(single_frame_time_axis)

        # c. 拼接 (V4的逻辑，正确且不变)
        ref_data = np.concatenate(window_ref_data_list)
        sur_data = np.concatenate(window_sur_data_list)
        time_axis = np.concatenate(window_time_axis_list)
        print(time_axis)
        assert len(ref_data) == len(time_axis), "数据和时间轴的最终长度不匹配!"

        # d. 计算CAF
        caf_matrix = calculate_caf_for_window(ref_data, sur_data, time_axis, doppler_axis, delay_axis_sec, sampling_rate)

        # e. 提取结果
        doppler_spectrum = caf_matrix[:, 0]
        spectrogram_data.append(np.abs(doppler_spectrum)**2)

        # f. 记录窗口的真实中心时间
        center_frame_index = i + window_size // 2
        # 这里的 frame_start_times 已经是校准过的值了
        window_center_time = frame_start_times[center_frame_index]
        spectrogram_time_axis.append(window_center_time)

    # --- 4. 绘图 ---
    if not spectrogram_data:
        print("处理完成，但没有足够的数据生成谱图。")
        return

    print("处理完成，正在绘制...")
    spectrogram_matrix = np.array(spectrogram_data).T
    spectrogram_matrix[spectrogram_matrix <= 0] = 1e-20
    spectrogram_db = 10 * np.log10(spectrogram_matrix / np.max(spectrogram_matrix))

    # <<< FIX 2: 绘图的X轴直接使用记录好的值，不再需要额外乘拉伸因子 >>>
    # 因为 spectrogram_time_axis 里的值在记录时就已经从校准后的 frame_start_times 获取了
    time_axis_for_plotting = np.array(spectrogram_time_axis)
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(time_axis_for_plotting, doppler_axis, spectrogram_db, shading='gouraud', cmap='jet', vmin=-30, vmax=-80)
    plt.colorbar(label='归一化功率 (dB)')
    plt.title(f'Doppler-Time Spectrogram (V5 - Calibrated Time Base, Stretch={time_stretch_factor}x)')
    plt.xlabel('Calibrated Time (s)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.ylim(top=doppler_range[1], bottom=doppler_range[0])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 参数设
    SAMPLING_RATE = 15e6
    #folder_name = "./data/Breath2/"
    # folder_name = "./data/1895fastwalk/"
    #folder_name = "./data/NoHuman/"
    #folder_name = "./data/Handmove2/"
    #folder_name = "./data/movefinger/"
    #Avgamp = Average_getdata_ampplot(folder_name,15e6)
    #Avgamp = Channel_Cali(Avgamp)
    #STFT(Avgamp,0,1,len(Avgamp),15e6//16384,300,freq_range=[-80,80])
    #CalculateDoppler(Avgamp,0,1,len(Avgamp),15e6//16384,300)
    # Complexdata = combine_selected_signals(folder_name,list(range(1950,2200)))
    # plot_time_complexamp(Complexdata)

    # #folder_name2 = "./data/moveHand/Channel1/"
    #lazyrawprocess(folder_name)
    # TxFolder = os.path.join(folder_name, "Channel1")  # 不变的
    # Complexdata = combine_selected_signals(TxFolder,list(range(22, 31)))[5861:140537]
    # analyze_signal_durations_print(Complexdata)
    # STFT(Complexdata,0,20,len(Complexdata),15e6,500,interpolation_factor=1)
    # Complexdata2 = combine_selected_signals(TxFolder,list(range(30, 39)))[1945:136622]
    # analyze_signal_durations_print(Complexdata2)
    # STFT(Complexdata2,0,20,len(Complexdata2),15e6,500,interpolation_factor=1)
    # #print(len(Complexdata2))
    folder_name = "./data/18_Slider_diff/"
    Channelfolder_name =  folder_name.replace("./data/", "./signaldata/")
    #lazyrawprocess(folder_name)
    #lazyChannel(Channelfolder_name,15e6)
    #RxFolder = os.path.join(Channelfolder_name, "Channel1")  # 人动的
    #Average_getdata_ampplot(TxFolder,15e6)
    #Average_getdata_ampplot(RxFolder,15e6)
    #lazyChannel(Channelfolder_name,15e6)
    #CompareChannel(Channelfolder_name, [0,0], 15e6)
   # StartSamplepoint = load_time_axis(Channelfolder_name)
    #time = StartSamplepoint/15e6
    #CompareChannel(Channelfolder_name,[0],15e6)
    #print(time,len(time))
    lazydoppler(Channelfolder_name,15e6,2e6,100,2000,400,avgcenterfreq=True,nolinear=True,STFTfreqrange=[-50,50])
    # lazy_caf_doppler_from_bin(
    #     folder_name=Channelfolder_name,
    #     sampling_rate=15e6,
    #     window_size=400,
    #     slide_step=100,
    #     doppler_range=(-400, 400),
    #     doppler_resolution=5,
    #     time_stretch_factor=1.7
    # )
    
    # H_matrix, F_rate, N_fft = get_H_f_t_matrix(Channelfolder_name, SAMPLING_RATE)

    #2. 如果成功获取矩阵，则生成并绘制距离-多普勒图
    # if H_matrix is not None:
    #     generate_ddm_with_zeropadding(H_matrix, SAMPLING_RATE, F_rate, N_fft, padding_factor=1)


    #Htf = OnlyCompareChannel(Channelfolder_name,[0,0],15e6,avgminus=True)
    #Htr = Hft_to_Hfr(Htf,plot=False)
    #SumHfrEnergy(Htr,plot=True)

    # # 参数设置
    # fs = 1000          # 采样率：每秒采样1000个点
    # f = 300            # 信号频率：300 Hz
    # duration = 1.0     # 信号持续时间：1秒

    # # 生成时间轴：从0到1秒（不包含1秒），间隔为 1/fs
    # t = np.arange(0, duration, 1/fs)

    # # 生成复数正弦信号：利用欧拉公式 exp(j*2*pi*f*t)
    # signal = np.exp(1j * 2 * np.pi * f * t)
    
    # # 打印部分数据以验证
    # print("时间轴前5个样本：", t[:5])
    # print("复数信号前5个样本：", signal[:5])
    # nolinearSTFT(t,signal,resolution=400)
    # STFT(signal, 0, 1, len(signal), 1000, resolution=400)

    # # 可视化复数信号的实部和虚部
    # plt.figure(figsize=(10, 4))
    # plt.plot(t, signal.real, label='实部')
    # plt.plot(t, signal.imag, label='虚部', linestyle='--')
    # plt.title('300Hz 复数正弦信号 (采样率 1000 Hz)')
    # plt.xlabel('时间 (秒)')
    # plt.ylabel('幅值')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

# 生成 y 值，全是 1
    # y_values = np.ones_like(time)

    # # 绘制图表
    # plt.figure(figsize=(10, 4))
    # plt.scatter(time, y_values, marker='|', color='blue', s=100)  # 竖线标记
    # plt.xlabel("Time (s)")
    # plt.ylabel("Signal Presence")
    # plt.title("Signal Start Times")

    # # 显示网格
    # plt.grid(True)

    # # 显示图像
    # plt.show()


#Signal detect forder preprocess code
#--------------------------------------------------------
#See Energy

#------------------Short frame Signal------------------
#Complexdata = combine_selected_signals("./data/fastwalkSignal/Channel0",[0,1])
#Start_point = 0
#Start_diff = 20
#sampling_rate = 15e6
#resolution = 500
#Complexdata = Complexdata[:100000]
#STFT(Complexdata, Start_point, Start_diff, len(Complexdata), sampling_rate, resolution)
#create_new_signal_Folder(Folder, firstframe, newfoldername, firstjump=0)


#------------------Big frame Signal------------------
#Complexdata = combine_selected_signals("./data/fastwalkSignal/Channel0",[0])
#arr = list(range(11))
#Complexdata = Complexdata[:100000]
#Start_point = 0
#Start_diff = 20
#sampling_rate = 15e6
#resolution = 500
#Complexdata = Complexdata[:100000]
#STFT(Complexdata, Start_point, Start_diff, len(Complexdata), sampling_rate, resolution)

#create_new_signal_Folder_Big(Folder, firstframe, newfoldername, firstjump=0)

#------------------All Signal------------------

#     folder_name = "./data/Handmove2/"
#     # Complexdata = combine_selected_signals(folder_name,list(range(1950,2200)))
#     # plot_time_complexamp(Complexdata)

#     # #folder_name2 = "./data/moveHand/Channel1/"
#     lazyrawprocess(folder_name)
#     # Complexdata2 = combine_selected_signals(folder_name,list(range(1300, 1302)))
#     # #print(len(Complexdata2))
#     Channelfolder_name =  folder_name.replace("./data/", "./signaldata/")
#     TxFolder = os.path.join(Channelfolder_name, "Channel0")  # 不变的
#     RxFolder = os.path.join(Channelfolder_name, "Channel1")  # 人动的
#     Average_getdata_ampplot(TxFolder,15e6)
#     Average_getdata_ampplot(RxFolder,15e6)
#     CompareChannel(Channelfolder_name, [0,0], 15e6)
#     time = load_time_axis(Channelfolder_name)

# # 生成 y 值，全是 1
#     y_values = np.ones_like(time)

#     # 绘制图表
#     plt.figure(figsize=(10, 4))
#     plt.scatter(time, y_values, marker='|', color='blue', s=100)  # 竖线标记
#     plt.xlabel("Time (s)")
#     plt.ylabel("Signal Presence")
#     plt.title("Signal Start Times")

#     # 显示网格
#     plt.grid(True)

#     # 显示图像
#     plt.show()





# if H_matrix is not None:
        
#         # ======================================================================
#         # === 最终测试：使用更多中心频谱点和更大的L值以获得更高分辨率 ===
#         # ======================================================================
#         print("\n" + "="*50)
#         print("Running Capon with MORE frequency bins and a LARGER subarray L for higher resolution.")
#         print("="*50)

#         # -------------------- 您可以在这里轻松调整参数 --------------------
#         # 我们要用多少个子载波 (M)。增加此值以支持更大的L。
#         subset_num_freqs = 2000

#         # 我们要处理多少个时间帧 (Doppler通道)。保持1000不变。
#         subset_num_times = 1000 

#         # Capon的子阵列长度L。L越大分辨率越高。现在M=2000，我们可以使用更大的L。
#         subarray_L_for_test = 400
#         # ----------------------------------------------------------------

#         # 2a. 自动计算切片的起止索引
#         total_freqs = H_matrix.shape[0]
#         total_times = H_matrix.shape[1]

#         # 计算频谱中心切片的起止点
#         start_freq = (total_freqs - subset_num_freqs) // 2
#         end_freq = start_freq + subset_num_freqs
        
#         # 计算时间轴中心切片的起止点
#         start_time = (total_times - subset_num_times) // 2
#         end_time = start_time + subset_num_times

#         # 2b. 从完整数据中安全地切片出子集
#         if total_freqs >= subset_num_freqs and total_times >= subset_num_times:
#             print(f"Slicing frequency bins from index {start_freq} to {end_freq}.")
#             print(f"Slicing time frames from index {start_time} to {end_time}.")
            
#             H_matrix_subset = H_matrix[start_freq:end_freq, start_time:end_time]
            
#             print(f"Using a new subset of H_matrix with shape: {H_matrix_subset.shape}")
            
#             # 性能警告
#             print(f"Subarray L increased from 100 to {subarray_L_for_test}. This will be significantly slower.")
#             print("Please be patient, this may take several minutes...")

#             # 2c. 调用Capon DDM生成函数
#             generate_ddm_with_capon(
#                 H_f_t=H_matrix_subset,
#                 adc_sampling_rate=SAMPLING_RATE,
#                 frame_rate=F_rate,
#                 fft_length=N_fft,
#                 subarray_L=subarray_L_for_test
#             )
#             print("\nCapon run on the new, larger subset finished.")

#         else:
#             print("Error: Full H_matrix is too small for the defined subset.")
#             print("Please adjust subset sizes in the script.")
#         # ======================================================================
#         # === Capon算法测试结束 ===
#         # ======================================================================


