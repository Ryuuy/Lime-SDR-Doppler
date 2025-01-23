# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows,stft,gaussian
from tqdm import tqdm
import os
from matplotlib.animation import FFMpegWriter,FuncAnimation
from collections import Counter

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
        dbm_matrix_expanded, aspect='auto', cmap='RdYlBu_r', vmin=-10, vmax=10,
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

def plot_3d_amp(dbm_matrix, FrameNumber, time_axis, FFTLength, freq_axis, output_file):
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
        dbm_matrix_expanded, aspect='auto', cmap='plasma', vmin=0, vmax=0.8,
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
                fft_result = fftshift(fft(complex_signal[i*FFTLength//N:(i+1)*FFTLength//N]))
                fft_magnitude = np.abs(fft_result)
                avg_magnitude = np.mean(fft_magnitude)
                # 计算 dBm 值
                # 计算该帧的平均 dBm 值
                avg_values.append(avg_magnitude)

            # 打印每帧的信息

            # 更新进度条
            progress_bar.update(1)

    # 根据 FFTLength 和 sampling_rate 计算时间轴
    time_axis = np.arange(len(avg_values)) * FFTLength / sampling_rate  # 每帧时间间隔

    # 绘制平均 dBm 的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, avg_values, '-')
    plt.title('Average dBm per Frame')
    plt.xlabel('Time (s)')
    plt.ylabel('Average dBm')
    plt.grid()

    # 保存平均 dBm 图像到指定目录
    output_file = os.path.join(output_dir, f"{base_folder_name}avgamp.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # 再保存一份带 DC 校准后缀的图片
    output_file_with_dc = os.path.join(output_dir, f"{base_folder_name}avgamp_withDCcalibration.png")
    plt.savefig(output_file_with_dc)
    print(f"Plot saved to {output_file_with_dc}")

    plt.close()

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




def STFT(complex_signal, Start_point, Start_diff, Final_point, sample_rate, resolution, Gaussian=False, interpolation_factor=1, freq_range=(-500, 500)):
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
    - freq_range: tuple, frequency range to display (min_freq, max_freq)

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
        dbm_results.append(fft_dbm)

        # Record time point
        time_point = start_idx / sample_rate
        time_axis.append(time_point)

    # Normalize dBm_results to the longest window length (for consistent plotting)
    max_length = max(len(dbm) for dbm in dbm_results)
    dbm_results_padded = [np.pad(dbm, (0, max_length - len(dbm)), constant_values=np.nan) for dbm in dbm_results]

    # Generate frequency axis
    interpolated_sample_rate = sample_rate * interpolation_factor
    freq_axis = np.linspace(-interpolated_sample_rate / 2, interpolated_sample_rate / 2, resolution * interpolation_factor, endpoint=False)

    # Apply frequency range filtering
    min_freq, max_freq = freq_range
    freq_mask = (freq_axis >= min_freq) & (freq_axis <= max_freq)
    freq_axis_filtered = freq_axis[freq_mask]
    dbm_results_filtered = [dbm[freq_mask] for dbm in dbm_results_padded]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.imshow(
        np.array(dbm_results_filtered).T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], min_freq, max_freq],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label='Power (dBm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Short-Time Fourier Transform (STFT)' + (' with Gaussian Window' if Gaussian else ''))
    plt.grid()
    plt.show()

    return dbm_results

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
    freq_mask = (freq_axis >= 0) & (freq_axis <= 100)
    dbm_results_filtered = np.array(dbm_results_padded)[:, freq_mask]
    freq_axis_filtered = freq_axis[freq_mask]

    # Create output directory
    output_dir = os.path.join("picture", "doppler")
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    output_file = os.path.join(output_dir, "stft_doppler_0_100Hz.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(
        dbm_results_filtered.T,  # Transpose for correct orientation
        extent=[time_axis[0], time_axis[-1], freq_axis_filtered[0]*0.012*np.pi, freq_axis_filtered[-1]*0.012*np.pi],
        aspect='auto',
        origin='lower',
        cmap='jet'
    )
    plt.colorbar(label='Power (dBm)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity(m/s)')
    plt.title('MicroDoppler')
    plt.grid()
    plt.savefig(output_file)
    plt.close()
    print(f"STFT plot (0-100 Hz) saved to {output_file}")

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

    # 遍历每个中心频率和时间帧，计算信道响应
    index = 0
    total_files = len(rx_sorted_files)

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
    base_folder_name = os.path.basename(os.path.dirname(RxFolder))
    if avgminus==True:
        avg_value = np.mean(channel_response_matrix)
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


    return channel_response_matrix, freq_axis, time_axis


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

def CompareChannelAvgFreq(Folder, DC_Calibration, sampling_rate, signalfilter=False):
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
            channel_response_matrix.append(np.mean(channel_response))
            valid_frames.append(frame)

    print(f"All frames processed for {len(rx_center_freqs)} frequency bands.")

    # 构造时间轴
    time_axis = np.array(valid_frames) * FFTLength / sampling_rate
    channel_response_matrix = np.array(channel_response_matrix)

    # 绘制振幅和相位
    amplitude_output_file = os.path.join(Folder, "Channel_Response_Amplitude.png")
    phase_output_file = os.path.join(Folder, "Channel_Response_Phase.png")

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.abs(channel_response_matrix), label="Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Channel Response Amplitude")
    plt.grid()
    plt.legend()
    plt.savefig(amplitude_output_file)
    plt.close()
    print(f"Amplitude plot saved to {amplitude_output_file}")

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.angle(channel_response_matrix), label="Phase")
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (radians)")
    plt.title("Channel Response Phase")
    plt.grid()
    plt.legend()
    plt.savefig(phase_output_file)
    plt.close()
    print(f"Phase plot saved to {phase_output_file}")

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
        angle_matrix_expanded, aspect='auto', cmap='plasma', vmin=-0.3*np.pi, vmax=0.3*np.pi,
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
def CompareChanneldetail(RxFolder, TxFolder,Start_diff,resolution, DC_Calibration, sampling_rate, signal_frame):

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

    # 获取 Rx 和 Tx 文件夹中的文件路径和元数据
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)
    rx_complex_signal = np.array([], dtype=np.complex64)
    tx_complex_signal = np.array([], dtype=np.complex64)
    FFTLength = resolution  # 每个时间帧的 FFT 长度
    # 确保 Rx 和 Tx 文件结构一致
    # 初始化存储信道响应的矩阵
    num_freq_bands = len(rx_center_freqs)
    for frame in signal_frame:
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
        rx_complex_signal = np.concatenate((rx_complex_signal, rx_complex_signal_frame))
        tx_complex_signal = np.concatenate((tx_complex_signal, tx_complex_signal_frame))
    
    total_frames = (len(rx_complex_signal)-resolution)//Start_diff
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
    print(abs_channel_response[:,0,3004+resolution//2])
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
    plt.figure(figsize=(6, 6))
    plt.scatter(channel_responses.real, channel_responses.imag, s=5, label=f"Frequency: {selected_frequency/1e6:.2f} MHz")
    plt.title(f"Complex Response Scatter Plot at {selected_frequency / 1e6:.2f} MHz")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')  # 确保比例一致
    plt.legend()
    plt.show()
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
def plot_complex_scatter(channel_matrix, output_file):
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

    print(f"Combined signal length: {len(combined_signal)} samples")

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

def create_new_signal_Folder(Folder, firstframe, newfoldername):
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

if __name__ == "__main__":
    # 参数设置

    folder_name = "./data/Insideboxhavesignal60dbm2386delay"
    folder_name2 = "./data/IQtestnoCali/Channel1"
    Calibrationfolder_name= "./data/InsideboxNosignal2386"
    file_number = 2300  # 选择文件编号
    Start_point = 0
    Start_diff = 40
    resolution = 512
    peaknumber = 1
    sampling_rate = 15e6
    Final_point = Start_point + Start_diff * 20 + resolution
    
    #widerdiffertimetry("./data/40Mbandwidthcase1",[(-0.492472) + (-0.492484)*1j],sampling_rate)
    # 读取文件数据
    Output = []
    #avg_values = Average_getdata_amp(folder_name,sampling_rate)
    #print(len(avg_values))
    #CalculateDopplerWithHamming(avg_values, Start_point, 1, len(avg_values), 15e6//16384, len(avg_values)//4)
 
    #TimedomainFreqoffset(folder_name2, Start_point, Start_diff, Final_point, sampling_rate, resolution, peaknumber, Output,framejumppoint=2)
    #Output = []
    #TimedomainFreqoffset(folder_name2, Start_point, Start_diff, Final_point, sampling_rate, resolution, peaknumber, Output,framejumppoint=5)
    #TimedomainFreqoffset(folder_name2, Start_point, Start_diff, Final_point, sampling_rate, resolution, peaknumber, Output,framejumppoint=0)
    #RxFolder = "./data/test"
    #TxFolder = "./data/test_withdelay10ms"
    #DC_Calibration_value_Rx = LongDCCalibration(folder_name,3)
    #DC_Calibration_value_Tx = LongDCCalibration(folder_name2,3)
    #print(DC_Calibration_value)
    #Average_freq_plotminus(folder_name,sampling_rate)

    

    #process_bin_files_to_video(folder_name2,60,sampling_rate,DCcalibration=(-0.492472 + -0.492484*1j))
    #print(folder_name)
    #print(LongDCCalibration(folder_name,3))
    #CompareChannel(folder_name, folder_name2, [0], sampling_rate)
    #Average_freq_plotminus(folder_name,sampling_rate)

    #process_bin_files_to_video(folder_name,60,sampling_rate,DCcalibration=(0))
    #process_bin_files_to_video(folder_name2,60,sampling_rate,DCcalibration=(-0.492472 + -0.492484*1j))
    #print(folder_name)
    #print(LongDCCalibration(folder_name,3))
    # CompareChannel(folder_name2, folder_name, [-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate)  

    # Signal = longestframe("./data/wifi2445stable/")
    #folder_name = "./data/nohumananother2501Ghz/"
    #readtxt_to_bin(folder_name)
    #arr = read_bin_to_array(folder_name)

    #print(count_continuous_segments(arr))
    newfoldername = "./data/fourHumannoMoveSignal/"
    folder_name = "./data/fourHumannoMove2501Ghz/"
    arr = read_bin_to_array(folder_name)
    firstframe = returnfirstsignalframe(arr)
    create_new_signal_Folder(folder_name,firstframe,newfoldername)
    channel_response_matrix, freq_axis, time_axis = CompareChannelOld(newfoldername,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=False,avgminus=False)
    #create_new_signal_Folder(folder_name,firstframe,newfoldername)
    #timedomainpic("./data/fastwalkSignal/Channel1/",1600)
    #timedomainpic("./data/fastwalkSignal/Channel0/",1600)
    #selected_frequency = 2e6  # 目标频率 (Hz)
    combinedsignalstable, time_axis = CompareChannelAvgFreq(newfoldername,[0,0], sampling_rate)
# 提取目标频率的复数响应
    #combinedsignalstable = extract_frequency_response(channel_response_matrix, freq_axis, selected_frequency)
    # amplitude = np.angle(combinedsignalstable)

    # # 创建横轴
    # x_axis = np.arange(len(combinedsignalstable))

    # 绘制振幅
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_axis, amplitude, label="Amplitude", color="blue")
    # plt.title("Amplitude of Combined Signal")
    # plt.xlabel("Index")
    # plt.ylabel("Amplitude")
    # plt.grid()
    # plt.legend()
    # plt.show()
    #combinedsignalstable = ChannelSpecificFreq(newfoldername,sampling_rate,2e6)
    
    # amplitude = np.abs(combinedsignalstable)

    # # 创建横轴
    # x_axis = np.arange(len(combinedsignalstable))

    # # 绘制振幅
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_axis, amplitude, label="Amplitude", color="blue")
    # plt.title("Amplitude of Combined Signal")
    # plt.xlabel("Index")
    # plt.ylabel("Amplitude")
    # plt.grid()
    # plt.legend()
    # plt.show()
    Start_point = 0
    Start_diff = 1
    resolution = 500 #0.5s
    sampling_rate = 1e3 #0.001s
    combinedsignalstable = Channel_Cali(combinedsignalstable)
    STFT(combinedsignalstable, Start_point, Start_diff, len(combinedsignalstable), sampling_rate, resolution,Gaussian=True,interpolation_factor=4, freq_range=(-500, 500))
    #timedomainpic("./data/fourHumannoMove2501Ghz/Channel0/",31)

    # newfoldername = "./data/fourHumanSignal/"
    # folder_name = "./data/fourHumannoMove2501Ghz/"
    # arr = read_bin_to_array(folder_name)
    # firstframe = returnfirstsignalframe(arr)
    # create_new_signal_Folder(folder_name,firstframe,newfoldername)

    # newfoldername = "./data/handmovingSignal/"
    # folder_name = "./data/handmoving2501Ghz/"
    # arr = read_bin_to_array(folder_name)
    # firstframe = returnfirstsignalframe(arr)
    # create_new_signal_Folder(folder_name,firstframe,newfoldername)

    # newfoldername = "./data/breathSignal/"
    # folder_name = "./data/breath2501Ghz/"
    # arr = read_bin_to_array(folder_name)
    # firstframe = returnfirstsignalframe(arr)
    # create_new_signal_Folder(folder_name,firstframe,newfoldername)

    # folder_name = "./data/fourHumannoMove2501Ghz/"
    # #timedomainpic(folder_name,)
    # arr = read_bin_to_array(folder_name)
    # count_continuous_segments(arr)
    # timedomainpic("./data/fourHumannoMove2501Ghz/Channel0",returnfirstsignalframe(arr)+19)
    

    # folder_name = "./data/handmoving2501Ghz/"
    # arr = read_bin_to_array(folder_name)
    # timedomainpic("./data/handmoving2501Ghz/Channel0",returnfirstsignalframe(arr)+19)

    # print(count_continuous_segments(arr))
    # folder_name = "./data/breath2501Ghz/"
    # arr = read_bin_to_array(folder_name)
    # print(returnfirstsignalframe(arr))
    # timedomainpic("./data/breath2501Ghz/Channel0",returnfirstsignalframe(arr)+19)
    # print(count_continuous_segments(arr))


    # timedomainpic("./data/fastwalk2501Ghz/Channel0",returnfirstsignalframe(arr)+19)
    


    #time_frame = list(range(1, 200))
    #combinedsignalstable = combine_selected_signals("./data/fastwalk2501Ghz/Channel1",time_frame)
    #STFT(combinedsignalstable, Start_point, Start_diff, len(combinedsignalstable), sampling_rate, resolution)
    #combinedsignalhumanmove = combine_selected_signals("./data/fastwalk2501Ghz/Channel1",[1:200])#人动变量
    #print(find_signal_delay(combinedsignalstable,combinedsignalhumanmove))
    

    # Signal = readFilter(folder_name)
    # channel_matrix,time = CompareChannelAvgFreq(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    # save_channel_data("./data/fastwalk.npz", channel_matrix, time)
    # channel_matrix,time=load_channel_data("./data/fastwalk.npz")
    # plot_complex_scatter(channel_matrix,output_file="./picture/fastmove.png")
    # # channel_matrix_npz,time_npz=Channel_Cali(channel_matrix,time)
    # # save_channel_data("./data/fastmovecali.npz", channel_matrix_npz, time_npz)
    # plot_complex_scatter(channel_matrix,output_file="./picture/fastmovecali.png")
    # CompareChannel(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    # channel_matrix,time = load_channel_data("./data/handmovingcali.npz")
    # print(len(channel_matrix))
    
    #channel_matrix,time = load_channel_data("./data/fastmovecali.npz")
    #plot_amplitude_and_phase(channel_matrix,time, "./picture/fastmoveamp.png", "./picture/fastmovepha.png")
    # folder_name = "./data/nohumananother2501Ghz/"
    # Signal = readFilter(folder_name)

    #CompareChannel(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True,avgminus=True)

    # channel_matrix,time = CompareChannelAvgFreq(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    
    #channel_matrix,time = load_channel_data("./data/nohumananothercali.npz")
    #plot_amplitude_and_phase(channel_matrix,time, "./picture/nohumananothercaliamp.png", "./picture/nohumananothercalipha.png")
    # save_channel_data("./data/nohumananother.npz", channel_matrix, time)
    # plot_complex_scatter(channel_matrix,output_file="./picture/nohumananother.png")
    # #save_channel_data("./data/handmovingcali.npz", channel_matr, time_npz)
    # channel_matrix,time = Channel_Cali(channel_matrix,time)
    #save_channel_data("./data/nohumananothercali.npz", channel_matrix, time)
    #plot_complex_scatter(channel_matrix,output_file="./picture/nohumananothercali.png")
    

    # folder_name = "./data/breath2501Ghz/"
    # Signal = readFilter(folder_name)
    # CompareChannel(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    # channel_matrix,time = CompareChannelAvgFreq(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    
    # #channel_matrix,time = load_channel_data("./data/handmoving.npz")
    # save_channel_data("./data/breath.npz", channel_matrix, time)
    # plot_complex_scatter(channel_matrix,output_file="./picture/breath2501Ghz.png")
    # #channel_matrix,time = load_channel_data("./data/handmovingcali.npz")
    # #save_channel_data("./data/handmovingcali.npz", channel_matr, time_npz)
    # channel_matrix,time = Channel_Cali(channel_matrix,time)
    # save_channel_data("./data/breathcali.npz", channel_matrix, time)
    # plot_complex_scatter(channel_matrix,output_file="./picture/breath2501Ghzcali.png")
    # plot_amplitude_and_phase(channel_matrix[1000:1100],time[1000:1100], "./picture/handmoving.png", "./picture/handmovingcali.png")
    # generate_complex_scatter_video(channel_matrix, time, output_file="./video/fastmove.mp4")
    

    # channel_matrix,time = load_channel_data("./data/fourHumannoMove.npz")
    # plot_complex_scatter(channel_matrix,output_file="./picture/fourHumannoMove.png")
    #channel_matrix,time = load_channel_data("./data/fourHumannoMovecali.npz")
    # #save_channel_data("./data/fourHumannoMovecali.npz", channel_matrix_npz, time_npz)
    # #CompareChannel(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,signalfilter=True)
    # plot_complex_scatter(channel_matrix,output_file="./picture/fourHumannoMovecali.png")
    #plot_amplitude_and_phase(channel_matrix,time, "./picture/fourHumannoMovecaliamp.png", "./picture/fourHumannoMovecaliphi.png")
    # # # CompareChannel(folder_name,[-0.492472 + -0.492484*1,-0.492472 + -0.492484*1], sampling_rate,)
    # # Signal = readFilter(folder_name)
    # # CompareChanneldetailSpecificFreqPlot(folder_name, Start_diff, resolution, sampling_rate, ffre,Signal, skipfirst=True,TakeAvg=True)
    
    