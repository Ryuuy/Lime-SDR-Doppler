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
import cupy as cp

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

def read_bin_file(file_name):
    # 读取 bin 文件并解释为 int16 数据
    raw_data = np.fromfile(file_name, dtype=np.int16)

    # 分离 I 和 Q 数据
    I_data = raw_data[0::2]
    Q_data = raw_data[1::2]

    return I_data, Q_data


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


    return channel_response_matrix, time_axis
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

    return channel_responses

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

    else:
        STFT(combinedsignalstable, 0, Start_diff, len(combinedsignalstable), STFTsampling_rate, resolution, freq_range=STFTfreqrange)


    return 0



if __name__ == "__main__":
    # 参数设
    SAMPLING_RATE = 15e6

    folder_name = "./data/18_Slider_diff/"
    Channelfolder_name =  folder_name.replace("./data/", "./signaldata/")
    lazydoppler(Channelfolder_name,15e6,2e6,100,2000,400,avgcenterfreq=True,nolinear=True,STFTfreqrange=[-50,50])
    