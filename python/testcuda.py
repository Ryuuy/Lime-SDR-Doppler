# -*- coding: utf-8 -*-
import os
import numpy as np  # Numpy仍然需要，用于文件IO和CPU操作
import cupy as cp   # NEW: 导入Cupy作为GPU计算的核心
from cupy.fft import fft, fftshift, ifft, fftfreq # MODIFIED: 使用cupy.fft替换numpy/scipy的fft
# 把它改成这样:
from scipy.signal.windows import gaussian # Scipy中我们只用到了这个函数
from tqdm import tqdm
import matplotlib.pyplot as plt
import finufft

# --- 以下函数涉及文件处理和CPU操作，无需修改 ---

def process_files(folder_name):
    """
    frame_length每个频率的长度frame
    """
    bin_files = [
        os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.bin')
    ]
    sorted_files = sorted(
        bin_files,
        key=lambda x: (
            int(x.split("center")[1].split("frame")[0]),
            int(x.split("frame")[1].split(".bin")[0])
        )
    )
    frequency_groups = {}
    for file in sorted_files:
        base_name = os.path.basename(file)
        center_freq = int(base_name.split("center")[1].split("frame")[0])
        frame_number = int(base_name.split("frame")[1].split(".bin")[0])
        if center_freq not in frequency_groups:
            frequency_groups[center_freq] = []
        frequency_groups[center_freq].append(frame_number)
    gap_length = None
    for freq, frames in frequency_groups.items():
        sorted_frames = sorted(frames)
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] > 1:
                gap_length = sorted_frames[i - 1] + 1
                break
        if gap_length is not None:
            frame_length = gap_length
            break
        else:
            frame_length = len(sorted_frames)
    frequency_frame_groups = [
        len(frames) // frame_length for frames in frequency_groups.values()
    ]
    center_freqs = sorted(frequency_groups.keys())
    return sorted_files, center_freqs, frame_length, frequency_frame_groups

def read_bin_file(file_name):
    raw_data = np.fromfile(file_name, dtype=np.int16)
    I_data = raw_data[0::2]
    Q_data = raw_data[1::2]
    return I_data, Q_data

# --- 以下函数是计算核心，需要进行GPU化修改 ---

def CompareChannelAvgFreq(Folder, DC_Calibration, sampling_rate, signalfilter=False, AvgFreqband=None):
    TxFolder = os.path.join(Folder, "Channel0")
    RxFolder = os.path.join(Folder, "Channel1")
    rx_sorted_files, rx_center_freqs, rx_frame_length, rx_frequency_frame_groups = process_files(RxFolder)
    tx_sorted_files, tx_center_freqs, tx_frame_length, tx_frequency_frame_groups = process_files(TxFolder)
    
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)
    FFTLength = total_data_count // 2
    
    if len(rx_sorted_files) != len(tx_sorted_files):
        raise ValueError("The number of files in RxFolder and TxFolder must be the same.")
    if rx_center_freqs != tx_center_freqs or rx_frame_length != tx_frame_length:
        raise ValueError("The structure of RxFolder and TxFolder must match.")

    channel_response_matrix = []
    valid_frames = []
    
    # MODIFIED: 使用tqdm显示循环进度条
    num_total_frames = sum(rx_frame_length * group for group in rx_frequency_frame_groups)
    
    with tqdm(total=num_total_frames, desc="Processing Frames (GPU)") as pbar:
        for k, center_freq in enumerate(rx_center_freqs):
            for frame_idx in range(rx_frame_length * rx_frequency_frame_groups[k]):
                rx_file_path = rx_sorted_files[frame_idx]
                tx_file_path = tx_sorted_files[frame_idx]
                rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
                tx_I_data, tx_Q_data = read_bin_file(tx_file_path)

                rx_complex_signal_cpu = rx_I_data + 1j * rx_Q_data
                tx_complex_signal_cpu = tx_I_data + 1j * tx_Q_data

                # NEW: 将数据从CPU上传到GPU
                rx_complex_signal = cp.asarray(rx_complex_signal_cpu)
                tx_complex_signal = cp.asarray(tx_complex_signal_cpu)

                # MODIFIED: 在GPU上执行FFT和所有后续计算
                rx_fft_result = fftshift(fft(rx_complex_signal))
                tx_fft_result = fftshift(fft(tx_complex_signal))
                
                # ... DC校准部分也可以直接在GPU上做 ...
                
                channel_response = rx_fft_result / tx_fft_result

                if AvgFreqband is not None:
                    startidx = int((AvgFreqband[0] / sampling_rate + 0.5) * len(channel_response))
                    finishidx = int((AvgFreqband[1] / sampling_rate + 0.5) * len(channel_response))
                    channel_response = channel_response[startidx:finishidx] # 在GPU上切片

                real_median = cp.median(channel_response.real)
                imag_median = cp.median(channel_response.imag)
                median_complex = real_median + 1j * imag_median

                # NEW: 将最终标量结果从GPU下载回CPU
                channel_response_matrix.append(median_complex.get())
                valid_frames.append(frame_idx)
                pbar.update(1)

    print(f"All frames processed for {len(rx_center_freqs)} frequency bands.")
    
    # MODIFIED: time_axis在CPU上计算，所以用numpy
    time_axis = np.array(valid_frames) * FFTLength / sampling_rate
    channel_response_matrix = np.array(channel_response_matrix)

    return channel_response_matrix, time_axis

def ChannelSpecificFreq(Folder, sampling_rate, selected_frequency):
    TxFolder = os.path.join(Folder, "Channel0")
    RxFolder = os.path.join(Folder, "Channel1")
    rx_sorted_files, _, _, _ = process_files(RxFolder)
    tx_sorted_files, _, _, _ = process_files(TxFolder)
    
    with open(rx_sorted_files[0], "rb") as f:
        total_data = np.fromfile(f, dtype=np.int16)
        total_data_count = len(total_data)
    FFTLength = total_data_count // 2
    
    # MODIFIED: 在CPU上准备频率轴
    freq_axis_cpu = np.fft.fftfreq(FFTLength, d=1 / sampling_rate)
    if selected_frequency < freq_axis_cpu.min() or selected_frequency > freq_axis_cpu.max():
        raise ValueError(f"Selected frequency {selected_frequency} Hz is out of range.")
    
    selected_index = np.argmin(np.abs(freq_axis_cpu - selected_frequency))
    print(f"Selected frequency index: {selected_index}")

    # NEW: 在GPU上创建DFT权重向量，避免在循环中重复创建和上传
    n = cp.arange(FFTLength)
    e_term = cp.exp(-1j * 2 * cp.pi * selected_index * n / FFTLength)

    channel_responses = []
    
    for rx_file_path, tx_file_path in tqdm(zip(rx_sorted_files, tx_sorted_files), total=len(rx_sorted_files), desc="Processing Specific Freq (GPU)"):
        rx_I_data, rx_Q_data = read_bin_file(rx_file_path)
        tx_I_data, tx_Q_data = read_bin_file(tx_file_path)
        
        rx_complex_signal_cpu = rx_I_data + 1j * rx_Q_data
        tx_complex_signal_cpu = tx_I_data + 1j * tx_Q_data

        # NEW: 上传信号到GPU
        rx_complex_signal = cp.asarray(rx_complex_signal_cpu)
        tx_complex_signal = cp.asarray(tx_complex_signal_cpu)
        
        # MODIFIED: 在GPU上进行点积(sum)运算
        rx_fft_result = cp.sum(rx_complex_signal * e_term)
        tx_fft_result = cp.sum(tx_complex_signal * e_term)

        if tx_fft_result != 0:
            channel_response = rx_fft_result / tx_fft_result
            # NEW: 下载结果回CPU
            channel_responses.append(channel_response.get())

    channel_responses = np.array(channel_responses)
    return channel_responses

def Channel_Cali(channel_matrix_gpu): # MODIFIED: 假设输入是GPU数组
    # MODIFIED: 在GPU上计算
    mean_value = cp.mean(channel_matrix_gpu)
    channel_matrix_gpu -= mean_value
    print(mean_value) # 打印cupy标量会自动下载
    return channel_matrix_gpu # MODIFIED: 返回GPU数组

def load_time_axis(newfoldername):
    time_axis_path = os.path.join(newfoldername, "time_axis.npy")
    if not os.path.exists(time_axis_path):
        raise FileNotFoundError(f"时间轴文件未找到: {time_axis_path}")
    return np.load(time_axis_path)

def nolinearSTFT(timeaxis, complexsignal_gpu, resolution, freq_range=None, output_filename_base=None, time_stretch_factor=1.7, Big_picture=True):
    # MODIFIED: complexsignal_gpu 是一个CuPy数组
    
    # ... 绘图参数设置不变 ...
    if Big_picture:
        tick_size, cbar_label_size, axis_label_size, figsize = 30, 32, 36, (12, 8)
    else:
        tick_size, cbar_label_size, axis_label_size, figsize = 18, 20, 22, (12, 5)

    # MODIFIED: 时间轴仍在CPU上处理
    timeaxis_for_plotting = np.array(timeaxis, dtype=np.float64) * time_stretch_factor

    K_FFT = 0.000195313 ** 2 / 50 / 0.001
    dbm_results = []
    time_axis_plot_points = []
    
    samplerate = len(timeaxis) / (timeaxis[-1] - timeaxis[0])
    # MODIFIED: 频率轴仍在CPU上处理
    freq_axis = np.linspace(-samplerate / 2, samplerate / 2, resolution)

    if freq_range is not None:
        min_freq, max_freq = freq_range
        startidx = np.searchsorted(freq_axis, min_freq)
        finishidx = np.searchsorted(freq_axis, max_freq)
    else:
        startidx, finishidx = 0, len(freq_axis)
        
    # NEW: 为finufft准备GPU选项
    gpu_opts = {'gpu_device_id': 0}

    for i in tqdm(range(len(timeaxis) // 5), desc="Non-linear STFT (GPU)"):
        start_idx = i * 5
        end_idx = start_idx + resolution
        if end_idx > len(timeaxis):
            break
        
        window_time_cpu = timeaxis_for_plotting[start_idx:end_idx]
        # MODIFIED: 直接在GPU上切片，无需数据传输
        window_signal_gpu = complexsignal_gpu[start_idx:end_idx]

        time_span = window_time_cpu[-1] - window_time_cpu[0]
        if time_span == 0:
            continue

        normalized_time_cpu = (window_time_cpu - window_time_cpu[0]) / time_span * 2 * np.pi

        # NEW: The Fix! Convert the GPU slice to a CPU NumPy array before passing to finufft
        window_signal_cpu = cp.asnumpy(window_signal_gpu)

# MODIFIED: Pass the CPU array to finufft; the calculation still happens on the GPU
# but the result is returned as a NumPy array.
        spectrum_nufft_cpu = finufft.nufft1d1(normalized_time_cpu, window_signal_cpu, resolution, **gpu_opts)

        # MODIFIED: Since the result is a NumPy array, use np for subsequent calculations.
        fft_magnitude = np.abs(spectrum_nufft_cpu)
        fft_dbm = 10 * np.log10(K_FFT * (fft_magnitude) ** 2 + 1e-20) 
        fft_dbm_filtered = fft_dbm[startidx:finishidx]

        # MODIFIED: The data is already on the CPU, so just append it (no .get() needed).
        dbm_results.append(fft_dbm_filtered)
        time_axis_plot_points.append(window_time_cpu[0])

    if not dbm_results:
        print("Warning: No results were generated to plot.")
        return None

    # MODIFIED: dbm_results现在是numpy数组列表，直接用np.array
    dbm_results = np.array(dbm_results).T
    dbm_results = dbm_results[::-1, :]
    
    # --- 绘图逻辑 (在CPU上)，无需修改 ---
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(time_axis_plot_points, freq_axis[startidx:finishidx], dbm_results, shading='auto', cmap='jet', vmin=-80, vmax=-30)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Channel Strength (dB)', fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    ax.set_xlabel('Time (s)', fontsize=axis_label_size)
    ax.set_ylabel('Doppler Frequency (Hz)', fontsize=axis_label_size)
    ax.grid(True)
    fig.tight_layout(pad=0.5)
    
    if output_filename_base:
        save_dir = 'picture/doppler/'
        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"{output_filename_base}.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f"图片已保存至: {save_path}")
    plt.close(fig)

    return dbm_results


def STFT(complex_signal_gpu, Start_point, Start_diff, Final_point, sample_rate, resolution, Gaussian=False, interpolation_factor=4, freq_range=None):
    # MODIFIED: complex_signal_gpu 是一个CuPy数组
    
    K_FFT = (0.000195313 / (resolution // 2)) ** 2 / 50 / 0.001
    
    dbm_results_gpu = [] # NEW: 创建一个GPU数组的列表
    time_axis = []
    
    if Gaussian:
        # NEW: 在CPU创建高斯窗，然后上传到GPU一次
        gaussian_window_cpu = gaussian(resolution, std=resolution / 4)
        gaussian_window_cpu /= np.max(gaussian_window_cpu)
        gaussian_window = cp.asarray(gaussian_window_cpu)

    # MODIFIED: 频率轴仍在CPU上处理
    freq_axis = np.linspace(-sample_rate / 2, sample_rate / 2, resolution * interpolation_factor, endpoint=False)
    if freq_range is not None:
        min_freq, max_freq = freq_range
        startidx = np.searchsorted(freq_axis, min_freq)
        finishidx = np.searchsorted(freq_axis, max_freq)
    else:
        startidx, finishidx = 0, len(freq_axis)
    
    for i in tqdm(range((Final_point - Start_point) // Start_diff), desc="STFT (GPU)"):
        start_idx = Start_point + i * Start_diff
        end_idx = start_idx + resolution
        if end_idx > Final_point: break

        # MODIFIED: 直接在GPU上切片
        window_signal = complex_signal_gpu[start_idx:end_idx].copy()
        if Gaussian:
            window_signal *= gaussian_window

        # MODIFIED: 在GPU上执行FFT
        fft_result = fft(window_signal, n=resolution * interpolation_factor)
        fft_result = fftshift(fft_result)
        fft_magnitude = cp.abs(fft_result)

        fft_dbm = 10 * cp.log10(K_FFT * (fft_magnitude) ** 2)
        dbm_results_gpu.append(fft_dbm) # NEW: 将GPU数组添加到列表

        time_axis.append(start_idx / sample_rate)

    # NEW: 将GPU数组列表高效地合并，然后一次性下载回CPU
    dbm_results = cp.stack(dbm_results_gpu).T.get()

    dbm_results_filtered = dbm_results[startidx:finishidx, :]
    freq_axis_filtered = freq_axis[startidx:finishidx]

    # --- 绘图逻辑 (在CPU上)，无需修改 ---
    plt.figure(figsize=(10, 6))
    im = plt.imshow(dbm_results_filtered, extent=[time_axis[0], time_axis[-1], freq_axis_filtered[0], freq_axis_filtered[-1]],
                    aspect='auto', origin='lower', cmap='jet', vmin=-140, vmax=-60)
    plt.colorbar(im, label='Channel Strength (dB)')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Short-Time Fourier Transform (STFT)' + (' with Gaussian Window' if Gaussian else ''))
    plt.grid()
    plt.show()

    return dbm_results_filtered
    
def lazydoppler(newfoldername, sampling_rate, selectfreq, Start_diff, STFTsampling_rate, resolution, avgcenterfreq=False, nolinear=False, centerfreqband=(-2e6, 2e6), STFTfreqrange=None):
    if avgcenterfreq:
        # MODIFIED: 此函数返回numpy数组
        combinedsignalstable, _ = CompareChannelAvgFreq(newfoldername, [0, 0], sampling_rate, signalfilter=False, AvgFreqband=centerfreqband)
    else:
        # MODIFIED: 此函数返回numpy数组
        combinedsignalstable = ChannelSpecificFreq(newfoldername, sampling_rate, selectfreq)
    
    # NEW: 将数据上传到GPU，准备进行后续计算
    combinedsignalstable_gpu = cp.asarray(combinedsignalstable)
    # MODIFIED: 校准函数在GPU上操作
    combinedsignalstable_gpu = Channel_Cali(combinedsignalstable_gpu)

    if nolinear:
        # MODIFIED: 加载时间轴用numpy
        StartSamplepoint = load_time_axis(newfoldername)
        time = StartSamplepoint / 15e6
        base_filename = os.path.basename(newfoldername.strip('/'))
        
        # MODIFIED: 将GPU数组传递给nolinearSTFT
        nolinearSTFT(time, combinedsignalstable_gpu, resolution, freq_range=STFTfreqrange, output_filename_base=base_filename)
    else:
        # MODIFIED: 将GPU数组传递给STFT
        STFT(combinedsignalstable_gpu, 0, Start_diff, len(combinedsignalstable_gpu), STFTsampling_rate, resolution, freq_range=STFTfreqrange)

    return 0

if __name__ == "__main__":
    # --- 参数设置 ---
    SAMPLING_RATE = 15e6
    folder_name = "./data/18_Slider_diff/"
    Channelfolder_name = folder_name.replace("./data/", "./signaldata/")
    
    # --- 运行主函数 ---
    lazydoppler(
        Channelfolder_name,
        15e6,
        2e6,
        100,
        2000,
        400,
        avgcenterfreq=True,
        nolinear=True,
        STFTfreqrange=[-50, 50]
    )