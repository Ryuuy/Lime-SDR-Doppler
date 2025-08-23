# caf_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ==============================================================================
# 核心CAF计算函数 (与之前版本相同，但作为模块的一部分)
# ==============================================================================

def calculate_caf_window(ref_data, sur_data, time_axis_sec, doppler_axis, delay_axis_sec, fs):
    """为一个数据窗口计算完整的二维CAF矩阵"""
    caf_matrix = np.zeros((len(doppler_axis), len(delay_axis_sec)), dtype=np.complex128)
    
    for i, fd in enumerate(doppler_axis):
        # 预计算多普勒补偿项
        doppler_phasor = np.exp(-1j * 2 * np.pi * fd * time_axis_sec)
        
        for j, delay_sec in enumerate(delay_axis_sec):
            delay_samples = int(delay_sec * fs)
            # 使用np.roll进行简化的循环移位来实现延迟
            ref_data_delayed = np.roll(ref_data, delay_samples)
            
            # 计算 x_s(t) * x_r*(t - τ')
            product = sur_data * np.conj(ref_data_delayed)
            
            # 乘以多普勒补偿项并积分
            caf_point = np.sum(product * doppler_phasor)
            caf_matrix[i, j] = caf_point
            
    return caf_matrix

def load_data_for_window(file_list, time_stamps_samples, fs):
    """为一个窗口加载所有数据并创建精确的时间轴"""
    all_data = []
    all_times = []
    for i in range(len(file_list)):
        data = np.load(file_list[i])
        all_data.append(data)
        
        start_time_sec = time_stamps_samples[i] / fs
        sample_times = start_time_sec + np.arange(len(data)) / fs
        all_times.append(sample_times)
        
    return np.concatenate(all_data), np.concatenate(all_times)


# ==============================================================================
#  ★★ 新的封装函数，可以被你的主程序直接调用 ★★
# ==============================================================================
def lazy_caf_doppler(channelfolder_name, 
                     sampling_rate,
                     window_size=400,
                     slide_step=100,
                     doppler_range=(-500, 500),
                     doppler_resolution=2.0,
                     time_stretch_factor=1.7,
                     output_filename_base=None):
    """
    一个模仿你lazydoppler接口的函数，但使用CAF方法生成多普勒谱图。
    (已修正文件加载逻辑，使其更健壮)
    """
    
    print("--- 开始执行基于交叉相关(CAF)的多普勒分析 (版本 2.0) ---")
    
    # --- 1. 数据加载 (★★ 这里是修正的部分 ★★) ---
    ref_channel_name = "Channel0"
    sur_channel_name = "Channel1"
    
    time_axis_path = os.path.join(channelfolder_name, "time_axis.npy")
    if not os.path.exists(time_axis_path):
        print(f"错误: 找不到时间戳文件 {time_axis_path}")
        return

    time_stamps_samples = np.load(time_axis_path)
    
    try:
        # -- 修正的文件加载逻辑 --
        ref_dir = os.path.join(channelfolder_name, ref_channel_name)
        sur_dir = os.path.join(channelfolder_name, sur_channel_name)

        # 过滤出文件名是纯数字的文件，并构建完整路径
        ref_files_filtered = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if os.path.splitext(f)[0].isdigit()]
        sur_files_filtered = [os.path.join(sur_dir, f) for f in os.listdir(sur_dir) if os.path.splitext(f)[0].isdigit()]

        # 按文件名中的数字进行排序
        ref_files = sorted(ref_files_filtered, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        sur_files = sorted(sur_files_filtered, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    except FileNotFoundError:
        print(f"错误: 请确保 {channelfolder_name} 目录下存在 '{ref_channel_name}' 和 '{sur_channel_name}' 文件夹。")
        return

    total_frames = min(len(ref_files), len(sur_files), len(time_stamps_samples))
    print(f"数据加载完成，共找到 {total_frames} 个有效数字文件名数据帧。")

    # --- 后续部分的代码完全不变 ---
    # --- 2. CAF参数定义 ---
    doppler_axis = np.arange(doppler_range[0], doppler_range[1] + doppler_resolution, doppler_resolution)
    delay_axis_sec = np.array([0.0]) # 零延迟假设

    # --- 3. 滑动窗口处理 ---
    spectrogram_data = []
    spectrogram_time_axis = []
    
    # ... (这部分的所有代码都和你本地的一样，所以省略) ...
    # ... The rest of the function remains the same as the one I sent you before ...
    for i in tqdm(range(0, total_frames - window_size + 1, slide_step), desc="Processing Windows with CAF"):
        window_slice = slice(i, i + window_size)
        
        current_ref_files = ref_files[window_slice]
        current_sur_files = sur_files[window_slice]
        current_time_stamps = time_stamps_samples[window_slice]
        
        # 这里的 load_data_for_window 函数不需要改变
        ref_data, time_axis = load_data_for_window(current_ref_files, current_time_stamps, sampling_rate)
        sur_data, _ = load_data_for_window(current_sur_files, current_time_stamps, sampling_rate)
        
        # 这里的 calculate_caf_for_window 函数不需要改变
        caf_matrix = calculate_caf_for_window(ref_data, sur_data, time_axis, doppler_axis, delay_axis_sec, sampling_rate)
        
        zero_delay_index = np.argmin(np.abs(delay_axis_sec - 0))
        doppler_spectrum = caf_matrix[:, zero_delay_index]
        spectrogram_data.append(np.abs(doppler_spectrum)**2)
        
        spectrogram_time_axis.append(np.mean(time_axis))

    if not spectrogram_data:
        print("处理完成，但没有足够的数据生成谱图。")
        return
        
    # --- 4. 绘图 ---
    print("处理完成，正在绘制多普勒-时间谱图...")
    spectrogram_matrix = np.array(spectrogram_data).T
    # 增加一个极小值防止log(0)
    spectrogram_db = 10 * np.log10(spectrogram_matrix / np.max(spectrogram_matrix) + 1e-20)
    time_axis_for_plotting = np.array(spectrogram_time_axis) * time_stretch_factor

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(
        time_axis_for_plotting,
        doppler_axis,
        spectrogram_db,
        shading='gouraud',
        cmap='jet',
        vmin=-50,
        vmax=0
    )
    
    plt.colorbar(label='归一化功率 (dB)')
    plt.title('Doppler-Time Spectrogram (via Cross-Correlation / CAF at Zero-Delay)')
    plt.xlabel(f'Time (s, stretched by {time_stretch_factor}x)')
    plt.ylabel('Doppler Frequency (Hz)')
    plt.ylim(top=doppler_range[1], bottom=doppler_range[0])
    plt.grid(True)
    
    if output_filename_base:
        save_dir = 'picture/caf_doppler/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{output_filename_base}_caf.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
        print(f"图片已保存至: {save_path}")
        
    plt.show()
if __name__ == '__main__':
    # --- 这是你原来的调用方式 ---
    # folder_name = "./data/5_Breath_diff/"
    # Channelfolder_name = folder_name.replace("./data/", "./signaldata/")
    # # 确保lazyrawprocess已经运行
    # lazydoppler(Channelfolder_name, 15e6, 2e6, 100, 2000, 400, avgcenterfreq=True, nolinear=True, STFTfreqrange=[-20, 20])

    # --- 现在，你可以用几乎一样的方式调用新的CAF方法 ---
    folder_name = "./data/3_Walking_far_diff/" # 换成你想测试的数据集
    Channelfolder_name = folder_name.replace("./data/", "./signaldata/")
    
    # 提取基础文件名，用于保存图片
    base_filename = os.path.basename(Channelfolder_name.strip('/'))

    print("\n\n--- 运行你原来的方法 (频谱比值法+NU-STFT) ---")
    # 为了对比，你可以先运行你自己的函数
    # from your_original_script import lazydoppler # 假设你的函数在另一个文件里
    # lazydoppler(Channelfolder_name, 15e6, 2e6, 100, 2000, 400, avgcenterfreq=True, nolinear=True, STFTfreqrange=[-20, 20])


    print("\n\n--- 运行新的方法 (交叉相关法/CAF) ---")
    lazy_caf_doppler(
        channelfolder_name=Channelfolder_name,
        sampling_rate=15e6,
        window_size=400,
        slide_step=100,
        doppler_range=(-20, 20), # 和你的STFTfreqrange保持一致
        doppler_resolution=0.5, # 可以设置得细一些
        time_stretch_factor=1.7,
        output_filename_base=base_filename
    )