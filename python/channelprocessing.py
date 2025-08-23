import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import windows,stft
from tqdm import tqdm
import os
from matplotlib.animation import FuncAnimation
import pandas as pd

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



channel_matrix, time = load_channel_data("./data/channel_dat.npz")

generate_complex_scatter_video(channel_matrix, time, output_file="./video/fastmove.mp4")