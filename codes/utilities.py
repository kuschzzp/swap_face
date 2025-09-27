import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List

from tqdm import tqdm

import codes.globals as globals

# 临时目录名称常量
TEMP_DIRECTORY = 'temp'
# 临时视频文件名常量
TEMP_VIDEO_FILE = 'temp.mp4'

# 针对macOS系统的SSL补丁，避免SSL证书验证问题
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    """
    执行FFmpeg命令
    
    Args:
        args (List[str]): FFmpeg命令参数列表
        
    Returns:
        bool: 执行成功返回True，失败返回False
    """
    commands = ['ffmpeg', '-hide_banner', '-loglevel', globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as  e:
        print(f'ffmpeg 处理异常 {str(e)}')
        pass
    return False


def detect_fps(target_path: str) -> float:
    """
    检测视频文件的帧率(FPS)

    Args:
        target_path (str): 视频文件路径

    Returns:
        float: 视频帧率，检测失败时返回默认值30
    """
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception as e:
        print(f'ffprobe 处理异常 {target_path}: {str(e)}')
        pass
    return 30


def extract_frames(target_path: str, fps: float = 30) -> bool:
    """
    从视频中提取帧图像
    
    Args:
        target_path (str): 视频文件路径
        fps (float): 提取帧率，默认为30
        
    Returns:
        bool: 提取成功返回True，失败返回False
    """
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = globals.temp_frame_quality * 31 // 100
    return run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + globals.temp_frame_format)])


def create_video(target_path: str, fps: float = 30) -> bool:
    """
    将帧图像合成为视频
    
    Args:
        target_path (str): 目标视频路径
        fps (float): 视频帧率，默认为30
        
    Returns:
        bool: 合成成功返回True，失败返回False
    """
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = (globals.output_video_quality + 1) * 51 // 100
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i',
                os.path.join(temp_directory_path, '%04d.' + globals.temp_frame_format), '-c:v',
                globals.output_video_encoder]
    if globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    return run_ffmpeg(commands)


def restore_audio(target_path: str, output_path: str) -> None:
    """
    将原始音频恢复到处理后的视频中
    
    Args:
        target_path (str): 原始视频路径
        output_path (str): 输出视频路径
    """
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        ['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y',
         output_path])
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    """
    获取临时帧图像文件路径列表
    
    Args:
        target_path (str): 目标文件路径
        
    Returns:
        List[str]: 帧图像文件路径列表
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + globals.temp_frame_format)))


def get_temp_directory_path(target_path: str) -> str:
    """
    获取临时目录路径
    
    Args:
        target_path (str): 目标文件路径
        
    Returns:
        str: 临时目录路径
    """
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    """
    获取临时输出文件路径
    
    Args:
        target_path (str): 目标文件路径
        
    Returns:
        str: 临时输出文件路径
    """
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)

def create_temp(target_path: str) -> None:
    """
    创建临时目录
    
    Args:
        target_path (str): 目标文件路径
    """
    temp_directory_path = get_temp_directory_path(target_path)
    print(f'创建临时目录......{temp_directory_path}')
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    """
    将临时文件移动到输出路径
    
    Args:
        target_path (str): 目标文件路径
        output_path (str): 输出文件路径
    """
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    """
    清理临时文件和目录
    
    Args:
        target_path (str): 目标文件路径
    """
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    """
    检查文件路径是否具有图像扩展名
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        bool: 如果是图像文件扩展名返回True，否则返回False
    """
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    """
    检查文件是否为图像文件
    
    Args:
        image_path (str): 文件路径
        
    Returns:
        bool: 如果是图像文件返回True，否则返回False
    """
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    """
    检查文件是否为视频文件
    
    Args:
        video_path (str): 文件路径
        
    Returns:
        bool: 如果是视频文件返回True，否则返回False
    """
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    """
    条件下载文件，如果文件不存在则下载
    
    Args:
        download_directory_path (str): 下载目录路径
        urls (List[str]): 要下载的文件URL列表
    """
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def inswapper_128_pre_check() -> bool:
    """
    预检查函数，确保面部交换模型存在，如果不存在则下载

    Returns:
        bool: 总是返回True（下载失败会抛出异常）
    """
    download_directory_path = 'models'
    # 下载inswapper_128.onnx模型文件
    conditional_download(download_directory_path,
                         ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def gfpgan_pre_check() -> bool:
    """
    预检查函数，确保面部增强模型存在，如果不存在则下载

    Returns:
        bool: 总是返回True（下载失败会抛出异常）
    """
    download_directory_path = 'models'
    # 下载GFPGANv1.4.pth模型文件
    conditional_download(download_directory_path,
                         ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True
