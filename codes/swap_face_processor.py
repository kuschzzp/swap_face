# 下面这个引入必须要有！！！！import codes.fix_torchvision
import codes.fix_torchvision
import os
import shutil
import sys
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from codes.models import process_image, clear_face_swapper, clear_face_enhancer, process_video

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Python 3.12: 兼容 collections.Iterable 移除问题
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import codes.globals as globals
from utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video,
    extract_frames, get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp
)
from face_reference import clear_face_reference


def run_processor(
        source_path: str,
        target_path: str,
        output_path: str,
        frame_processors: List[str] = None,
        execution_provider: str = 'cpu',
        keep_fps: bool = True,
        many_faces: bool = False,
        skip_audio: bool = False,
        reference_face_position: int = 0,
        reference_frame_number: int = 0,
        similar_face_distance: float = 0.85,
        max_frame_threads: int = 4
) -> bool:
    """
    控制整个换脸流程，包括图像/视频处理、帧提取、音频恢复等步骤

    Args:
        source_path (str): 源图像路径（提供要换入的面部）
        target_path (str): 目标图像/视频路径（提供要被换脸的面部）
        output_path (str): 输出路径
        frame_processors (List[str]): 帧处理器列表，默认为['face_swapper']
        execution_provider (str): 执行提供者，默认为'cpu'
        keep_fps (bool): 是否保持原视频帧率，默认为True
        many_faces (bool): 是否处理多个人脸，默认为False
        skip_audio (bool): 是否跳过音频处理，默认为False
        reference_face_position (int): 参考面部位置索引，默认为0
        reference_frame_number (int): 参考帧编号，默认为0
        similar_face_distance (float): 相似面部距离阈值，默认为0.85
        max_frame_threads (int): 视频帧处理的最大线程数，默认为4
        
    Returns:
        bool: 处理成功返回True，失败返回False
    """

    # 设置全局变量
    globals.source_path = source_path
    globals.target_path = target_path
    globals.output_path = output_path
    globals.frame_processors = frame_processors or ['face_swapper']
    # 处理执行提供者名称（兼容不同格式）
    globals.execution_providers = [execution_provider] if not execution_provider.endswith('ExecutionProvider') else [
        execution_provider]
    globals.keep_fps = keep_fps
    globals.many_faces = many_faces
    globals.skip_audio = skip_audio
    globals.reference_face_position = reference_face_position
    globals.reference_frame_number = reference_frame_number
    # 设置临时帧参数
    globals.temp_frame_format = 'png'
    globals.temp_frame_quality = 0
    # 设置输出视频参数
    globals.output_video_encoder = 'libx264'
    globals.output_video_quality = 35
    globals.similar_face_distance = similar_face_distance

    # 确保模型目录存在
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    try:
        # 处理图像
        if has_image_extension(target_path):
            # 复制目标图像到输出路径作为基础
            shutil.copy2(target_path, output_path)
            # 处理图像
            process_image(source_path, output_path, output_path, globals.frame_processors)
            if is_image(output_path):
                print('处理图像成功！')
                return True
            else:
                print('图像处理失败！')
                return False

        # 处理视频
        print('创建临时视频处理资源......')
        # 创建临时目录
        create_temp(target_path)

        # 提取帧图像
        if keep_fps:
            fps = detect_fps(target_path)
            print(f'正在以 {fps} FPS 提取帧...')
            extract_frames(target_path, fps)
        else:
            print('正在以 30 FPS 提取帧...')
            extract_frames(target_path)

        # 获取帧图像路径列表
        temp_frame_paths = get_temp_frame_paths(target_path)
        if temp_frame_paths:
            print('正在处理帧...')
            # 处理视频帧（支持多线程）
            process_video(source_path, temp_frame_paths, globals.frame_processors, max_frame_threads)
        else:
            print('未找到帧...')
            return False

        # 合成视频
        if keep_fps:
            fps = detect_fps(target_path)
            print(f'正在以 {fps} FPS 创建视频...')
            create_video(target_path, fps)
        else:
            print('正在以 30 FPS 创建视频...')
            create_video(target_path)

        # 恢复音频
        if skip_audio:
            # 跳过音频，直接移动临时文件
            move_temp(target_path, output_path)
            print('正在跳过音频...')
        else:
            if keep_fps:
                print('正在恢复音频...')
            else:
                print('正在恢复音频可能会导致问题，因为未保持帧率...')
            # 恢复原始音频到处理后的视频
            restore_audio(target_path, output_path)

        print('正在清理临时资源...')
        # 清理临时文件
        clean_temp(target_path)

        if is_video(output_path):
            print('视频处理成功！')
            return True
        else:
            print('视频处理失败！')
            return False

    except Exception as e:
        print(f'处理失败，错误信息：{str(e)}')
        return False
    finally:
        # 清理资源
        clear_face_swapper()
        clear_face_enhancer()
        clear_face_reference()


def batch_run_processor(
        source_path: str,
        target_path: str,
        output_path: str,
        frame_processors: List[str] = None,
        execution_provider: str = 'cpu',
        keep_fps: bool = True,
        many_faces: bool = False,
        skip_audio: bool = False,
        reference_face_position: int = 0,
        reference_frame_number: int = 0,
        similar_face_distance: float = 0.85,
        max_threads: int = 4,
        max_frame_threads: int = 4
) -> bool:
    """
    批量处理文件夹中的图像和视频文件
    
    Args:
        source_path (str): 源图像路径（提供要换入的面部）
        target_path (str): 目标文件夹路径（包含要被换脸的图像/视频）
        output_path (str): 输出文件夹路径
        frame_processors (List[str]): 帧处理器列表，默认为['face_swapper']
        execution_provider (str): 执行提供者，默认为'cpu'
        keep_fps (bool): 是否保持原视频帧率，默认为True
        many_faces (bool): 是否处理多个人脸，默认为False
        skip_audio (bool): 是否跳过音频处理，默认为False
        reference_face_position (int): 参考面部位置索引，默认为0
        reference_frame_number (int): 参考帧编号，默认为0
        similar_face_distance (float): 相似面部距离阈值，默认为0.85
        max_threads (int): 最大线程数，默认为4
        max_frame_threads (int): 视频帧处理的最大线程数，默认为4
        
    Returns:
        bool: 所有文件处理成功返回True，有失败则返回False
    """
    # 检查源文件是否存在
    if not os.path.exists(source_path):
        print(f'源文件不存在: {source_path}')
        return False

    # 检查目标路径是否存在且为目录
    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        print(f'目标路径不存在或不是目录: {target_path}')
        return False

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 收集所有待处理的文件
    target_files = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            file_path = os.path.join(root, file)
            if has_image_extension(file_path) or is_video(file_path):
                # 计算相对于target_path的相对路径
                relative_path = os.path.relpath(file_path, target_path)
                target_files.append((file_path, relative_path))

    if not target_files:
        print('目标文件夹中未找到图像或视频文件')
        return False

    print(f'找到 {len(target_files)} 个待处理文件')

    # 处理每个文件
    success_count = 0
    failed_files = []

    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 提交所有任务
        future_to_file = {}
        for target_file, relative_path in target_files:
            # 构造输出文件路径
            output_file_path = os.path.join(output_path, relative_path)
            # 确保输出文件的目录存在
            output_file_dir = os.path.dirname(output_file_path)
            os.makedirs(output_file_dir, exist_ok=True)

            # 提交任务
            future = executor.submit(
                run_processor,
                source_path,
                target_file,
                output_file_path,
                frame_processors,
                execution_provider,
                keep_fps,
                many_faces,
                skip_audio,
                reference_face_position,
                reference_frame_number,
                similar_face_distance,
                max_frame_threads
            )
            future_to_file[future] = (target_file, output_file_path)

        # 收集结果
        for future in as_completed(future_to_file):
            target_file, output_file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    print(f'处理成功: {target_file}')
                else:
                    failed_files.append(target_file)
                    print(f'处理失败: {target_file}')
            except Exception as e:
                failed_files.append(target_file)
                print(f'处理异常 {target_file}: {str(e)}')

    print(f'批量处理完成: 成功 {success_count}/{len(target_files)} 个文件')
    if failed_files:
        print(f'失败文件列表:')
        for file in failed_files:
            print(f'  - {file}')
        return False

    return True
