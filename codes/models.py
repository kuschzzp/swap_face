import os
import threading
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import insightface
import codes.globals as globals
from codes.utilities import gfpgan_pre_check, inswapper_128_pre_check
from codes.face_analyser import get_one_face, get_many_faces, find_similar_face
from codes.face_reference import get_face_reference, set_face_reference

# THREAD_LOCK: 线程锁，用于确保模型实例的线程安全初始化
THREAD_LOCK = threading.Lock()
# THREAD_SEMAPHORE: 线程信号量，用于限制同时进行的面部增强操作数量
THREAD_SEMAPHORE = threading.Semaphore()
# FACE_SWAPPER: 面部交换模型实例
FACE_SWAPPER = None
# FACE_ENHANCER: 面部增强模型实例(GFPGAN)
FACE_ENHANCER = None


def get_face_swapper() -> Any:
    """
    获取面部交换器模型实例

    使用insightface库加载inswapper_128.onnx模型，该模型用于执行面部交换操作。
    通过线程锁确保在多线程环境中只初始化一次。

    Returns:
        insightface.model_zoo.FaceModel: 面部交换模型实例
    """
    global FACE_SWAPPER

    if not inswapper_128_pre_check():
        raise "inswapper_128.onnx 模型不存在"

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # 构建模型文件路径
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/inswapper_128.onnx')
            # 加载模型，使用全局定义的执行提供者（如CPU、GPU等）
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=globals.execution_providers)
    return FACE_SWAPPER


def get_device() -> str:
    """
    获取当前使用的计算设备类型

    根据全局设置的execution_providers判断使用的设备：
    - 如果使用CUDAExecutionProvider，返回'cuda'
    - 如果使用CoreMLExecutionProvider，返回'mps'
    - 否则返回'cpu'

    Returns:
        str: 设备类型字符串('cuda', 'mps'或'cpu')
    """
    if 'CUDAExecutionProvider' in globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_swapper() -> None:
    """
    清除面部交换器实例，释放资源
    """
    global FACE_SWAPPER
    if FACE_SWAPPER is not None:
        # 如果模型对象有清理方法，调用它
        if hasattr(FACE_SWAPPER, 'destroy'):
            FACE_SWAPPER.destroy()
        # 或者如果有会话对象，关闭会话
        if hasattr(FACE_SWAPPER, 'session'):
            FACE_SWAPPER.session = None
    FACE_SWAPPER = None


def clear_face_enhancer() -> None:
    """
    清除面部增强器实例，释放资源
    """
    global FACE_ENHANCER
    if FACE_ENHANCER is not None:
        # 如果模型对象有清理方法，调用它
        if hasattr(FACE_ENHANCER, 'destroy'):
            FACE_ENHANCER.destroy()
        # 或者如果有会话对象，关闭会话
        if hasattr(FACE_ENHANCER, 'session'):
            FACE_ENHANCER.session = None
    FACE_ENHANCER = None


def swap_face(source_face: Any, target_face: Any, temp_frame: Any) -> Any:
    """
    执行面部交换操作

    使用inswapper模型将源面部替换到目标面部位置

    Args:
        source_face (Any): 源面部信息
        target_face (Any): 目标面部信息
        temp_frame (Any): 包含面部的图像帧

    Returns:
        Any: 完成面部交换后的图像帧
    """
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def enhance_face(target_face: Any, temp_frame: Any) -> Any:
    """
    对面部进行增强处理

    使用GFPGAN模型对检测到的面部区域进行修复和增强

    Args:
        target_face (Any): 目标面部信息
        temp_frame (Any): 包含面部的图像帧

    Returns:
        Any: 完成面部增强后的图像帧
    """
    face_enhancer = get_face_enhancer()
    if face_enhancer is None:
        return temp_frame

    # 获取面部边界框坐标
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    # 添加额外的边界框填充
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    # 提取面部区域
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        # 使用信号量限制并发执行
        with THREAD_SEMAPHORE:
            _, _, temp_face = face_enhancer.enhance(temp_face, paste_back=True)
        # 将增强后的面部区域放回原图
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame


def process_frame(source_face: Any, reference_face: Any, temp_frame: Any, processors: List[str]) -> Any:
    """
    处理单帧图像

    根据指定的处理器列表对图像帧进行面部交换和/或面部增强处理

    Args:
        source_face (Any): 源面部信息
        reference_face (Any): 参考面部信息
        temp_frame (Any): 要处理的图像帧
        processors (List[str]): 处理器列表(['face_swapper', 'face_enhancer'])

    Returns:
        Any: 处理后的图像帧
    """
    # 面部交换处理
    if 'face_swapper' in processors:
        if globals.many_faces:
            # 处理多个人脸
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            # 只处理与参考面部相似的人脸
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)

    # 面部增强处理
    if 'face_enhancer' in processors:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)

    return temp_frame


def process_single_frame(frame_args) -> bool:
    """
    处理单个帧图像
    
    Args:
        frame_args (tuple): 包含(source_face, reference_face, temp_frame_path, processors)的元组
        
    Returns:
        bool: 处理成功返回True，失败返回False
    """
    source_face, reference_face, temp_frame_path, processors = frame_args
    try:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame, processors)
        cv2.imwrite(temp_frame_path, result)
        return True
    except Exception as e:
        print(f'处理帧 {temp_frame_path} 时出错: {str(e)}')
        return False


def process_frames(source_path: str, temp_frame_paths: List[str], processors: List[str], max_threads: int = 4) -> None:
    """
    处理多个帧图像文件（支持多线程）

    Args:
        source_path (str): 源图像路径
        temp_frame_paths (List[str]): 帧图像文件路径列表
        processors (List[str]): 处理器列表
        max_threads (int): 最大线程数，默认为4
    """
    # 获取源面部信息
    source_face = get_one_face(cv2.imread(source_path))
    # 获取参考面部信息（如果需要）
    reference_face = None if globals.many_faces else get_face_reference()

    # 使用线程池处理帧图像
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 准备任务参数
        frame_tasks = [
            (source_face, reference_face, temp_frame_path, processors) 
            for temp_frame_path in temp_frame_paths
        ]
        
        # 提交所有任务
        futures = [executor.submit(process_single_frame, task) for task in frame_tasks]
        
        # 等待所有任务完成并检查结果
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if not result:
                    print(f'处理帧失败: {temp_frame_paths[i]}')
            except Exception as e:
                print(f'处理帧异常 {temp_frame_paths[i]}: {str(e)}')


def process_image(source_path: str, target_path: str, output_path: str, processors: List[str]) -> None:
    """
    处理单张图像

    Args:
        source_path (str): 源图像路径
        target_path (str): 目标图像路径
        output_path (str): 输出图像路径
        processors (List[str]): 处理器列表
    """
    # 获取源面部信息
    source_face = get_one_face(cv2.imread(source_path))
    # 读取目标图像
    target_frame = cv2.imread(target_path)
    # 获取参考面部信息
    reference_face = None if globals.many_faces else get_one_face(
        target_frame, globals.reference_face_position
    )
    # 处理图像帧
    result = process_frame(source_face, reference_face, target_frame, processors)
    # 保存结果
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str], processors: List[str], max_threads: int = 4) -> None:
    """
    处理视频（通过处理所有帧图像）

    Args:
        source_path (str): 源图像路径
        temp_frame_paths (List[str]): 帧图像文件路径列表
        processors (List[str]): 处理器列表
        max_threads (int): 最大线程数，默认为4
    """
    # 如果是单人脸模式且还没有参考面部，则设置参考面部
    if 'face_swapper' in processors and not globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, globals.reference_face_position)
        set_face_reference(reference_face)
    # 处理所有帧图像
    process_frames(source_path, temp_frame_paths, processors, max_threads)


def get_face_enhancer() -> Any:
    """
    获取面部增强器(GFPGAN)模型实例

    GFPGAN是一个用于面部修复和增强的模型，可以提高换脸结果的质量。
    通过线程锁确保在多线程环境中只初始化一次。

    Returns:
        GFPGANer: 面部增强模型实例

    Raises:
        ImportError: 如果GFPGAN库未安装
        Exception: 其他初始化错误
    """
    global FACE_ENHANCER

    if not gfpgan_pre_check():
        raise 'GFPGAN未安装，面部增强功能不可用'

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            try:
                from gfpgan import GFPGANer
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/GFPGANv1.4.pth')
                # 初始化GFPGAN模型
                # FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
                FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1) # 设备源码内部会自动判断用不用GPU
            except ImportError as e:
                FACE_ENHANCER = None
                raise e
    return FACE_ENHANCER
