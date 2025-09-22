import threading
from typing import Any, Optional, List
import insightface
from insightface.app.common import Face
import numpy
import os

import codes.globals as globals

Frame = numpy.ndarray[Any, Any]

# 全局变量，用于存储人脸分析器实例
FACE_ANALYSER = None
# 线程锁，确保多线程环境下人脸分析器的初始化安全
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    """
    获取人脸分析器实例
    
    使用insightface库创建或返回已存在的人脸分析器实例。
    通过全局锁确保在多线程环境中只初始化一次。
    使用'buffalo_l'模型进行人脸检测和分析。
    
    Returns:
        insightface.app.FaceAnalysis: 人脸分析器实例
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # 指定模型根目录为项目根目录下的models文件夹
            model_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            # 使用buffalo_l模型和全局定义的执行提供者（如CPU、GPU等）
            # insightface会自动下载模型到指定的model_root目录
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', root=model_root, providers=globals.execution_providers)
            # 初始化分析器，ctx_id=0表示使用默认设备
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    """
    清除人脸分析器实例
    
    将全局的人脸分析器实例重置为None，释放资源。
    
    Returns:
        None
    """
    global FACE_ANALYSER

    if FACE_ANALYSER is not None:
        # 如果模型对象有清理方法，调用它
        if hasattr(FACE_ANALYSER, 'destroy'):
            FACE_ANALYSER.destroy()
        # 或者如果有会话对象，关闭会话
        if hasattr(FACE_ANALYSER, 'session'):
            FACE_ANALYSER.session = None

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    从图像帧中获取指定位置的人脸
    
    Args:
        frame (Frame): 包含人脸的图像帧
        position (int): 要获取的人脸位置索引，默认为0（第一张人脸）
    
    Returns:
        Optional[Face]: 指定位置的人脸对象，如果未检测到人脸则返回None
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            # 如果指定位置超出范围，则返回最后一张人脸
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    从图像帧中检测并获取所有的人脸
    
    Args:
        frame (Frame): 包含人脸的图像帧
    
    Returns:
        Optional[List[Face]]: 检测到的人脸列表，如果发生错误则返回None
    """
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    """
    在图像帧中查找与参考人脸相似的人脸
    
    通过计算人脸特征向量之间的欧氏距离来判断相似性。
    只有当距离小于全局设置的similar_face_distance阈值时才认为是相似人脸。
    
    Args:
        frame (Frame): 要搜索人脸的图像帧
        reference_face (Face): 作为参考的人脸对象
    
    Returns:
        Optional[Face]: 找到的相似人脸，未找到则返回None
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            # 确保两个人脸对象都有normed_embedding属性
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                # 计算两个特征向量之间的欧氏距离
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                # 如果距离小于阈值，则认为是相似的人脸
                if distance < globals.similar_face_distance:
                    return face
    return None