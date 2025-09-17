from typing import Optional

from insightface.app.common import Face

# 全局变量，用于存储参考人脸对象
# 在人脸交换过程中，这是目标需要替换成的人脸
FACE_REFERENCE = None


def get_face_reference() -> Optional[Face]:
    """
    获取当前设置的参考人脸
    
    Returns:
        Optional[Face]: 当前存储的参考人脸对象，如果未设置则返回None
    """
    return FACE_REFERENCE


def set_face_reference(face: Face) -> None:
    """
    设置参考人脸
    
    Args:
        face (Face): 要设置为参考的人脸对象
    """
    global FACE_REFERENCE

    FACE_REFERENCE = face


def clear_face_reference() -> None:
    """
    清除参考人脸
    
    将全局的参考人脸变量重置为None，释放相关资源
    """
    global FACE_REFERENCE

    FACE_REFERENCE = None
