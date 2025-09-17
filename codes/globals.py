from typing import List, Optional

# 源文件路径（源人脸图片/视频）
source_path: Optional[str] = None
# 目标文件路径（目标人脸图片/视频）
target_path: Optional[str] = None
# 输出文件路径
output_path: Optional[str] = None
# 帧处理器列表（用于处理视频帧的模块）
frame_processors: List[str] = []
# 是否保持原视频的帧率
keep_fps: Optional[bool] = None
# 是否保留临时帧文件
keep_frames: Optional[bool] = None
# 是否跳过音频处理
skip_audio: Optional[bool] = None
# 是否处理多个人脸（True表示处理画面中所有检测到的人脸）
many_faces: Optional[bool] = None
# 参考人脸在画面中的位置索引
reference_face_position: Optional[int] = None
# 参考帧编号（在视频中用作参考的那一帧）
reference_frame_number: Optional[int] = None
# 相似人脸距离阈值（用于判断两张人脸是否相似，距离小于该值认为是同一个人）
similar_face_distance: Optional[float] = None
# 临时帧文件格式（如jpg、png等）
temp_frame_format: Optional[str] = None
# 临时帧文件质量（压缩质量）
temp_frame_quality: Optional[int] = None
# 输出视频编码器（如libx264等）
output_video_encoder: Optional[str] = None
# 输出视频质量
output_video_quality: Optional[int] = None
# 最大内存使用限制（MB）
max_memory: Optional[int] = None
# 执行提供者列表（如CPUExecutionProvider、CUDAExecutionProvider等）
execution_providers: List[str] = []
# 执行线程数
execution_threads: Optional[int] = None
# ffmpeg日志级别（默认为info级别）
log_level: str = 'info'