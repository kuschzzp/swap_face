from codes.swap_face_processor import run_processor, batch_run_processor


def main():
    """
    主函数，演示如何使用run_processor函数
    """
    success = run_processor(
        source_path='./source.jpg',
        target_path='./target.jpg',
        output_path='./target_sssssss.jpg',
        frame_processors=['face_swapper', 'face_enhancer'],
        execution_provider='CPUExecutionProvider',
        keep_fps=True,
        many_faces=True,
        skip_audio=False,
        reference_face_position=0,
        similar_face_distance=0.85,
        max_frame_threads=4  # 视频帧处理的最大线程数
    )

    if success:
        print("处理完成!")
    else:
        print("处理失败!")


def batch_main():
    """
    批量处理主函数，演示如何使用batch_run_processor函数
    """
    success = batch_run_processor(
        source_path='./source.jpg',
        target_path='./testFiles',     # 包含待处理图像/视频的文件夹
        output_path='./testFilesOutputs',    # 输出文件夹
        frame_processors=['face_swapper', 'face_enhancer'],
        execution_provider='CPUExecutionProvider',
        keep_fps=True,
        many_faces=True,
        skip_audio=False,
        reference_face_position=0,
        similar_face_distance=0.85,
        max_threads=4,                    # 批量处理的最大并发线程数
        max_frame_threads=4               # 视频帧处理的最大线程数
    )

    if success:
        print("批量处理完成!")
    else:
        print("批量处理失败!")


if __name__ == '__main__':
    # 运行单个文件处理示例
    # main()
    
    # 运行批量处理示例
    batch_main()