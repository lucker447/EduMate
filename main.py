"""EduMate 入口：识别一张试卷图片并打印结果。

调用链：图片 → core.service.recognize_exam → core.agent.chat_with_image
        → core.prompt.EXAM_OCR_PROMPT → core.service.normalize 归一化 → 输出
"""

from edu_mate.core.service import recognize_exam

IMG_PATH = "img.png"

if __name__ == "__main__":
    print(recognize_exam(IMG_PATH))
