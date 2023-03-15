import cv2
import numpy.typing as npt
from collections.abc import Generator

def read_video(path: str = "los_angeles.mp4") -> Generator[npt.NDArray]:
    """Read video from path 

    Args:
        path (str): path to video to read

    Yields:
        Generator[npt.NDArray]: Frame that has been read
    """
    video_cap = cv2.VideoCapture(path)
    sucess: bool
    frame: npt.NDArray
    sucess, frame= video_cap.read()
    while sucess:
        yield frame
        sucess, frame = video_cap.read()
    video_cap.release()
    cv2.destroyAllWindows()

