import cv2
import face_recognition
import numpy as np
import os
import time
from functools import reduce
from typing import Tuple, List, Callable
from itertools import takewhile as _takewhile

def safe_load_image(image_path: str) -> List[np.ndarray]:
    return (
        face_recognition.face_encodings(
            face_recognition.load_image_file(image_path)
        )
        if os.path.exists(image_path) 
        else []
    )

def process_known_faces_directory(directory: str) -> Tuple[Tuple[np.ndarray, ...], Tuple[str, ...]]:
    return tuple(
        reduce(
            lambda acc, _: (
                acc[0] + (safe_load_image(os.path.join(directory, _))[0],) 
                if safe_load_image(os.path.join(directory, _))[0] is not None 
                else acc[0],
                acc[1] + (os.path.splitext(_)[0],) 
                if safe_load_image(os.path.join(directory, _)) 
                else acc[1]
            ),
            filter(
                lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg')), 
                os.listdir(directory)
            ),
            (tuple(), tuple())
        )
    )

def recognize_face(face_encoding: np.ndarray, known_encodings: Tuple[np.ndarray, ...], known_names: Tuple[str, ...]) -> str:
    return reduce(
        lambda best, current: (
            current if face_recognition.face_distance([current[1]], face_encoding)[0] < best[0] 
            else best
        ),
        map(
            lambda known_encoding: (
                face_recognition.face_distance([known_encoding], face_encoding)[0], 
                known_encoding,
                known_names[known_encodings.index(known_encoding)]
            ),
            known_encodings
        ),
        (1.0, None, "Desconocido")
    )[2]

def process_frame(frame: np.ndarray, known_encodings: Tuple[np.ndarray, ...], known_names: Tuple[str, ...]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    return (
        list(
            map(
                lambda face_data: (
                    recognize_face(face_data[0], known_encodings, known_names), 
                    face_data[1]
                ),
                zip(
                    face_recognition.face_encodings(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        face_recognition.face_locations(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                    ),
                    face_recognition.face_locations(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                )
            )
        )
        if frame is not None and frame.size > 0
        else []
    )

def draw_faces(frame: np.ndarray, recognized_faces: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
    list(map(lambda face_data: cv2.rectangle(frame, (face_data[1][3], face_data[1][0]), 
                                              (face_data[1][1], face_data[1][2]), (255, 0, 0), 2), recognized_faces))
    list(map(lambda face_data: cv2.putText(frame, face_data[0], (face_data[1][3], face_data[1][0] - 10), 
                                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 1), recognized_faces))
    
    return frame

def takewhile(pred, iterable):
    return _takewhile(pred, iterable)

def video_capture_stream(interval: int, camera_index: int) -> Callable[[], None]:
    def capture_and_process():
        known_data = process_known_faces_directory('known_faces')
        video_capture = cv2.VideoCapture(camera_index)

        def process_video_stream():
            last_time = time.time()
            persistent_recognized_faces = []
            
            while (ret := video_capture.read())[0]:
                frame = ret[1]
                current_time = time.time()

                # Update recognized faces only at the specified interval
                if current_time - last_time >= interval:
                    persistent_recognized_faces = process_frame(frame, known_data[0], known_data[1])
                    last_time = current_time

                # Always draw persistent faces
                annotated_frame = draw_faces(frame, persistent_recognized_faces)
                cv2.putText(annotated_frame, f"Intervalo: {interval}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Reconocimiento Facial', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            video_capture.release()
            cv2.destroyAllWindows()

        if video_capture.isOpened():
            process_video_stream()
        else:
            print("Error: No se pudo abrir la camara")
        
    return capture_and_process

stream_func = video_capture_stream(2, 0)
stream_func()  

if __name__ == "_main_":
    video_capture_stream(2, 0)()