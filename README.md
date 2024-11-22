# Real-Time Facial Recognition Using Functional Programming in Python

## Overview

This project is a real-time facial recognition system, built entirely in Python, employing a purely functional programming paradigm. The application makes use of the `OpenCV` and `face_recognition` libraries to process video frames, detect faces, and recognize individuals without using mutable variables, loops, or traditional control structures. Instead, it leverages functional programming constructs like `map`, `filter`, `reduce`, and recursion to achieve all operations.

The goal is to provide a concise and clear implementation of facial recognition, demonstrating the effectiveness of functional programming in real-world applications. The project is capable of recognizing pre-trained faces in a live video stream and provides relevant annotations in real time.

## Features
- Real-time facial detection and recognition.
- Uses a purely functional approach with Python.
- Works with pre-trained facial encodings to recognize known individuals.
- Displays the recognized names and facial boundary boxes on the screen.
- Error-handling for missing or inaccessible images.

## Technologies Used
- **Python 3.9**
- **OpenCV** for video capture and rendering.
- **face_recognition** library for facial detection and encoding.
- **NumPy** for efficient numerical computations.
- Functional programming constructs: `map`, `filter`, `reduce`, and recursion.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Install the following dependencies:
  ```sh
  pip install opencv-python face_recognition numpy
  ```

### Running the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/facial-recognition-functional.git
   ```
2. Navigate to the project directory:
   ```sh
   cd facial-recognition-functional
   ```
3. Run the main script:
   ```sh
   python main.py
   ```
4. The application will open a video stream from your default camera and start recognizing faces.

### Usage
- Make sure you have a directory named `known_faces` in the project folder, containing images of the people you want to recognize.
- Each image should be named with the person's name (e.g., `john_doe.jpg`).
- Run the application and it will recognize faces from the video stream and label them accordingly.
- Press `q` to exit the video stream.

## Functional Programming Approach
This project uses functional programming to ensure the use of:
- **Immutable data**: Avoiding mutable state throughout the program.
- **Pure functions**: Functions without side effects.
- **Higher-order functions**: Making extensive use of functions like `map`, `filter`, and `reduce`.
- **Recursion**: Instead of traditional `for` or `while` loops.

## Project Structure
- **`main.py`**: The main script that runs the facial recognition process.
- **`utils.py`**: Contains helper functions for processing images, frames, and facial encodings.
- **`known_faces/`**: A folder containing images of people to recognize.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for any improvements or additional features.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments
- Thanks to the developers of the `face_recognition` library for providing a powerful tool for face detection and recognition.
- Inspired by the challenge of implementing practical applications using a purely functional approach in Python.

