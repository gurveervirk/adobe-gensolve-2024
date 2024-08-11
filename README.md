# Curvetopia

<div align="center">
  <img width="640" alt="sample_output" src="https://github.com/gurveervirk/adobe-gensolve-2024/blob/main/misc-outputs/curvetopia.png">
</div>

## Description

This project aims to take as input hand-drawn shapes and:

1. regularize the shape, if possible
2. check for symmetry in the image, and return a line of symmetry if it does
3. complete incomplete shapes using the above

## Installation

To install the project, follow these steps:

1. Clone the repository.
2. Run `pip install -r requirements.txt` to install the dependencies.

## Usage

To use the project, follow these steps:

1. Use `detect_shapes.py` to regularize the shapes if possible
2. Use `find_reflection_symmetry.py` to find a line of symmetry, if shape is symmetric
3. Use `curve_completion.py` to complete an input curve, which makes use of the above

## Miscellaneous

- [Demo](https://youtu.be/YcmWPHTnhBQ)
- [Hosted Streamlit App](https://curvetopia-adobe.streamlit.app/)