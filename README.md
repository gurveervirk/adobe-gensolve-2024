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

The `main` part of most scripts are commented for streamlit deployment. Kindly go through the comments before uncommenting and using them.

### Curve Completion

The `curve_completion.py` script is designed to complete incomplete curves by fitting shapes to connected polylines. Kindly check the video under Miscellaneous to use this code.

To run:

bash
python scripts/curve_completion.py


### Curve Extrapolation

The `curve_extrapolation.py` script extrapolates missing parts of a curve using interpolation.

To run:

bash
python scripts/curve_extrapolation.py


### Shape Detection

The `detect_shapes.py` script detects and regularizes different shapes (e.g., lines, ellipses, polygons) within a set of points.

To run:

bash
python scripts/detect_shapes.py


### Reflection Symmetry Detection

The `find_reflection_symmetry_line.py` script finds the line of reflection symmetry for a given set of points.

To run:

bash
python scripts/find_reflection_symmetry_line.py


### Polyline Splitting and Merging

The `split_disjoint.py` script splits polylines into disjoint segments and attempts to merge and extend them based on angle and proximity criteria.

To run:

bash
python scripts/split_disjoint.py

## Curve Completion Examples
<table align="center">
  <tr>
    <td align="center">
      <h4>frag2</h4>
      <img width="200" alt="sample_output" src="https://github.com/gurveervirk/adobe-gensolve-2024/blob/main/misc-outputs/occlusion1_completed.png"
    </td>
    <td align="center">
      <h4>isolated</h4>
      <img width="200" alt="sample_output" src="https://github.com/gurveervirk/adobe-gensolve-2024/blob/main/misc-outputs/occlusion2_completed.png"
    </td>
  </tr>
  <tr>
    <td align="center">
      <h4>occlusion1</h4>
      <img width="200" alt="sample_output" src="https://github.com/gurveervirk/adobe-gensolve-2024/blob/main/misc-outputs/frag2_completed.png"
    </td>
    <td align="center">
      <h4>occlusion2</h4>
      <img width="200" alt="sample_output" src="https://github.com/gurveervirk/adobe-gensolve-2024/blob/main/misc-outputs/isolated.png"
    </td>
  </tr>
</table>

## Miscellaneous

- [Demo](https://youtu.be/YcmWPHTnhBQ)
- [Demo for Curve Completion](https://drive.google.com/file/d/1_V41Bb5XKwe1rqgN81oTptldy6xEB3JK/view)
- [Hosted Streamlit App](https://curvetopia-adobe.streamlit.app/)
