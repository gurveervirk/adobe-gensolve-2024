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

## Curve Completion Examples
<table align="center">
  <tr>
    <td align="center">
      <h4>frag2</h4>
      <img width="200" alt="sample_output" src="https://res.cloudinary.com/utubee/image/upload/v1723400446/jtjatbdye47ts9shenvy.png">
    </td>
    <td align="center">
      <h4>isolated</h4>
      <img width="200" alt="sample_output" src="https://res.cloudinary.com/utubee/image/upload/v1723400446/psyqd4e0xiqugehnagwm.png">
    </td>
  </tr>
  <tr>
    <td align="center">
      <h4>occlusion1</h4>
      <img width="200" alt="sample_output" src="https://res.cloudinary.com/utubee/image/upload/v1723400446/fq1nlkbfcksdfgwpctku.png">
    </td>
    <td align="center">
      <h4>occlusion2</h4>
      <img width="200" alt="sample_output" src="https://res.cloudinary.com/utubee/image/upload/v1723400446/joarwkgd4cyckb6qjpzz.png">
    </td>
  </tr>
</table>




## Miscellaneous

- [Demo](https://youtu.be/YcmWPHTnhBQ)
- [Demo for Curve Completion](https://drive.google.com/file/d/1_V41Bb5XKwe1rqgN81oTptldy6xEB3JK/view)
- [Hosted Streamlit App](https://curvetopia-adobe.streamlit.app/)
