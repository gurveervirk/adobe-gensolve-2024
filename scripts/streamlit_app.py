import streamlit as st
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
from helper import read_csv
from detect_shapes import fit_shape, plot
from curve_completion import complete_curves
from split_disjoint import split_polylines_to_disjoint, extend_and_connect_polylines
import os
import time  # For simulating processing time
from io import BytesIO

def path_to_points(path):
    points = []
    current_point = None

    for command in path:
        cmd = command[0]
        if cmd == 'M':
            current_point = (command[1], command[2])
            points.append(current_point)
        elif cmd == 'L':
            current_point = (command[1], command[2])
            points.append(current_point)
        elif cmd == 'Q':
            control_point = (command[1], command[2])
            end_point = (command[3], command[4])
            if current_point:
                points.append(current_point)
            points.append(control_point)
            points.append(end_point)
            current_point = end_point

    return np.array(points)

def main():
    st.title("Curvetopia")

    # Create tabs
    tab1, tab2 = st.tabs(["Draw Shape", "Upload CSV"])

    canvas_height = 600
    canvas_width = 600

    # Initialize canvas state
    if "background_image" not in st.session_state:
        st.session_state["background_image"] = None
    if "current_canvas" not in st.session_state:
        st.session_state["current_canvas"] = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background

    # Tab 1: Draw Shape
    with tab1:
        st.header("Draw a Shape")
        st.write("Use the canvas on the left to draw shapes. The processed result will appear on the right.")

        # Create columns for layout
        col1, col2 = st.columns([0.7, 0.3], gap='medium')  # Adjust column width ratios as needed

        with col1:
            # Create a canvas component for drawing shapes
            canvas_result = st_canvas(
                stroke_width=2,
                stroke_color="#000",
                background_image=st.session_state["background_image"],
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                key="canvas",
            )

        with col2:
            # Create a placeholder for the output canvas
            st.write("**Output**")
            canvas_placeholder = st.empty()

            # Display the updated canvas
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                latest_shape = canvas_result.json_data["objects"][-1]  # Get the latest drawn shape

                if "path" in latest_shape:
                    # Convert path to points
                    points = path_to_points(latest_shape["path"])

                    # Pass points to fit_shape (replace True with False if needed)
                    best_points, _, _, symmetry_lines = fit_shape(points, True)

                    # Clear the canvas placeholder
                    canvas_placeholder.empty()

                    # Create new canvas with the original and updated shapes
                    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255 if st.session_state['background_image'] is None else st.session_state['current_canvas']
                    
                    # Draw original shape
                    for obj in canvas_result.json_data["objects"]:
                        if "path" in obj:
                            path_points = path_to_points(obj["path"])
                            for point in path_points:
                                cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

                    # Draw updated shape
                    for i in range(len(best_points)-1):
                        cv2.line(canvas, (int(best_points[i][0]), int(best_points[i][1])), (int(best_points[i+1][0]), int(best_points[i+1][1])), (0, 255, 0), 2)

                    # Draw symmetry lines
                    for symmetry_line in symmetry_lines:
                        slope, intercept = symmetry_line
                        if np.isinf(slope):
                            x = intercept
                            cv2.line(canvas, (int(x), 0), (int(x), canvas_height), (0, 0, 255), 1)
                        elif slope == 0:
                            y = intercept
                            cv2.line(canvas, (0, int(y)), (canvas_width, int(y)), (0, 0, 255), 1)
                        else:
                            y1 = 0
                            x1 = (y1 - intercept) / slope
                            y2 = canvas_height
                            x2 = (y2 - intercept) / slope
                            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                    
                    # Update the session state with the new image
                    st.session_state["background_image"] = Image.fromarray(canvas)
                    st.session_state["current_canvas"] = np.array(canvas)

                    # Render the updated canvas
                    canvas_placeholder.image(st.session_state["background_image"], use_column_width=True)
                    img_bytes = BytesIO()
                    st.session_state["background_image"].save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()

                    st.download_button(
                        label="Download Image",
                        data=img_bytes,
                        file_name="drawn_shape.png",
                        mime="image/png",
                        key='1'
                    )

                else:
                    st.write("No valid path data found. Please draw a shape on the canvas.")
            else:
                st.write("Please draw a shape on the canvas.")
            
    # Tab 2: Upload CSV
    with tab2:
        st.header("Upload CSV File")
        st.write("Upload a CSV file containing polylines to process and generate curves.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            st.write("Processing file...")

            # Add a progress bar
            progress_bar = st.progress(0)

            # Simulate processing time
            time.sleep(1)
            progress_bar.progress(20)
            
            # Pass the DataFrame to read_csv (replace with your function)
            polylines = read_csv(uploaded_file)

            # Split into disjoint polylines
            disjoint_polylines = split_polylines_to_disjoint(polylines)
            progress_bar.progress(40)

            # Connect disjoint polylines naturally
            connected_polylines = extend_and_connect_polylines(disjoint_polylines)
            progress_bar.progress(60)

            result = complete_curves(connected_polylines)
            progress_bar.progress(80)
            for i, r in enumerate(result):
                print(f"Completed curve {i}: {r[0]}")

            completed_polylines = [r[2] for r in result]
            names = [r[3] for r in result]
            best_symmetry_lines = [r[4] for r in result]

            path = "completed_shape.png"

            # Preferably use plot for complex plots
            plot([completed_polylines], path, names, best_symmetry_lines)

            final_path = os.path.join('misc-outputs', path)

            # Update progress bar to 100%
            progress_bar.progress(100)
            
            # Display the plotted image
            if os.path.exists(final_path):
                st.image(final_path, caption="Completed Polylines", use_column_width=True)
                
                # Convert image to a byte stream for download
                with open(final_path, "rb") as file:
                    img_bytes = file.read()

                st.download_button(
                    label="Download Image",
                    data=img_bytes,
                    file_name=path,
                    mime="image/png",
                    key='2'
                )
            else:
                st.write("Plot image not found.")

if __name__ == "__main__":
    main()
