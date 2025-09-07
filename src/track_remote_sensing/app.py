import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

st.title("Click Control Points")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original", use_container_width=True)

    # Capture clicks
    coords = streamlit_image_coordinates(img)

    if coords is not None:
        st.write(f"Clicked: {coords['x']}, {coords['y']}")
        # Store clicked points
        if "points" not in st.session_state:
            st.session_state.points = []
        st.session_state.points.append((coords["x"], coords["y"]))
        st.write("All points:", st.session_state.points)

    # Run SAM
    if st.button("Run SAM"):
        points = st.session_state.points
        st.success(f"Would run SAM with {points}")
