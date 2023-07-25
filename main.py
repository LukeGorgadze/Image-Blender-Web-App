import cv2 as cv
import numpy as np
import streamlit as st
import time

def blend(a, b, alpha, maxLen, maxWid, c):
    pxlA = []
    pxlB = []

    for row in range(maxLen):
        for column in range(maxWid):
            pxlA = a[row][column]
            pxlB = b[row][column]
            c[row][column] =  pxlA * alpha + pxlB * (1-alpha)
    return c

def main():
    st.title("Image blending with python")
    st.subheader("Author: Luka Gorgadze")

    # Add instruction text
    st.markdown("For best performance and result, choose images with the same aspect ratio and low resolution, e.g., 300x300 px.")

    Alpha = 0
    uploaded_file_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"])
    uploaded_file_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"])

    if uploaded_file_a and uploaded_file_b:
        a = cv.imdecode(np.frombuffer(uploaded_file_a.read(), np.uint8), 1)
        b = cv.imdecode(np.frombuffer(uploaded_file_b.read(), np.uint8), 1)

        # Convert images to RGB format
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
        b = cv.cvtColor(b, cv.COLOR_BGR2RGB)

        maxLen = min(a.shape[0], b.shape[0])
        maxWid = min(a.shape[1], b.shape[1]) 
        c = np.zeros((maxLen, maxWid, 3))

        cols = st.columns(3)
        cols[0].image(a, caption='Image A', use_column_width=True)
        cols[1].image(b, caption='Image B', use_column_width=True)

        st.markdown("Blending Images:")

        # Create an empty space to display the video frame
        result_placeholder = cols[2].empty()

        # Create a progress bar to indicate blending progress
        progress_bar = st.progress(0)

        while Alpha < 1:
            c = blend(a, b, Alpha, maxLen, maxWid, c)
            c_display = cv.cvtColor(c.astype(np.uint8), cv.COLOR_RGB2RGBA)

            # Display the blended video frame
            result_placeholder.image(c_display, caption='Blended Image', use_column_width=True)

            time.sleep(0.05)

            # Update progress bar
            progress_bar.progress(Alpha)

            Alpha += 0.1
            if Alpha > 1:
                Alpha = 0
    else:
        st.markdown("## Python Code:")
        with st.expander("Click to view the code"):
                st.code("""
                    import cv2 as cv
                    import numpy as np

                    def blend(a,b,alpha,maxLen,maxWid,c):
                        pxlA = []
                        pxlB = []

                        for row in range(maxLen):
                            for column in range(maxWid):
                                pxlA = a[row][column]
                                pxlB = b[row][column]
                                c[row][column] =  pxlA * alpha + pxlB * (1-alpha)
                        return c


                    Alpha = 0
                    a = cv.imread('image1.png')
                    b = cv.imread('image2.png')  
                    maxLen = min(a.shape[0],b.shape[0])
                    maxWid = min(a.shape[1],b.shape[1]) 
                    c = np.zeros((maxLen,maxWid,3))
                    while Alpha < 1:
                        c = blend(a,b,Alpha,maxLen,maxWid,c)
                        cv.imshow("Video",c.astype(np.uint8))
                        Alpha += .1
                        cv.waitKey(1000)

                    """,language="python")
        st.markdown("""So, this code is all about blending two images together to create a smooth transition effect between them. We start by defining a function called blend, which does the magic. It takes two images (a and b), an alpha value (a number between 0 and 1), the maximum length, maximum width, and an empty canvas c.

Now, the cool part happens inside the function. It goes through each pixel of the images a and b, and then it combines them using the alpha value to create a new pixel for the blended image c. The alpha value controls how much of each image contributes to the final result.

After defining the blend function, we load two images named image1.png and image2.png using OpenCV (cv2). We then find out the maximum dimensions between the two images and create an empty canvas c with those dimensions.

Now comes the fun part! We set Alpha to 0 and start a loop. Inside the loop, we call the blend function with the current Alpha value to blend the images and store the result in c. Then, we use cv.imshow to show the blended image as a video. By increasing the Alpha value slightly in each iteration, we make the images blend gradually, creating a smooth transition effect. And that's how we get a nice animation of the two images smoothly blending into each other!

Oh, and one last thing! The cv.waitKey(1000) line adds a 1-second delay between frames, making the blending effect visible as an animation. So, we keep doing this until Alpha reaches 1 and then start again from the beginning, giving us a cool looping animation of the blended images.""")

if __name__ == "__main__":
    main()
