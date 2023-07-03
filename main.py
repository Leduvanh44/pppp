
import base64
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import time
import cv2
import tempfile
DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(max_num_faces=3)
#------------------------------------------------Font_text adn Background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://free4kwallpapers.com/uploads/originals/2019/01/19/one-amazing-universe-wallpaper.jpg");
    background-size: 180%;
    background-position: top left;
    background-repeat: nrepeat-y;
    background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: linear-gradient(AliceBlue, MidnightBlue);
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
    right: 2rem;
    color: white;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
css = """
<style>
    .my-text1{
        color: white;
        font-size: 40px;
    }
</style>
<style>
    .my-text2{
        color: white;
        font-size: 25px;
    }
</style>
<style>
    .my-text3{
        color: white;
    }
</style>
<style>
    .my-link {
        color: ghostwhite;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
#------------------------------------------------
st.markdown('<h1 class="my-text1">Begin Mediatime</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
        width: 350px
        color: white
    }
    [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
        width: 350px
        margin-left: -350px
        color: white
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown('<h1>Mediapipe Sidebar</h1>', unsafe_allow_html = True)

@st.cache_data()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = width/float(w)
        dim = (int(r*w), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
app_mode = st.sidebar.selectbox('Choose The app mode',
                                ['About App', 'Run Image', 'Run Weejo'])
if app_mode == 'About App':
    st.markdown('<p class = "my-text2">In this application we re using **Mediapipe** for creating Face mesh app</p>',unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
            width: 350px
            color: white
        }
        [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
            width: 350px
            margin-left: -350px
            color: white
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write('''<p class = "my-text2">
        About Mediapipe, you can learn more at:
            <a href="https://codepen.io/mediapipe/pen/RwGWYJw" target="_blank">Mediapipe</a>
        </p>''', unsafe_allow_html=True)
    st.markdown('''<p class = "my-text2">
        About Streamlit, you can learn more at: 
            <a href="https://docs.streamlit.io/library/get-started" target="_blank">Streamlit</a>
        </p>''', unsafe_allow_html=True)

elif app_mode == 'Run Weejo':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=1, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    if not use_webcam:
        st.sidebar.video(tfflie.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown("<h2 style = 'color: ghostwhite; font-size: 25px;' >FrameRate</h2>", unsafe_allow_html=True)
        kpi1_txt = st.markdown("green:[0]")

    with kpi2:
        st.markdown("<h2 style = 'color: ghostwhite; font-size: 25px;' >Detected Faces</h2>", unsafe_allow_html=True)
        kpi2_txt = st.markdown("green:[0]")

    with kpi3:
        st.markdown("<h2 style = 'color: ghostwhite; font-size: 25px;' >Image Width x Height</h2>", unsafe_allow_html=True)
        kpi3_txt = st.markdown(":green[0]")

    st.markdown("<hr/>", unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces) as face_mesh:
        prevTime = 0
        while vid.isOpened():
            if use_webcam:
                vid_in = cv2.VideoCapture(0)
                FRAME_WINDOW = st.sidebar.image([])
                ff, frame_cam = vid_in.read()
                frameRGB = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)

                FRAME_WINDOW.image(frameRGB, channels='BGR', use_column_width=True)
            i += 1
            sus, frame = vid.read()
            if not sus:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                out.write(frame)
            kpi1_txt.write(f"<p style='text-align: left; color: white;'>{int(fps)}</p>", unsafe_allow_html=True)
            kpi2_txt.write(f"<p style='text-align: left; color: white;'>{face_count}</p>", unsafe_allow_html=True)
            kpi3_txt.write(f"<p style='text-align: left; color: white;'>{width} x {height}</p>", unsafe_allow_html=True)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=840)
            stframe.image(frame, channels='BGR', use_column_width=True)

    # output_video = open('output1.mp4', 'rb')
    # out_bytes = output_video.read()
    # st.video(out_bytes)
    # vid.release()
    # out.release()

elif app_mode == 'Run Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
            width: 400px
        }
        [data-testid="stSidebar"][aria-expanded= "true"] > div:first=child{
            width: 400px
            margin-left: -400px
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class = "my-text3">Detected Face</h2>',unsafe_allow_html=True)
    kpi1_text = st.markdown('<p class = "my-text3">0</p>', unsafe_allow_html=True)
    max_faces = st.sidebar.number_input('Maximum Number of Face', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png"])
    if img_file is not None:
        image = np.array(Image.open(img_file))
    else:
        demo_img = DEMO_IMAGE
        image = np.array(Image.open(demo_img))

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    face_count = 0
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
                image=out_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>Number of Face in Image: {face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image, use_column_width=True)














