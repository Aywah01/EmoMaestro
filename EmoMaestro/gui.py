
import streamlit as st
import time

from playsound import playsound

from face import FaceModel
from music import MusicModel
from prompt import Prompt
from text import TextModel
from util import fuse
from vad_map import VADMap

st.set_page_config(
    page_title = "EmoMaestro GUI",
    layout = "wide"
)

# ===== Title =====
st.title("🎵 EmoMaestro")
st.caption("감정 기반 음악 생성 시스템")

st.divider()

# ===== Layout =====
left, center, right = st.columns([1.2, 1.5, 1.3])

# ===== LEFT: Input =====
with left:
    st.subheader("1️⃣ 사용자 입력")

    user_text = st.text_area(
        "텍스트 입력: 현재 느끼는 감정을 설명하세요",
        placeholder = "어느 봄날 당신의 사랑으로 응달지던 내 뒤란에 햇빛이 들이치는...",
        height = 150
    )

    visual_type = st.radio(
        "입력 유형 선택",
        ["라이브 스트리밍", "이미지 업로드"],
        horizontal = True
    )

    if visual_type == "라이브 스트리밍":
        visual_path = FaceModel.live_stream_key

    else:
        visual_path = st.file_uploader(
            "얼굴 사진을 업로드하세요",
            type = FaceModel.supported_img_types
        )

        if visual_path:
            st.image(visual_path, caption = "업로드된 이미지", use_container_width = True)


    generate_btn = st.button("🎶 음악 생성", type = "primary", use_container_width = True)

# ===== CENTER: Emotion Analysis =====
with center:
    st.subheader("2️⃣ 감정 분석")

    if generate_btn:
        with st.spinner("감정 분석..."):
            fm = FaceModel()
            fm.load_weights()

            tm = TextModel()

            try:
                music_model = MusicModel()
                prompt = Prompt()

                text_dict = dict()
                text_vad = None
                face_vad = None

                st.subheader("User Input")

                if user_text:
                    st.code(f"Text Diary: {user_text}")
                    text_dict = tm.predict(user_text)

                if visual_path:
                    st.code(f"Visual Path: {visual_path}\n")
                    face_vad = fm.predict(visual_path)

                if text_dict:
                    emotions = []
                    emotion_weight = []

                    for k, v in text_dict.items():
                        emotions.append(k)
                        emotion_weight.append(v)

                    vad_map = VADMap(set(emotions))

                    tot_v, tot_a, tot_d = 0, 0, 0,

                    for i, emo in enumerate(emotions):
                        v, a, d = vad_map.get_emo_vad(emo)
                        tot_v += v * emotion_weight[i]
                        tot_a += a * emotion_weight[i]
                        tot_d += d * emotion_weight[i]

                    text_vad = tot_v, tot_a, tot_d

                vad = fuse(text_vad = text_vad, face_vad = face_vad)

                if not vad:
                    raise Exception("No VAD metric!")

            except Exception as e:
                print(f"Pipeline Error: {e}")

        st.success("감정 분석 완료")

        st.code(f"Valence {round(vad[0], 2)}")
        st.code(f"Arousal {round(vad[1], 2)}")
        st.code(f"Dominance {round(vad[2], 2)}")
    else:
        st.info("사용자 입력을 기다립니다...")

# ===== RIGHT: Music Output =====
with right:
    st.subheader("3️⃣ 음악 출력")

    if generate_btn and vad:
        p = prompt.gen(vad)
        st.subheader("생성된 음악 프롬프트")
        st.code(p)

    if generate_btn and p:
        with st.spinner("음악 생성..."):
            audio_path = music_model.gen(p)

        st.success("음악이 성공적으로 생성되었습니다.")
        st.code(f"오디오 경로: {audio_path}")

        st.audio(
            f"{audio_path}",
            format = "audio/wav"
        )

    else:
        st.info("음악 출력이 여기에 표시됩니다")