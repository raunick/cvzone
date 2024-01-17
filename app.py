import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Importando os módulos necessários
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Configurando o modelo de detecção de mãos
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

import cv2
import mediapipe as mp


def contar_dedos_levantados(pontos_mao):
    pontos_dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]

    dedos_levantados = 0

    for ponto in pontos_dedos:
        if pontos_mao.landmark[ponto].y < pontos_mao.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
            dedos_levantados += 1

    return dedos_levantados


def contar_dedos_verticais(pontos_mao):
    pontos_dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]

    dedos_verticais = 0

    for ponto in pontos_dedos:
        if pontos_mao.landmark[ponto].y < pontos_mao.landmark[mp_hands.HandLandmark.WRIST].y:
            dedos_verticais += 1

    return dedos_verticais
def contar_dedos_abaixados(pontos_mao):
    pontos_dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]

    dedos_abaixados = 0

    for ponto in pontos_dedos:
        if pontos_mao.landmark[ponto].y > pontos_mao.landmark[mp_hands.HandLandmark.WRIST].y:
            dedos_abaixados += 1

    return dedos_abaixados

def contar_dedos_dobrados(pontos_mao):
    pontos_dedos = [mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]

    dedos_dobrados = 0

    for ponto in pontos_dedos:
        if pontos_mao.landmark[ponto].y > pontos_mao.landmark[mp_hands.HandLandmark.WRIST].y:
            dedos_dobrados += 1

    return dedos_dobrados

def contar_dedos_estendidos(pontos_mao):
    pontos_dedos = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP]

    dedos_estendidos = 0

    for ponto in pontos_dedos:
        if pontos_mao.landmark[ponto].y < pontos_mao.landmark[mp_hands.HandLandmark.WRIST].y:
            dedos_estendidos += 1

    return dedos_estendidos

# Função para processar cada frame de imagem
def process(image):
    # Tornando a imagem não gravável para otimizar o processamento
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    i = 0
    # Desenhando as anotações da mão na imagem
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Contando dedos levantados
            raised_fingers = contar_dedos_levantados(hand_landmarks)

            # Atualizando a variável com o número de dedos levantados
            i = raised_fingers

    # Invertendo horizontalmente a imagem
    img_final = cv2.flip(image, 1)

    # Adicionando uma label à imagem
    label_text = f"Dedos Levantados: {i}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)  # Cor do texto em branco (BGR)

    # Obtendo as dimensões da imagem
    height, width, _ = img_final.shape

    label_position = (10, height - 30)  # Posição da label na imagem (alterada para aparecer na parte inferior)

    cv2.putText(img_final, label_text, label_position, font, font_scale, font_color, font_thickness)

    return img_final


# Configuração do servidor de Comunicação em Tempo Real (RTC)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Classe para processamento de vídeo
class VideoProcessor:
    def recv(self, frame):
        # Convertendo o frame para um array NumPy no formato "bgr24"
        img = frame.to_ndarray(format="bgr24")
        
        # Processando o frame utilizando a função definida anteriormente
        img = process(img)
        
        # Convertendo o frame processado de volta para o formato "bgr24"
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Iniciando a aplicação Streamlit
st.write('# Detector de Mãos')    

# Configurando a transmissão WebRTC
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
