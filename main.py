import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)
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

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

windowName = "Hand Tracking"

while True:
    ret, frame = cap.read()

    if not ret:
        print("Não há frames disponíveis.")
        break

    # Converter o frame para escala de cinza
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar mãos no frame
    results = hands.process(color)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Iterar sobre os pontos da mão
            for idx, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                #printa os pontos da mão
                print(f'Ponto {idx}: ({cx}, {cy})')
            t = contar_dedos_levantados(hand_landmarks)
            print(f'dedos levantados:{t}')
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow(windowName, frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

    if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
cap.release()
print("Encerrou")
