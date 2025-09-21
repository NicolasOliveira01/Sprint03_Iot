import cv2
import json
import os
from deepface import DeepFace
import numpy as np
import mediapipe as mp
import time

JSON_FILE = "banco.json"

# ---- Funções de banco de dados ----
def carregar_banco():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            return json.load(f)
    return []

def salvar_banco(banco):
    with open(JSON_FILE, "w") as f:
        json.dump(banco, f, indent=4)

# ---- Funções de DeepFace ----
def gerar_embedding(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(frame_rgb, model_name="Facenet", enforce_detection=True)
        vetor = embedding[0]["embedding"]
        return normalizar_embedding(vetor)
    except Exception as e:
        print("Não foi possível detectar o rosto:", e)
        return None

def normalizar_embedding(embedding):
    emb = np.array(embedding)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

# ---- Cadastro ----
def capturar_rosto(caption="Pressione 's' para capturar"):
    cap = cv2.VideoCapture(0)
    print(f"Ajuste o rosto na frente da webcam e pressione 's' para capturar...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow(caption, frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def cadastrar():
    banco = carregar_banco()
    frame = capturar_rosto()
    embedding = gerar_embedding(frame)
    if embedding is not None:
        banco.append({"embedding": embedding.tolist()})
        salvar_banco(banco)
        print("Cadastro realizado com sucesso!")

# ---- Validação com landmarks ----
def validar():
    banco = carregar_banco()
    if not banco:
        print("Nenhum usuário cadastrado.")
        return

    # Configuração MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    delay_seconds = 0.5
    reconhecido = False

    print("Ajuste o rosto na frente da webcam e pressione 's' para capturar...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_vis = frame.copy()  # <- cópia para desenhar landmarks
        rgb_frame = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                total_points = len(face_landmarks.landmark)
                elapsed = time.time() - start_time

                progress = min(int((elapsed - delay_seconds) * 200), total_points)
                h, w, _ = frame_vis.shape

                # Desenha os landmarks na cópia
                for i, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    color = (0, 255, 0) if i < progress else (0, 0, 255)
                    cv2.circle(frame_vis, (x, y), 6, (255, 255, 255), -1)
                    cv2.circle(frame_vis, (x, y), 4, color, -1)

                # Contornos em branco
                for conn in [mp_face_mesh.FACEMESH_FACE_OVAL,
                             mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                             mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                             mp_face_mesh.FACEMESH_LIPS]:
                    mp_drawing.draw_landmarks(
                        image=frame_vis,
                        landmark_list=face_landmarks,
                        connections=conn,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec
                    )

        cv2.imshow("Validação Facial", frame_vis)
        key = cv2.waitKey(1)

        if key == ord('s'):  # captura para validação
            # aqui usamos o frame original, sem landmarks
            novo_embedding = gerar_embedding(frame)
            if novo_embedding is not None:
                for registro in banco:
                    emb_salvo = normalizar_embedding(registro["embedding"])
                    distancia = np.linalg.norm(emb_salvo - novo_embedding)
                    if distancia < 0.9:
                        reconhecido = True
                        break
            break
        elif key == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

    if reconhecido:
        print("Usuário reconhecido!")
    else:
        print("Usuário não encontrado.")

# ---- Menu ----
def main():
    while True:
        print("\nEscolha uma opção:")
        print("1 - Cadastrar novo rosto")
        print("2 - Entrar (validar rosto)")
        print("0 - Sair")
        opcao = input("Opção: ")
        if opcao == "1":
            cadastrar()
        elif opcao == "2":
            validar()
        elif opcao == "0":
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()
