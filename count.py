from ultralytics import YOLO
import cv2
import os

model_path = "/home/inovacai/Desktop/caio/24-02/figo_exn2.pt"
video_path = "/home/inovacai/Downloads/4.mp4"

if not os.path.exists(video_path):
    print("Erro: O arquivo de vídeo não foi encontrado!")
    exit()

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo! Verifique o formato e os codecs.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
tempo_total = frame_count / fps

tempo_deitado = 0
tempo_em_pe = 0
tempo_sentado = 0
tempo_nao_detectado = 0

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.predict(frame, conf=0.7, device="0", show=True)

    if len(results[0].boxes) > 0:
        classe = results[0].boxes.cls[0].item()
        if classe == 0:
            tempo_deitado += 1 / fps
        elif classe == 1:
            tempo_em_pe += 1 / fps
        elif classe == 2:
            tempo_sentado += 1 / fps
    else:
        tempo_nao_detectado += 1 / fps

cap.release()

porcentagem_deitado = (tempo_deitado / tempo_total) * 100
porcentagem_em_pe = (tempo_em_pe / tempo_total) * 100
porcentagem_sentado = (tempo_sentado / tempo_total) * 100
porcentagem_nao_detectado = (tempo_nao_detectado / tempo_total) * 100

print(f"Tempo em pé: {tempo_em_pe:.2f} segundos ({porcentagem_em_pe:.2f}% do tempo total)")
print(f"Tempo sentado: {tempo_sentado:.2f} segundos ({porcentagem_sentado:.2f}% do tempo total)")
print(f"Tempo deitado: {tempo_deitado:.2f} segundos ({porcentagem_deitado:.2f}% do tempo total)")
print(f"Tempo com usuário não detectado: {tempo_nao_detectado:.2f} segundos ({porcentagem_nao_detectado:.2f}% do tempo total)")
print(f"Duração total: {tempo_total:.2f} segundos")