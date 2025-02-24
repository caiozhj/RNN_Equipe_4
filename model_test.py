from ultralytics import YOLO
import os

def testar_modelo_pasta(caminho_pasta, caminho_modelo, confianca=0.7, tamanho=(640, 640)):
    # Carrega o modelo YOLO a partir do arquivo especificado
    modelo = YOLO(caminho_modelo)
    
    # Define as extensões suportadas para imagens e vídeos
    extensoes_imagem = {'.jpg', '.jpeg', '.png', '.bmp'}
    extensoes_video = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # Lista todos os arquivos na pasta informada
    arquivos = os.listdir(caminho_pasta)
    
    for arquivo in arquivos:
        caminho_arquivo = os.path.join(caminho_pasta, arquivo)
        _, extensao = os.path.splitext(arquivo)
        extensao = extensao.lower()
        
        # Verifica se o arquivo é uma imagem
        if extensao in extensoes_imagem:
            print(f"Processando imagem: {arquivo}")
            # Realiza a predição com a dimensão especificada
            resultados = modelo.predict(source=caminho_arquivo, show=True, conf=confianca, imgsz=tamanho)
            print(f"Resultados para {arquivo}:", resultados)
        
        # Verifica se o arquivo é um vídeo
        elif extensao in extensoes_video:
            print(f"Processando vídeo: {arquivo}")
            # Realiza a predição com a dimensão especificada
            resultados = modelo.predict(source=caminho_arquivo, show=True, conf=confianca, imgsz=tamanho)
            print(f"Resultados para {arquivo}:", resultados)
        
        else:
            print(f"Ignorando arquivo: {arquivo} (extensão não suportada)")

if __name__ == '__main__':
    # Defina o caminho para a pasta contendo imagens e vídeos
    caminho_pasta = "c:/Users/pc-vitu/Desktop/exemplos"  # ajuste para o caminho real
    
    # Defina o caminho/modelo que deseja utilizar
    caminho_modelo = "C:/Users/pc-vitu/Desktop/figo_exn/runs/detect/train/weights/best.pt"
    
    # Executa a função com as configurações desejadas
    testar_modelo_pasta(caminho_pasta, caminho_modelo, confianca=0.5, tamanho=(640, 640))
