from ultralytics import YOLO

def train_model():
    # Carrega o modelo pré-treinado
    model = YOLO("yolo11n.pt")  # Confirme que esse modelo existe
    model.to("cuda")  # Envia para GPU

    # Treinamento do modelo
    results = model.train(
        data="data.yaml",
        epochs=600,
        imgsz=640,
        batch=32,  # Reduza se der erro de memória
        device="cuda",
        #patience=100,
        #lr0=0.03,
        #lrf=0.0003,
        #optimizer="adam",
        dropout=0.3,
        augment=True  # Ative se precisar de regularização
        # freeze=[0, 1, 2]  # Ative se quiser congelar camadas iniciais
    )

    # Avaliação do modelo nos dados de validação
    metrics = model.val()
    print(metrics)  # Exibe os resultados

    # Exporta o modelo para TorchScript (opcional)
    #model.export(format="pt")
    #model.export(format="onnx")

    return metrics

if __name__ == '__main__':
    train_model()
