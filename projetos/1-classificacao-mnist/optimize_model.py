import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

def load_trained_model(model_path="model.h5"):
    """Função de carregamento do modelo treinado."""

    # carregamento do modelo salvo:
    # recupera arquitetura, pesos e configuração gerados
    # na etapa de treinamento;
    model = keras.models.load_model(model_path)

    print(f"Modelo '{model_path}' carregado com sucesso.")

    return model


def convert_to_tflite(model):
    """Função de conversão do modelo para TensorFlow Lite com otimização."""

    # criação do conversor:
    # utiliza o modelo Keras carregado como fonte
    # para a conversão ao formato TFLite;
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # aplicação da técnica de otimização:
    # o modo DEFAULT ativa a Dynamic Range Quantization
    # para converter os pesos de float32 para int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # execução da conversão:
    # gera os bytes do modelo já otimizado;
    tflite_model = converter.convert()

    print("Conversão para TensorFlow Lite concluída.")

    return tflite_model


def save_tflite_model(tflite_model, output_path="model.tflite"):
    """Função responsável por salvar o modelo convertido."""

    # persistência do modelo otimizado:
    # grava os bytes do modelo TFLite em disco;
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Modelo otimizado salvo como {output_path}")


def compare_model_sizes(original_path="model.h5", optimized_path="model.tflite"):
    """Função de comparação do tamanho dos modelos antes e depois da otimização."""

    # obtenção do tamanho dos arquivos:
    # converte de bytes para KB para facilitar a leitura;
    original_size = os.path.getsize(original_path) / 1024
    optimized_size = os.path.getsize(optimized_path) / 1024

    # cálculo da redução percentual:
    # indica o quanto o modelo diminuiu após a quantização;
    red = (1 - optimized_size / original_size) * 100

    print(f"Tamanho original (model.h5): {original_size:.2f} KB")
    print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} KB")
    print(f"Redução de tamanho: {red:.2f}%")


def main():

    model = load_trained_model()

    tflite_model = convert_to_tflite(model)

    save_tflite_model(tflite_model)

    compare_model_sizes()


if __name__ == "__main__":
    main()