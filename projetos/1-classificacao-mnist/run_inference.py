import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do MNIST
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------

# refatorei o código original deste arquivo para seguir o padrão
# utilizado nos demais, mas preservando as instruções já presentes

N_SAMPLES = 5

def load_test_samples():
    """Função de carregamento e normalização das amostras de teste."""

    # aqui só o conjunto de teste é necessário, já que o modelo
    # foi treinado e otimizado nas etapas anteriores;
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # mantém o mesmo pré-processamento aplicado no treinamento
    x_test = x_test.astype("float32") / 255.0

    # há várias formas de formatar, mas preservei a forma original 
    x_test = np.expand_dims(x_test, axis=-1)

    return x_test, y_test


def load_interpreter(model_path="model.tflite"):
    """Função de carregamento do interpretador TensorFlow Lite."""

    # resolução do caminho absoluto:
    # garante que o model.tflite seja encontrado independentemente
    # do diretório a partir do qual o script é executado;
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, model_path)

    # criação do interpretador:
    # carrega o artefato de edge já quantizado, em vez do model.h5;
    interpreter = tf.lite.Interpreter(model_path=full_path)

    # alocação dos tensores:
    # reserva a memória necessária para entradas, saídas
    # e tensores intermediários do modelo;
    interpreter.allocate_tensors()

    return interpreter


def run_inference(interpreter, x_test, y_test, n_samples=N_SAMPLES):
    """Função responsável por rodar a inferência amostra a amostra."""

    # obtenção dos detalhes de entrada e saída:
    # necessários para saber o índice e o dtype esperado pelo interpretador;
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Rodando inferência em {n_samples} amostras usando model.tflite:\n")

    for i in range(n_samples):

        # preparação da amostra:
        # adiciona a dimensão de batch e ajusta o dtype
        # conforme exigido pelo tensor de entrada;
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])

        # execução da inferência:
        # define o tensor de entrada e invoca o interpretador;
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()

        # obtenção da predição:
        # recupera o tensor de saída e extrai a classe de maior
        # probabilidade;
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        predicted_class = int(np.argmax(pred))

        print(f"Amostra {i + 1}: predito={predicted_class} | real={int(y_test[i])}")


def main():

    x_test, y_test = load_test_samples()

    interpreter = load_interpreter()

    run_inference(interpreter, x_test, y_test)


if __name__ == "__main__":
    main()