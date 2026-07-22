import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------


def load_dataset():
    """Função de carregamento e normalização do dataset MNIST."""

    # carregamento dos dados brutos:
    # o dataset é retornado como arrays uint8, agilizando load_data(),
    # de shape (60000, 28, 28) para treino e (10000, 28, 28) para teste;
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # normalização dos pixels:
    # converte explicitamente os valores de uint8 para float32, e os normaliza
    # para o intervalo [0, 1], facilitando o treinamento também evitando overflow;
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # ajuste do formato das imagens:
    # adiciona explicitamente o canal de cores exigido pelas camadas convolucionais
    # de (28, 28) para (28, 28, 1), indicando assim um pixel monocromático 
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Função de criação, inicialização e compilação do modelo."""

    model = keras.Sequential([

        # camada de entrada:
        # define o formato esperado para cada imagem do dataset;
        keras.Input(shape=(28, 28, 1)),

        # PRIMEIRO BLOCO CONVOLUCIONAL:
        
        # aprende características simples como bordas, linhas
        # e pequenas formas presentes na imagem usando 32 filtros 3x3;
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),

        # normalização das ativações:
        # estabiliza o treinamento reduzindo mudanças bruscas
        # na distribuição dos dados entre as camadas;
        layers.BatchNormalization(),

        # redução da resolução:
        # diminui a dimensão espacial preservando as
        # características mais importantes;
        layers.MaxPooling2D((2, 2)),

        # Os demais blocos dobram os filtros que aprendem caracteristicas 
        # ao passo que reduzem a resolução da imagem pela metade

        # SEGUNDO BLOCO CONVOLUCIONAL:
        # aprende padrões mais complexos combinando as
        # características extraídas anteriormente;
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # TERCEIRO BLOCO CONVOLUCIONAL:
        # identifica características de maior nível,
        # importantes para a classificação final;
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # achatamento dos mapas de características:
        # transforma a saída convolucional em um vetor
        # unidimensional para alimentar as camadas densas;
        layers.Flatten(),

        # camada densa:
        # combina todas as características aprendidas
        # para produzir uma representação final;
        layers.Dense(128, activation="relu"),

        # regularização:
        # desativa aleatoriamente parte dos neurônios durante
        # o treinamento para reduzir overfitting;
        layers.Dropout(0.3),

        # camada de saída:
        # produz uma probabilidade para cada uma das
        # dez classes do dataset MNIST usando softmax;
        layers.Dense(10, activation="softmax")
    ])

    # configuração do treinamento:
    # define o algoritmo de otimização, a função de perda
    # e a métrica utilizada durante o treinamento;
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


def train_model(model, x_train, y_train):
    """Função responsável pelo treinamento do modelo."""

    # configuração do early stopping:
    # interrompe o treinamento caso a perda de validação
    # deixe de melhorar por duas épocas consecutivas;
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # treinamento do modelo:
    # utiliza 10% das imagens de treino como conjunto
    # de validação para acompanhar o desempenho;
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1, # SPLIT !
        callbacks=[early_stop]
    )

    # obtenção das melhores métricas:
    # como restore_best_weights=True, essas métricas
    # correspondem ao melhor modelo encontrado;
    val_acc = max(history.history["val_accuracy"])
    val_loss = min(history.history["val_loss"])

    print(f"Acurácia de validação: {val_acc:.4f}")
    print(f"Perda de validação: {val_loss:.4f}")

    return history


def evaluate_model(model, x_test, y_test):
    """Função de avaliação do modelo. (Apesar de não cobrado)"""

    # avaliação final:
    # mede o desempenho utilizando imagens nunca vistas
    # durante o treinamento;
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(f"Acurácia no teste: {test_acc:.4f}")
    print(f"Perda no teste: {test_loss:.4f}")


def save_model(model):
    """Função responsável por salvar o modelo treinado."""

    # persistência do modelo:
    # salva arquitetura, pesos e configuração para
    # utilização posterior;
    model.save("model.h5")

    print("Modelo salvo como model.h5")


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset()

    model = build_model()

    train_model(model, x_train, y_train)

    evaluate_model(model, x_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()