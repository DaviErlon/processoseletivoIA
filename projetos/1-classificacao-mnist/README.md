# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Davi Erlon Lopes de Morais

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` é composta por 3 blocos convolucionais, responsáveis por extrair
progressivamente características das imagens do MNIST, identificando elementos mais simples, como bordas
e linhas, até padrões mais complexos, relevantes para a classificação final. Cada bloco é formado por três camadas:
uma `Conv2D` (com 32, 64 e 128 filtros 3x3, respectivamente, dobrando a profundidade a cada bloco) usando a
função de ativação ReLU, responsável por introduzir não-linearidade e permitir que a rede aprenda relações
mais complexas entre os pixels; seguida de uma `BatchNormalization`, que estabiliza o treinamento normalizando
as ativações entre as camadas e reduz mudanças bruscas na distribuição dos dados; e uma `MaxPooling2D`, que
reduz a resolução espacial pela metade a cada bloco, preservando as características mais importantes e
diminuindo o custo computacional das camadas seguintes.

Após os três blocos convolucionais, a saída é achatada usando a função `Flatten` e transformada em um vetor unidimensional
(3, 3, 128) -> (1152), que alimenta uma camada densa de 128 neurônios, também com ativação ReLU, responsável por combinar
todas as características extraídas em uma representação final antes da classificação. Em seguida, uma camada de
`Dropout` (0.3) desativa aleatoriamente 30% dos neurônios durante o treinamento, técnica de regularização
que reduz o overfitting ao evitar que a rede dependa excessivamente de neurônios específicos. Por fim, a
camada de saída utiliza ativação `softmax`, produzindo uma distribuição de probabilidade entre as 10 classes
do dataset, que somadas dêem 1.

Para a estratégia de validação, 10% dos dados de treino são separados automaticamente via `validation_split=0.1`,
permitindo acompanhar o desempenho do modelo em dados não vistos durante cada época. O treinamento também conta
com `EarlyStopping`, monitorando a perda de validação (`val_loss`) com paciência de 2 épocas — ou seja, o
treinamento é interrompido caso a métrica não melhore por 2 épocas consecutivas (sendo uma época cada vez que o modelo percorre todo o conjunto de treino). Combinado com `restore_best_weights=True`, isso garante que os pesos salvos ao final correspondam ao melhor ponto de desempenho observado, evitando que o modelo final seja prejudicado por overfitting nas últimas épocas.

#### Escolhas de hiperpârametros

Algumas escolhas de hiperparâmetro merecem destaque. A progressão de filtros (32 → 64 → 128,
dobrando a cada bloco) segue uma prática comum em CNNs: começar com poucos filtros na primeira
camada, já que características de baixo nível (bordas, linhas) não exigem muitos detectores
diferentes, e aumentar a profundidade nos blocos seguintes, onde a rede precisa combinar essas
características simples em padrões mais complexos e específicos das classes. Começar já em 128
filtros seria custoso e desnecessário logo na primeira camada; começar em 32 e dobrar mantém o
crescimento de capacidade proporcional à complexidade que cada bloco precisa capturar.

O valor de `Dropout(0.3)` foi escolhido como meio-termo: valores mais baixos (0.1) dariam
regularização insuficiente pra uma rede com 3 blocos convolucionais e 128 neurônios na camada
densa, enquanto valores mais altos (0.5+) descartariam informação demais numa tarefa já
relativamente simples como o MNIST — a proximidade entre a acurácia de validação (99,42%) e a de
teste (99,39%) sugere que o equilíbrio funcionou.

Já o `patience=2` do `EarlyStopping` foi definido baixo de propósito: como o MNIST converge rápido
e o `val_loss` tende a estabilizar cedo, uma paciência maior só arriscaria treinar épocas extras
sem ganho real, enquanto um valor menor (0 ou 1) poderia interromper o treino por uma flutuação
normal da métrica.

### 2️⃣ Bibliotecas Utilizadas

O projeto utiliza duas bibliotecas externas, listadas no `requirements.txt`:

- **`tensorflow==2.15.1`** — usada em todas as etapas do pipeline: construção e treinamento da CNN
  (`train_model.py`), conversão e quantização do modelo (`optimize_model.py`) e inferência via
  `tf.lite.Interpreter` (`run_inference.py`).
- **`numpy>=1.26.4`** — usada para manipulação dos arrays de imagens (normalização, reshape e
  montagem dos batches de entrada).

Além dessas, também é utilizada a biblioteca padrão python para interação trivial com arquivos do sistema.

### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, o modelo é convertido para TensorFlow Lite utilizando **Dynamic Range
Quantization**, habilitada através de `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. Essa técnica quantiza os pesos do modelo de `float32` para `int8`, reduzindo significativamente o tamanho do arquivo e acelerando a inferência, sem exigir um dataset representativo para calibração (diferente da Full Integer Quantization). O resultado é salvo como `model.tflite`, o artefato de edge utilizado na etapa de inferência.

### 4️⃣ Resultados Obtidos

O modelo atingiu **99,42% de acurácia de validação** (com perda de validação de 0,0261), obtida na época com melhor desempenho e restaurada via `restore_best_weights=True`. No conjunto de teste, a acurácia foi de **99,39%**.

Tamanho dos artefatos gerados:

| Arquivo | Tamanho |
|---|---|
| `model.h5` | 2.913,79 KB (~2,91 MB) |
| `model.tflite` | 247,73 KB (~0,24 MB) |

A quantização dinâmica aplicada em `optimize_model.py` reduziu o tamanho do modelo em **91,50%**,
tornando-o significativamente mais leve para cenários de inferência em edge, com impacto mínimo
esperado sobre a acurácia.

### 5️⃣ Comentários Adicionais

Durante o treinamento, o `EarlyStopping` interrompeu o processo na **9ª época**, mesmo com o limite configurado em 10 épocas. Isso aconteceu porque a perda de validação parou de melhorar por 2 épocas consecutivas (a partir da época 7, que teve o melhor `val_loss` de 0,0261), atingindo a paciência configurada. Graças ao `restore_best_weights=True`, os pesos salvos correspondem à época 7, e não à última executada, evitando que o modelo final fosse prejudicado por um leve overfitting.

Também foi testada uma variação da arquitetura com um **quarto bloco convolucional**, idêntico ao
terceiro (128 filtros + `BatchNormalization` + `MaxPooling2D`). O resultado não trouxe ganho de
acurácia relevante, mas deixou o modelo cerca de **800 KB mais pesado**, tornando essa alteração
inviável para os objetivos do projeto — especialmente considerando a etapa seguinte de otimização.

Por fim, vale uma observação sobre a etapa de otimização: embora a Dynamic Range
Quantization reduza bastante o tamanho do arquivo, o ganho de performance
de inferência não é tão expressivo quanto o de tamanho, já que os pesos ficam armazenados em int8, mas ainda precisam ser convertidos de volta para float durante a computação — as ativações continuam em float32. Ou seja, essa técnica otimiza principalmente armazenamento e transferência,
não necessariamente velocidade de inferência. Para um ganho de performance mais real para edge em problemas mais complexos, 
o caminho mais adequado seria a **Full Integer Quantization**.

### 6️⃣ Exemplo de Inferência

Saída ao rodar `run_inference`:

```bash
  Rodando inferência em 5 amostras usando model.tflite:

  Amostra 1: predito=7 | real=7
  Amostra 2: predito=2 | real=2
  Amostra 3: predito=1 | real=1
  Amostra 4: predito=0 | real=0
  Amostra 5: predito=4 | real=4
```

Todas as amostras foram respondidas corretamente pelo modelo quantizado e otimizado. Vale destacar a amostra 4 (dígito 0): por ter um contorno simples e bem definido, tende a ser um dos dígitos mais fáceis de classificar mesmo após a quantização, já que sua forma geométrica é pouco ambígua se comparada a dígitos visualmente próximos entre si, como 4/9 ou 3/8.
