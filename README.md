# Reconhecimento Facial com OpenCV, MediaPipe e DeepFace

## 📌 Objetivo
Este projeto implementa um sistema local de **cadastro e validação facial** utilizando `Numpy`, `OpenCV`, `MediaPipe`, `DeepFace`, `json`, `os` e `time`. No momento, 
ele funciona em ambiente **desktop/notebook** com webcam, em etapas futuras, será integrado a um **aplicativo mobile em React Native**, 
permitindo que usuários se cadastrem por reconhecimento facial.

---

## ⚙️ Como executar
1. Primeiro certifique-se que você tenha o `CMake` e o `Visual C++ Build Tools`
    1. CMAKE:
    ```bash
    cmake --version
    ```
   Caso não tenha instalado -> https://cmake.org/download/
    
    2. Visual C++ Build Tools
    ```bash
    Painel de controle - Programas - Programs e Recursos
    ```
   Caso não tenha instalado -> https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/
2. Clone este repositório  
    ```bash
    git clone https://github.com/NicolasOliveira01/Sprint03_Iot
    ```
3. Instale as dependências
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
5. Execute o script principal:
   ```bash
   python main.py
   ```
   
---

## 📦 Dependências e versões recomendadas

### **OpenCV (`cv2`)**
- Captura vídeo da webcam.
- Exibe imagens em janelas.
- Converte imagens entre formatos BGR/RGB.

### **JSON (`json`)**
- Salva e carrega embeddings faciais no arquivo `banco.json`.
- Permite persistência dos dados sem necessidade de banco de dados complexo.

### **OS (`os`)**
- Verifica se o arquivo `banco.json` existe antes de carregá-lo.

### **DeepFace (`deepface`)**
- Gera embeddings faciais a partir da imagem do rosto.
- Baseado em modelos de deep learning, neste projeto usamos **Facenet**.
- Embeddings são normalizados para comparação entre rostos.

### **NumPy (`numpy`)**
- Manipula arrays de embeddings.
- Calcula distância euclidiana entre embeddings para validação do usuário.
- Converte embeddings para listas antes de salvar no JSON.

### **MediaPipe (`mediapipe`)**
- Detecta landmarks faciais em tempo real.
- Desenha pontos e contornos do rosto, sobrancelhas e boca.
- Proporciona feedback visual para o usuário durante validação.

### **Time (`time`)**
- Controla a animação dos landmarks (mudança progressiva de cor).
- Mede o tempo decorrido para definir o progresso do desenho.

---

## Funções principais

### **Banco de dados**
- `carregar_banco()`: Carrega os embeddings do arquivo `banco.json`.  
- `salvar_banco(banco)`: Salva os embeddings no arquivo `banco.json`.  

### **DeepFace**
- `gerar_embedding(frame)`: Gera o embedding facial de uma imagem capturada.  
- `normalizar_embedding(embedding)`: Normaliza o embedding para comparação.

### **Cadastro**
- `capturar_rosto(caption)`: Captura a imagem do rosto via webcam ao pressionar `s`.  
- `cadastrar()`: Realiza o cadastro de um novo usuário gerando e salvando o embedding.

### **Validação**
- `validar()`: Captura o rosto, mostra landmarks em tempo real, e compara o embedding com o banco para validar o usuário.

### **Menu**
- `main()`: Exibe o menu principal com opções de cadastro, validação ou sair do programa.

---

# Threshold de Reconhecimento Facial

No projeto, o **threshold** é o valor usado para determinar se um rosto capturado corresponde a um rosto cadastrado no banco de dados.

## Como funciona

- Cada rosto é transformado em um **embedding** (vetor numérico) usando DeepFace.  
- Para validar um usuário, calculamos a **distância euclidiana** entre o embedding capturado e os embeddings salvos no banco.  
- O threshold define o limite máximo dessa distância para considerar que os rostos são da mesma pessoa.  

```python
distancia = np.linalg.norm(emb_salvo - novo_embedding)
if distancia < 0.9:  # threshold
    reconhecido = True
