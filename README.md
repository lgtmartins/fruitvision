# FruitVision

Projeto acadêmico de detecção de frutas usando Machine Learning/Deep Learning
e um dataset próprio fornecido pelo grupo.

## Classes

9 classes: apple, avocado, banana, guava, kiwi, mango, orange, peach, pineapple.

## Importante sobre a restrição do trabalho

O projeto não usa nenhuma API de IA generativa/externa em nenhuma camada
(treino, backend ou frontend). O detector é treinado pelo grupo a partir
das imagens e anotações do dataset, e o backend apenas roda esse modelo
localmente — ele não chama nenhum serviço de IA de terceiros.

O treinamento inicial usa `yolo26n.yaml`, ou seja, a arquitetura é
inicializada sem pesos pré-treinados. Não use `yolo26n.pt` nesta primeira etapa.

## Estrutura

```
fruitvision/
├── dataset/            imagens, labels e data.yaml
├── notebooks/
│   └── treino_colab.ipynb  treino completo no Google Colab com GPU
├── training/
│   ├── train.py         treinamento do modelo YOLO
│   ├── validate.py       avaliação (mAP, precisão, recall)
│   ├── predict.py        teste em uma imagem via linha de comando
│   ├── audit_dataset.py  auditoria do dataset (contagem de classes, labels ausentes...)
│   └── package_colab.py  gera o zip para treinar no Google Colab
├── database/
│   └── db.py             camada SQLite: histórico de detecções
├── backend/
│   ├── app.py             API REST (Flask) que roda o modelo treinado
│   ├── requirements.txt
│   ├── uploads/           imagens enviadas pelos usuários (criado em runtime)
│   └── results/           imagens anotadas pelo modelo (criado em runtime)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js             interface web que consome a API
└── runs/                  saída do treinamento (pesos, gráficos, métricas)
```

## Instalação

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## 1. Auditoria do dataset

```bash
python training/audit_dataset.py
```

## 2. Treinamento

```bash
python training/train.py
```

O dispositivo é escolhido automaticamente: GPU CUDA se houver, senão CPU.
Na CPU o treino completo é muito lento, então **prefira o Google Colab** (abaixo).

Isso gera os pesos em `runs/fruitvision_yolo26n_scratch/weights/best.pt`,
usados pelo backend e pelos demais scripts.

Opções úteis:

```bash
python training/train.py --epochs 3 --imgsz 320 --name teste_rapido   # só para testar o pipeline
python training/train.py --batch 16                                   # GPU com mais memória
python training/train.py --pretrained                                 # transfer learning (yolo26n.pt), só se permitido
```

### Treinando no Google Colab (recomendado)

1. Gere o pacote com `dataset/`, `training/` e `requirements.txt`:
   ```bash
   python training/package_colab.py
   ```
2. Envie o zip para a pasta **MyDrive** do Google Drive.
3. Abra `notebooks/treino_colab.ipynb` no Colab, escolha ambiente **GPU** e rode as células.
4. Baixe o `fruitvision_runs.zip` que o notebook salva no Drive e extraia na raiz do projeto.

## 3. Validação

```bash
python training/validate.py
python training/validate.py --weights runs/outro_treino/weights/best.pt
```

Mostra mAP50, mAP50-95, precisão, recall e o mAP50 de cada classe.

## 4. Predição via linha de comando (rápido, sem subir o backend)

```bash
python training/predict.py caminho/para/imagem.jpg
python training/predict.py caminho/para/imagem.jpg --weights runs/outro_treino/weights/best.pt
```

## 5. Backend (API REST)

O backend carrega `runs/fruitvision_yolo26n_scratch/weights/best.pt` e expõe
endpoints para o frontend consumir. Ele também grava cada análise no banco
SQLite (`database/fruitvision.db`, criado automaticamente).

```bash
pip install -r backend/requirements.txt
python backend/app.py
```

Para usar outros pesos sem mexer no código, defina `FRUITVISION_WEIGHTS`:

```powershell
$env:FRUITVISION_WEIGHTS = "runs/fruitvision_yolo26n_pretrained/weights/best.pt"
python backend/app.py
```

A API sobe em `http://localhost:5000`. Endpoints principais:

| Método | Rota                      | Descrição                                   |
|--------|----------------------------|----------------------------------------------|
| GET    | `/api/health`              | Verifica se a API e o modelo estão prontos    |
| POST   | `/api/predict`             | Recebe uma imagem (`image`) e um `conf` opcional; retorna as detecções |
| GET    | `/api/history?limit=50`    | Lista as últimas detecções salvas            |
| GET    | `/api/history/<id>`        | Detalhe de uma detecção específica            |
| DELETE | `/api/history/<id>`        | Remove uma detecção do histórico              |
| GET    | `/api/stats`                | Contagem de detecções por classe de fruta    |
| GET    | `/api/images/<pasta>/<arquivo>` | Serve as imagens enviadas/anotadas       |

## 6. Frontend

Página estática simples (sem build), basta servir a pasta `frontend/`:

```bash
cd frontend
python -m http.server 5500
```

Depois abra `http://localhost:5500` no navegador, com o backend (passo 5)
rodando em paralelo em outro terminal.

O frontend permite:
- enviar uma imagem e ver as caixas detectadas com a confiança de cada fruta;
- ajustar o limiar de confiança mínima;
- consultar o histórico de análises já feitas (aba **Histórico**);
- ver a contagem total de detecções por espécie de fruta (aba **Estatísticas**).

## Fluxo completo, resumido

1. `python training/audit_dataset.py` — confirma que o dataset está íntegro.
2. `python training/train.py` — treina o modelo do zero com o dataset.
3. `python training/validate.py` — mede mAP/precisão/recall no conjunto de validação.
4. `python backend/app.py` — sobe a API que roda o modelo treinado.
5. `cd frontend && python -m http.server 5500` — sobe a interface web.
