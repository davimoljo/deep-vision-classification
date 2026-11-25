# deep-vision-classification

Trabalho prático de Visão Computacional (UFJF): classificação do CIFAR-10 com
uma CNN própria (baseline + ablações) e com transfer learning a partir de redes
pré-treinadas no ImageNet. Entrega: 01/12/2026.

## Estrutura

```
src/
  pipeline.py   # dados, modelos, treino e métricas (o código que importa)
  plots.py      # figuras e tabelas
notebooks/
  01_baseline.ipynb         # CNN de referência
  02_ablacoes.ipynb         # uma mudança arquitetural por run
  03_transfer.ipynb         # redes pré-treinadas x estratégias
  04_avaliacao_final.ipynb  # teste, só para as configs escolhidas na validação
report/      # trechos do relatório em LaTeX (introdução, métodos parcial, .bib)
artifacts/   # split_indices.npz (versionado) e pesos .pth (ignorados)
results/     # um JSON por run + tabelas .tex
figures/     # figuras em PDF
```

Os notebooks são finos: só definem a configuração e chamam `src/`. Não
copiem o laço de treino para dentro deles.

## Como rodar: VS Code + extensão do Google Colab

O notebook fica aberto no seu VS Code, mas **executa num servidor do Colab com
GPU**. Seu computador não treina nada.

### 1. Preparar o VS Code (uma vez)

1. Instale o [VS Code](https://code.visualstudio.com/) e o [Git](https://git-scm.com/).
2. Em Extensões (`Ctrl+Shift+X`), instale:
   - **Jupyter** (Microsoft, `ms-toolsai.jupyter`)
   - **Google Colab** (Google, `google.colab`)
3. Clone o repositório e abra a pasta no VS Code:
   ```bash
   git clone https://github.com/davimoljo/deep-vision-classification.git
   code deep-vision-classification
   ```

### 2. Conectar um notebook ao Colab

1. Abra o notebook, por exemplo `notebooks/01_baseline.ipynb`.
2. Clique em **Select Kernel** (canto superior direito) → **Colab** →
   **New Colab Server**.
3. Faça login com sua conta Google quando o navegador abrir e clique em
   **Allow**.
4. Escolha uma máquina com **GPU (T4)**. Para conferir, rode `!nvidia-smi`
   numa célula.

### 3. Montar o Google Drive (recomendado)

Sem o Drive, os resultados e os pesos somem quando o servidor desliga, e o
notebook 04 precisa dos pesos.

1. `Ctrl+Shift+P` → **Colab: Mount Google Drive to Server...**
2. Rode a célula que a extensão adiciona e autorize o acesso.
3. Rode a primeira célula do notebook (Setup). Ela deve imprimir
   `Saídas em /content/drive/MyDrive/deep-vision-classification`.

Se o Drive não montar, a célula de Setup avisa. Ainda dá para rodar, mas
baixe os resultados antes de fechar o servidor (veja o passo 5).

### 4. Rodar

1. Execute as células na ordem. A primeira clona o repositório **do GitHub**
   no servidor e instala as dependências.
2. **Importante:** o servidor roda o código que está no GitHub, não o da sua
   máquina. Se você mudou algo em `src/`, faça commit e push antes e rode a
   célula de Setup de novo (ela dá `git pull`).
3. Os notebooks 02 e 03 pulam runs que já têm JSON salvo. Se a sessão cair,
   reconecte e rode de novo para continuar de onde parou.
4. Ao terminar, encerre o servidor (`Ctrl+Shift+P` → **Colab: Remove Server**)
   para não gastar a cota de GPU.

### 5. Guardar os resultados no repositório

Cada run gera `results/<run>.json` (e uma figura em `figures/`) na pasta do
Drive `MyDrive/deep-vision-classification`.

1. Baixe os JSONs do Drive para a pasta `results/` do seu clone local.
2. Faça commit e push:
   ```bash
   git add results/ figures/
   git commit -m "Resultados: <nome da run>"
   git push
   ```

Os pesos `.pth` ficam só no Drive. Eles são grandes e estão no `.gitignore`.

### Alternativa: Colab pelo navegador

Em [colab.research.google.com](https://colab.research.google.com) → **GitHub** →
`davimoljo/deep-vision-classification` → escolha o notebook. Depois
**Ambiente de execução → Alterar o tipo → GPU T4**. A célula de Setup funciona
do mesmo jeito e monta o Drive sozinha.

## Regras do enunciado e como o código as garante

| Regra | Onde |
|---|---|
| Split fixo em todos os experimentos | `build_split()` grava `artifacts/split_indices.npz` (estratificado, 45.000/5.000, seed 42), que está no git; se o arquivo existe, ele nunca é regerado |
| Teste intocado até o fim | `fit()` usa só treino/validação; `evaluate_on_test()` só é chamado no notebook 04 |
| Hiperparâmetros constantes | `DEFAULTS` em `pipeline.py`; os notebooks não os alteram dentro de um grupo de comparação |
| Uma característica por vez | Cada ablação é um dicionário com uma única chave sobre a baseline (há um `assert` no notebook 02) |
| Teste da melhor CNN e da melhor estratégia de cada rede pré-treinada | Seleção automática pela validação no notebook 04 |

## Decisões pendentes do grupo

Os valores no código são **propostas** (marcadas com `PROPOSTA` nos
comentários). Fechem tudo **antes do primeiro treino oficial**: mudar qualquer
valor depois obriga a refazer as runs.

| Decisão | Onde | Proposta atual | Observação |
|---|---|---|---|
| Arquitetura da baseline | `BaselineCNN.__init__` (padrões) | 3 blocos, 32 filtros (dobrando), kernel 3, 2 convs/bloco, BN, dropout 0.3, FC 256, flatten | O relatório precisa descrever camadas, filtros, kernels, ativações, pooling, FC, BN/Dropout e nº de parâmetros |
| Hiperparâmetros de treino | `DEFAULTS` em `pipeline.py` | AdamW, lr 1e-3, wd 5e-4, cosine, batch 128, 30 épocas, entropia cruzada | Fixos para todas as ablações |
| Ablações | `ABLATIONS` no notebook 02 | 14 (profundidade, filtros, kernel, convs/bloco, BN, dropout, FC, GAP, residual) | O mínimo é 3. Cada uma é um treino completo; mantenham só as que vão para a discussão |
| Transfer learning | `ARCHS` e `TL_TRAIN` no notebook 03 | as 6 redes × frozen/partial/full; 10 épocas; batch 128 | O mínimo é 2 redes × 2 estratégias. 18 runs em 224 px provavelmente não cabem no Colab até 01/12; batch 128 em 224 px pode estourar a memória da T4 em redes grandes |
| lr do fine-tuning completo | `TL_TRAIN` | 1e-3 (o mesmo da CNN) | 1e-3 pode destruir os pesos do ImageNet. O enunciado permite ajustar (ex.: 1e-4) desde que justificado |
| Normalização no transfer | `get_loaders` | média/desvio do CIFAR-10 | Alternativa: estatísticas do ImageNet, que são as que as redes viram no pré-treino |
| Treino do zero (opcional) | notebook 03 | `resnet18` sem pesos | Opcional no enunciado; custa uma run a mais |

## Saída de cada run (`results/<run>.json`)

Hiperparâmetros, configuração do modelo, GPU usada, histórico por época
(perda/acurácia de treino e validação), parâmetros totais e treináveis, melhor
`val_acc` e época, tempo de treino, tempo de inferência por imagem e, depois
do notebook 04, `test_acc` e a matriz de confusão.

## Relatório

`report/` tem o texto que não depende de resultados, para colar no template
LaTeX do professor: `introducao.tex`, `metodos_parcial.tex` (lacunas marcadas
com `TODO GRUPO`) e `referencias.bib`.
