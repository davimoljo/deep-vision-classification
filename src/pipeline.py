"""Pipeline de experimentos: dados, modelos, treino e métricas.

Todos os notebooks importam daqui. A ideia é que um experimento seja apenas
uma configuração; qualquer lógica de treino escrita nos notebooks seria uma
segunda fonte de verdade e quebraria a comparabilidade entre runs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# O split fica SEMPRE dentro do repositório: ele é versionado no git para que
# todo o grupo treine exatamente com as mesmas 45.000/5.000 imagens.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLIT_PATH = ROOT / "artifacts" / "split_indices.npz"

# Saídas (pesos, JSONs, figuras) podem ir para outro lugar via OUTPUT_ROOT.
# No Colab isso aponta para o Google Drive; sem isso, os pesos das melhores
# épocas somem quando a sessão cai e o notebook 04 não teria o que avaliar.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", ROOT))
CHECKPOINT_DIR = OUTPUT_ROOT / "artifacts"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
NUM_CLASSES = len(CLASSES)

# Estatísticas do conjunto de treino do CIFAR-10 (padrão da literatura).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

SEED = 42
VAL_SIZE = 5_000

# Hiperparâmetros de treino compartilhados. O enunciado exige que eles fiquem
# constantes entre configurações comparadas; centralizar aqui evita que um
# notebook mude o lr "só para testar" e contamine a comparação.
# PROPOSTA — confirmar com o grupo antes do primeiro treino oficial: depois
# disso, mudar qualquer valor obriga a refazer todas as runs já feitas.
DEFAULTS = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
}


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Fixa as sementes de todas as fontes de aleatoriedade.

    Sem isso, duas runs que diferem em um único argumento também diferem na
    inicialização dos pesos e na ordem dos batches, e não dá para atribuir a
    diferença de acurácia à mudança arquitetural.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark escolhe algoritmos de convolução pelo tempo, o que
    # introduz não-determinismo. Perdemos um pouco de velocidade em troca de
    # runs repetíveis.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _seed_worker(_worker_id: int) -> None:
    # Cada worker do DataLoader tem seu próprio estado de numpy/random; sem
    # isto, a augmentation varia entre execuções mesmo com a semente fixa.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def build_split(path: Path = SPLIT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (train_idx, val_idx) sobre as 50.000 imagens de treino.

    Se o arquivo já existe, ele é carregado e nunca regerado: o split tem de
    ser idêntico em todos os experimentos e em todas as máquinas do grupo.
    Regerar dependeria de versões de sklearn/numpy darem o mesmo resultado,
    e o arquivo versionado elimina essa dúvida.
    """
    path = Path(path)
    if path.exists():
        data = np.load(path)
        train_idx, val_idx = data["train_idx"], data["val_idx"]
    else:
        labels = np.array(datasets.CIFAR10(DATA_DIR, train=True, download=True).targets)
        # Estratificado: com 5.000 imagens, um sorteio simples poderia deixar
        # uma classe sub-representada e enviesar a acurácia de validação.
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=VAL_SIZE,
            stratify=labels,
            random_state=SEED,
        )
        train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, train_idx=train_idx, val_idx=val_idx)

    # Checagem barata que pega um arquivo corrompido ou de outro split.
    assert len(train_idx) == 45_000 and len(val_idx) == VAL_SIZE, "split inesperado"
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "treino e validação se sobrepõem"
    return train_idx, val_idx


def _build_transforms(img_size: int, augment: bool):
    aug = []
    if augment:
        # Crop e flip são feitos ANTES do resize: assim a augmentation é a
        # mesma (deslocamento de até 4 px na escala original) para 32 e 224,
        # e custa menos CPU do que fazer padding numa imagem de 224x224.
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    resize = [transforms.Resize(img_size, antialias=True)] if img_size != 32 else []
    return transforms.Compose(
        aug + resize + [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )


def _default_workers() -> int:
    # No Windows/Jupyter, workers com spawn costumam travar; no Colab (Linux,
    # 2 vCPUs) 2 workers já mantêm a GPU ocupada.
    if sys.platform.startswith("win"):
        return 0
    return min(4, os.cpu_count() or 1)


def get_loaders(batch_size: int = DEFAULTS["batch_size"], img_size: int = 32,
                augment: bool = True, num_workers: int | None = None) -> dict:
    """Devolve {"train", "val", "test", "meta"}.

    O teste existe aqui só para que o notebook 04 use exatamente o mesmo
    pré-processamento; nenhuma decisão de modelo pode olhar para ele.
    """
    train_idx, val_idx = build_split()
    num_workers = _default_workers() if num_workers is None else num_workers

    # Duas instâncias da mesma base de treino: uma com augmentation e outra
    # sem. Um Subset sobre uma única instância herdaria a augmentation também
    # na validação, o que deixaria a métrica de seleção ruidosa.
    train_full = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                  transform=_build_transforms(img_size, augment))
    eval_full = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=_build_transforms(img_size, False))
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                transform=_build_transforms(img_size, False))

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
    )
    generator = torch.Generator().manual_seed(SEED)
    return {
        "train": DataLoader(Subset(train_full, train_idx), shuffle=True,
                            generator=generator, **common),
        "val": DataLoader(Subset(eval_full, val_idx), shuffle=False, **common),
        "test": DataLoader(test_set, shuffle=False, **common),
        "meta": {"batch_size": batch_size, "img_size": img_size, "augment": augment},
    }


# ---------------------------------------------------------------------------
# Modelos próprios
# ---------------------------------------------------------------------------
def _conv_bn_relu(in_c: int, out_c: int, k: int, use_bn: bool, relu: bool = True):
    # Com BN logo depois, o bias da conv é redundante (BN subtrai a média).
    layers = [nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return layers


class ConvBlock(nn.Module):
    """[Conv -> BN? -> ReLU] x n_convs -> MaxPool."""

    def __init__(self, in_c, out_c, kernel_size, n_convs, use_bn):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers += _conv_bn_relu(in_c if i == 0 else out_c, out_c, kernel_size, use_bn)
        self.body = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.body(x))


class ResidualBlock(nn.Module):
    """Mesmo bloco, mas com atalho somado antes da última ReLU.

    O atalho usa projeção 1x1 quando o número de canais muda (sempre que os
    filtros dobram), porque a soma exige tensores do mesmo formato.
    """

    def __init__(self, in_c, out_c, kernel_size, n_convs, use_bn):
        super().__init__()
        layers = []
        for i in range(n_convs):
            last = i == n_convs - 1
            # A última conv não tem ReLU: ativar depois da soma é o que permite
            # ao bloco aprender um resíduo negativo (formulação da ResNet).
            layers += _conv_bn_relu(in_c if i == 0 else out_c, out_c, kernel_size,
                                    use_bn, relu=not last)
        self.body = nn.Sequential(*layers)
        if in_c != out_c:
            self.shortcut = nn.Sequential(*_conv_bn_relu(in_c, out_c, 1, use_bn, relu=False))
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.body(x) + self.shortcut(x)))


class BaselineCNN(nn.Module):
    """CNN parametrizada: cada ablação é a troca de UM argumento.

    Todos os argumentos têm o valor da baseline como padrão, então
    `BaselineCNN(use_bn=False)` é exatamente "baseline sem BN" e nada mais.
    """

    block_cls = ConvBlock
    model_type = "baseline"

    # PROPOSTA — confirmar com o grupo: os padrões abaixo SÃO a baseline.
    # Alterá-los depois que as ablações rodarem invalida a comparação.
    def __init__(self, blocks: int = 3, base_filters: int = 32, kernel_size: int = 3,
                 n_convs: int = 2, use_bn: bool = True, dropout: float = 0.3,
                 fc_dim: int | None = 256, global_pool: bool = False, img_size: int = 32):
        super().__init__()
        # Cada bloco reduz a resolução pela metade; além disso o mapa vira 0x0.
        if img_size // 2**blocks < 1:
            raise ValueError(f"{blocks} blocos reduzem {img_size}px a menos de 1px")
        # Guardado para o JSON e para reconstruir o modelo no notebook 04.
        self.config = dict(type=self.model_type, blocks=blocks, base_filters=base_filters,
                           kernel_size=kernel_size, n_convs=n_convs, use_bn=use_bn,
                           dropout=dropout, fc_dim=fc_dim, global_pool=global_pool,
                           img_size=img_size)

        stages, in_c = [], 3
        for b in range(blocks):
            # Dobrar filtros a cada pooling mantém o custo por camada parecido
            # enquanto a resolução cai: troca-se informação espacial por canais.
            out_c = base_filters * 2**b
            stages.append(self.block_cls(in_c, out_c, kernel_size, n_convs, use_bn))
            in_c = out_c
        self.features = nn.Sequential(*stages)

        if global_pool:
            # GAP elimina a maior parte dos parâmetros da cabeça (que no
            # flatten cresce com a resolução) e age como regularizador.
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = in_c
        else:
            self.pool = nn.Identity()
            feat_dim = in_c * (img_size // 2**blocks) ** 2

        # Dropout só na cabeça: é onde está a maior parte dos parâmetros e o
        # risco de overfitting; nas convs o BN já regulariza.
        if fc_dim:
            head = [nn.Dropout(dropout), nn.Linear(feat_dim, fc_dim), nn.ReLU(inplace=True),
                    nn.Dropout(dropout), nn.Linear(fc_dim, NUM_CLASSES)]
        else:
            head = [nn.Dropout(dropout), nn.Linear(feat_dim, NUM_CLASSES)]
        self.classifier = nn.Sequential(nn.Flatten(), *head)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class ResidualCNN(BaselineCNN):
    """Mesma interface da BaselineCNN, trocando só o tipo de bloco.

    Herdar garante que "residual vs. não residual" seja a única diferença
    entre as duas: cabeça, filtros e pooling são o mesmo código.
    """

    block_cls = ResidualBlock
    model_type = "residual"


def build_cnn(residual: bool = False, **kwargs) -> BaselineCNN:
    """Permite tratar "residual" como mais um argumento de ablação."""
    return (ResidualCNN if residual else BaselineCNN)(**kwargs)


# ---------------------------------------------------------------------------
# Transfer learning
# ---------------------------------------------------------------------------
def _replace_head(model, name):
    """Troca a última camada por uma de 10 saídas e devolve essa camada."""
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model.fc
    if name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        return model.classifier
    # vgg16 (classifier[6]), mobilenet_v2 e efficientnet_b0 (classifier[1])
    idx = len(model.classifier) - 1
    model.classifier[idx] = nn.Linear(model.classifier[idx].in_features, NUM_CLASSES)
    return model.classifier[idx]


# Partes descongeladas na estratégia "partial": o último estágio de cada rede
# mais o classificador inteiro. Os estágios finais codificam características
# específicas do ImageNet (partes de objetos), as que mais precisam se adaptar;
# os iniciais (bordas, texturas) são genéricos e podem ficar fixos.
_PARTIAL = {
    "resnet18": lambda m: [m.layer4, m.fc],
    "resnet50": lambda m: [m.layer4, m.fc],
    "vgg16": lambda m: [m.features[24:], m.classifier],  # último bloco conv5
    "densenet121": lambda m: [m.features.denseblock4, m.features.norm5, m.classifier],
    "mobilenet_v2": lambda m: [m.features[-3:], m.classifier],
    "efficientnet_b0": lambda m: [m.features[-2:], m.classifier],
}

_ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
}
STRATEGIES = ("frozen", "partial", "full")


def build_pretrained(name: str, strategy: str = "full", pretrained: bool = True) -> nn.Module:
    """Rede do torchvision com pesos do ImageNet e cabeça de 10 classes.

    frozen  = só a nova camada final treina (mede o quanto as features do
              ImageNet já separam o CIFAR-10 sem adaptação);
    partial = último estágio + classificador;
    full    = fine-tuning completo.
    """
    if name not in _ARCHS:
        raise ValueError(f"arquitetura desconhecida: {name}; opções: {list(_ARCHS)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy}; opções: {STRATEGIES}")
    if not pretrained and strategy != "full":
        warnings.warn("pretrained=False com backbone congelado treina sobre pesos aleatórios")

    builder, weights = _ARCHS[name]
    model = builder(weights=weights if pretrained else None)
    head = _replace_head(model, name)

    if strategy != "full":
        for p in model.parameters():
            p.requires_grad = False
        trainable = [head] if strategy == "frozen" else _PARTIAL[name](model)
        for module in trainable:
            for p in module.parameters():
                p.requires_grad = True

    model.config = dict(type="pretrained", arch=name, strategy=strategy, pretrained=pretrained)
    return model


def build_from_config(config: dict) -> nn.Module:
    """Reconstrói o modelo a partir do bloco "model" de um JSON de resultado."""
    cfg = dict(config)
    kind = cfg.pop("type")
    if kind == "pretrained":
        return build_pretrained(cfg["arch"], cfg["strategy"], cfg["pretrained"])
    return build_cnn(residual=(kind == "residual"), **cfg)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, treináveis). Os dois importam: em "frozen" o modelo é grande,
    mas o custo de treino é proporcional só aos treináveis."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _freeze_frozen_batchnorm(model: nn.Module) -> None:
    # requires_grad=False não impede o BN de atualizar running_mean/var em
    # modo train(). Sem isto, a estratégia "frozen" alteraria o backbone
    # mesmo sem gradiente, e deixaria de ser um backbone congelado.
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                m.eval()


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------
def _make_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                               momentum=0.9, nesterov=True)
    raise ValueError(f"otimizador desconhecido: {name}")


def _make_scheduler(name, optimizer, epochs):
    if name in (None, "none"):
        return None
    if name == "cosine":
        # Decai o lr suavemente até ~0 no fim: as últimas épocas refinam em
        # vez de oscilar, sem precisar escolher marcos como no StepLR.
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"scheduler desconhecido: {name}")


@torch.no_grad()
def evaluate(model, loader, device=None, criterion=None):
    """Perda, acurácia, predições e rótulos sobre um loader."""
    device = device or get_device()
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / len(targets), float((preds == targets).mean()), preds, targets


@torch.no_grad()
def measure_inference_time(model, input_shape=(3, 32, 32), device=None,
                           batch_size: int = 1, n_warmup: int = 10, n_runs: int = 100) -> float:
    """Tempo médio de inferência em milissegundos por imagem.

    As primeiras passadas são descartadas porque incluem inicialização de
    kernels CUDA e alocação de memória, que não fazem parte do custo real.
    As chamadas CUDA são assíncronas: sem synchronize(), o cronômetro mediria
    só o enfileiramento das operações, não a execução.
    """
    device = device or get_device()
    model.eval().to(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(n_warmup):
        model(x)
    sync()
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    sync()
    return (time.perf_counter() - start) / (n_runs * batch_size) * 1000


def _result_path(run_name):
    return RESULTS_DIR / f"{run_name}.json"


def _checkpoint_path(run_name):
    return CHECKPOINT_DIR / f"{run_name}.pth"


def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_path(result["run_name"])
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result(run_name: str) -> dict:
    return json.loads(_result_path(run_name).read_text(encoding="utf-8"))


def result_exists(run_name: str) -> bool:
    """Útil no Colab: permite retomar uma lista de runs após a sessão cair."""
    return _result_path(run_name).exists()


def list_results(prefix: str = "") -> list[str]:
    return sorted(p.stem for p in RESULTS_DIR.glob(f"{prefix}*.json"))


def load_best_weights(model: nn.Module, run_name: str, device=None) -> nn.Module:
    device = device or get_device()
    state = torch.load(_checkpoint_path(run_name), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def fit(model: nn.Module, loaders: dict, run_name: str,
        epochs: int = DEFAULTS["epochs"], lr: float = DEFAULTS["lr"],
        weight_decay: float = DEFAULTS["weight_decay"],
        optimizer: str = DEFAULTS["optimizer"], scheduler: str | None = DEFAULTS["scheduler"],
        device=None, verbose: bool = True) -> dict:
    """Treina, mantém a melhor época pela VALIDAÇÃO e salva o JSON da run.

    Ao final o modelo devolvido está com os pesos da melhor época, não da
    última: com o cosine o último epoch costuma ser bom, mas em configurações
    que sofrem overfitting (ex.: sem dropout) a última pode ser bem pior.
    """
    # Re-semear aqui faz a ordem dos batches não depender de quantas runs
    # vieram antes no mesmo notebook.
    set_seed()
    device = device or get_device()
    model = model.to(device)
    use_amp = device.type == "cuda"

    # Só parâmetros treináveis vão para o otimizador; os congelados ainda
    # sofreriam weight decay no AdamW se estivessem lá.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _make_optimizer(optimizer, params, lr, weight_decay)
    sched = _make_scheduler(scheduler, opt, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = _checkpoint_path(run_name)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s")}
    best_acc, best_epoch = -1.0, -1
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        _freeze_frozen_batchnorm(model)
        run_loss, correct, seen = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            # AMP quase dobra a vazão nas GPUs do Colab (T4) sem efeito
            # relevante na acurácia; isso é o que torna o transfer em 224px viável.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], device, criterion)
        history["train_loss"].append(run_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(opt.param_groups[0]["lr"])
        history["epoch_time_s"].append(time.perf_counter() - t0)
        if sched is not None:
            sched.step()

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt)
        if verbose:
            print(f"[{run_name}] ep {epoch:3d}/{epochs} | "
                  f"train {history['train_loss'][-1]:.4f}/{history['train_acc'][-1]:.4f} | "
                  f"val {val_loss:.4f}/{val_acc:.4f}"
                  f"{'  *' if best_epoch == epoch else ''}")

    train_time = time.perf_counter() - t_start
    load_best_weights(model, run_name, device)

    total, trainable = count_parameters(model)
    meta = loaders.get("meta", {})
    img_size = meta.get("img_size", 32)
    result = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name() if use_amp else str(device),
        "model": getattr(model, "config", {"type": type(model).__name__}),
        "hparams": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                    "optimizer": optimizer, "scheduler": scheduler, "seed": SEED, **meta},
        "params_total": total,
        "params_trainable": trainable,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "inference_ms_per_image": measure_inference_time(model, (3, img_size, img_size), device),
        "history": history,
        # Preenchidos só no notebook 04, depois que a escolha já foi feita.
        "test_acc": None,
        "confusion_matrix": None,
    }
    save_result(result)
    if verbose:
        print(f"[{run_name}] melhor val_acc={best_acc:.4f} (época {best_epoch}), "
              f"{train_time / 60:.1f} min, {trainable:,}/{total:,} parâmetros treináveis")
    return result


def evaluate_on_test(model: nn.Module, loaders: dict, result: dict, device=None) -> dict:
    """Acurácia e matriz de confusão no conjunto de TESTE.

    Chame apenas para configurações já escolhidas pela validação. Se o teste
    influenciar qualquer escolha, ele deixa de ser uma estimativa honesta do
    desempenho em dados novos e vira mais um conjunto de validação.
    """
    _, acc, preds, targets = evaluate(model, loaders["test"], device)
    result["test_acc"] = acc
    result["confusion_matrix"] = confusion_matrix(
        targets, preds, labels=list(range(NUM_CLASSES))).tolist()
    save_result(result)
    return result
