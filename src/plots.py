"""Figuras e tabelas a partir dos JSONs em results/.

Tudo aqui lê os resultados salvos, nunca os modelos: assim as figuras do
relatório podem ser refeitas sem retreinar nada.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import CLASSES, FIGURES_DIR, RESULTS_DIR, load_result


def _save(fig, name: str):
    # PDF é vetorial: o texto e as linhas ficam nítidos em qualquer zoom no
    # relatório LaTeX, ao contrário de PNG.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_curves(run_names: list[str] | str, name: str | None = None):
    """Perda e acurácia por época, treino (tracejado) vs. validação (sólido).

    Treino e validação no mesmo eixo e na mesma cor tornam o overfitting
    visível de imediato: a distância entre as duas linhas é o gap.
    """
    if isinstance(run_names, str):
        run_names = [run_names]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, run in enumerate(run_names):
        h = load_result(run)["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[i % len(colors)]
        ax_loss.plot(epochs, h["train_loss"], "--", color=c, alpha=0.7)
        ax_loss.plot(epochs, h["val_loss"], "-", color=c, label=run)
        ax_acc.plot(epochs, h["train_acc"], "--", color=c, alpha=0.7)
        ax_acc.plot(epochs, h["val_acc"], "-", color=c, label=run)
    ax_loss.set(xlabel="Época", ylabel="Perda (entropia cruzada)", title="Perda")
    ax_acc.set(xlabel="Época", ylabel="Acurácia", title="Acurácia")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
    # Legenda única: cor = run; estilo = conjunto.
    handles = [plt.Line2D([], [], color="gray", ls="--", label="treino"),
               plt.Line2D([], [], color="gray", ls="-", label="validação")]
    ax_acc.legend(handles=ax_acc.get_legend_handles_labels()[0] + handles, fontsize=8)
    fig.tight_layout()
    _save(fig, name or "curvas_" + "_".join(run_names)[:80])
    return fig


def plot_confusion(run_name: str):
    """Matriz de confusão normalizada por linha (classe verdadeira).

    Normalizar por linha transforma cada célula em "fração da classe real
    prevista como X", que é comparável entre classes mesmo se o teste não
    fosse balanceado.
    """
    cm = _confusion(run_name)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predita", ylabel="Verdadeira", title=f"Matriz de confusão — {run_name}")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save(fig, f"confusao_{run_name}")
    return fig


def _confusion(run_name: str) -> np.ndarray:
    cm = load_result(run_name).get("confusion_matrix")
    if cm is None:
        raise ValueError(f"{run_name} ainda não foi avaliado no teste (notebook 04)")
    return np.array(cm)


def comparison_table(run_names: list[str], name: str = "comparacao") -> pd.DataFrame:
    """Tabela comparativa; também exporta results/<name>.tex para o relatório."""
    rows = []
    for run in run_names:
        r = load_result(run)
        rows.append({
            "Configuração": run,
            "Val acc (%)": 100 * r["best_val_acc"],
            "Test acc (%)": 100 * r["test_acc"] if r.get("test_acc") is not None else np.nan,
            "Parâmetros (M)": r["params_total"] / 1e6,
            "Treináveis (M)": r["params_trainable"] / 1e6,
            "Treino (min)": r["train_time_s"] / 60,
            "Inferência (ms/img)": r["inference_ms_per_image"],
        })
    df = pd.DataFrame(rows).set_index("Configuração")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_latex(RESULTS_DIR / f"{name}.tex", float_format="%.2f", na_rep="--",
                caption="Comparação entre configurações", label=f"tab:{name}")
    return df


def per_class_accuracy(run_name: str) -> pd.DataFrame:
    """Acurácia por classe e com qual classe cada uma é mais confundida.

    A acurácia global esconde erros concentrados (ex.: cat <-> dog); olhar o
    maior valor fora da diagonal mostra onde o modelo realmente falha.
    """
    cm = _confusion(run_name)
    support = cm.sum(axis=1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    worst = off.argmax(axis=1)
    return pd.DataFrame({
        "Acurácia (%)": 100 * np.diag(cm) / support,
        "Mais confundida com": [CLASSES[j] for j in worst],
        "Fração confundida (%)": 100 * off[np.arange(len(CLASSES)), worst] / support,
    }, index=pd.Index(CLASSES, name="Classe")).sort_values("Acurácia (%)")
