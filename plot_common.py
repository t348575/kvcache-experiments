import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def configure_plots() -> None:
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def token_k_formatter(x, _: object) -> str:
    return f"{x / 1024:.0f}k"


def plain_number_formatter(x, _: object) -> str:
    return f"{x:g}"


def thousands_formatter(x, _: object) -> str:
    return f"{x:,.0f}"


def save_figure(path: str, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def set_log_y_axis(ax, base: int = 10) -> None:
    ax.set_yscale("log", base=base)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=base, subs=[1], numticks=10))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(plain_number_formatter))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
