# helper.py
import matplotlib.pyplot as plt

# Try to use IPython's display utilities when available (e.g., VS Code / Jupyter)
try:
    from IPython import display
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False

# Enable interactive mode so plots update without blocking
plt.ion()

def plot(scores, mean_scores, save_path: str | None = None):
    """
    Live-plot training progress.

    Args:
        scores (list[float|int]): episode scores in order.
        mean_scores (list[float|int]): running mean values (e.g., last 10).
        save_path (str|None): optional file path to save the current figure.
    """
    if _HAS_IPY:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Games")
    plt.ylabel("Score")

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean (recent)")

    # Make axes sensible even at the start of training
    ymax = 10
    if scores:
        ymax = max(ymax, max(scores[-100:]))  # scale to recent range
    if mean_scores:
        ymax = max(ymax, max(mean_scores[-100:]))
    plt.ylim(0, ymax * 1.1)

    plt.xlim(0, max(len(scores), 1))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Annotate latest values
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")

    if save_path:
        plt.savefig(save_path, dpi=120)

    # Render without blocking
    plt.show(block=False)
    plt.pause(0.1)


def moving_average(values, window: int = 10):
    """Simple moving average utility (optional)."""
    if window <= 0:
        raise ValueError("window must be > 0")
    if not values:
        return []
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
            out.append(s / window)
        else:
            out.append(s / (i + 1))
    return out
