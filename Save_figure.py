import os

def save_figure_pdf(fig, filename: str, folder: str = "Figures") -> None:
    """
    Save a matplotlib figure as a PDF file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Name of the file (without extension).
    folder : str, optional
        Folder where the figure will be saved, by default "figures".
    """

    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Full path
    filepath = os.path.join(folder, f"{filename}.pdf")

    # Save figure
    fig.savefig(filepath, format="pdf", bbox_inches="tight")

    print(f"Figure saved at: {filepath}")