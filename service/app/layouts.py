from typing import List, Dict, Tuple
from .utils import mm_to_px

def grid_layout_pixels(
    dpi: int,
    rows: int,
    cols: int,
    page_margins_mm: Dict[str, float],
    box_mm: float,
    gap_mm: float
) -> List[Tuple[int,int,int,int]]:
    """Restituisce una lista di bbox [x,y,w,h] in pixel (riga-major)."""
    top = mm_to_px(page_margins_mm.get("top", 20), dpi)
    left = mm_to_px(page_margins_mm.get("left", 20), dpi)
    right = mm_to_px(page_margins_mm.get("right", 20), dpi)
    bottom = mm_to_px(page_margins_mm.get("bottom", 20), dpi)

    box = mm_to_px(box_mm, dpi)
    gap = mm_to_px(gap_mm, dpi)

    # Calcolo progressivo, nessun centraggio sofisticato per semplicità
    bboxes = []
    x0 = left
    y0 = top
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (box + gap)
            y = y0 + r * (box + gap)
            bboxes.append((x, y, box, box))
    return bboxes
