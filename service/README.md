# KanjiNin Image Service (Essential)

Servizio **FastAPI** per ricevere **immagini da camera** (foto del foglio con risposte), normalizzarle in **A4 @ 300 DPI**, 
estrarre i riquadri secondo un **layout a griglia** e valutarli con un **modello PyTorch** fornito dall'utente.

## TL;DR
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Avvio (porta predefinita 8088)
./run.sh
# Oppure
uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 1
```

Endpoint:
- `GET /healthz` → ok
- `POST /evaluate` (multipart/form-data)
  - `doc` (file immagine: jpg/png)
  - `kanji_list` (text) → JSON: array di oggetti, almeno `kanji` (es. come kanjitest2.json)
  - `options` (text, opzionale) → JSON, ad es.:
    ```json
    {
      "dpi": 300,
      "grid": {"rows": 6, "cols": 7},
      "page_margins_mm": {"top": 20, "left": 20, "right": 20, "bottom": 20},
      "box_mm": 30,
      "gap_mm": 10,
      "return_crops": true
    }
    ```

**Output 200 (JSON)**
```json
{
  "session_id": "20251112-113045-7f2c",
  "source": {"type": "image", "dpi": 300, "normalized": "A4"},
  "summary": {
    "total_boxes": 42,
    "matched": 40,
    "accuracy_top1": 0.925,
    "avg_quality": 0.81
  },
  "items": [
    {
      "id":"p1-b03","page":1,"bbox":[180,420,354,354],
      "expected_kanji":"漢","predicted_kanji":"漢",
      "confidence":0.97,"quality":0.84,
      "assets":{"crop_relpath":"sessions/.../crops/p1-b03.png"}
    }
  ]
}
```

> **Nota:** Copia il tuo modello in `models/model.pth` e la mappa etichette in `models/label_map.json`.
> La label map deve essere un `{chr: class_index}` o `{class_index: chr}` coerente con il modello.

## Opzioni principali
- **Imaging**
  - `dpi` (default 300): risoluzione target dopo omografia su A4.
  - `return_crops` (bool): se `true`, salva i ritagli in `sessions/<id>/crops` e restituisce i path relativi.
- **Layout**
  - `grid.rows`, `grid.cols` (default 6x7 → 42 box).
  - `box_mm` (default 30): lato del quadrato di risposta.
  - `gap_mm` (default 10): spazio tra box.
  - `page_margins_mm` (default 20 mm su tutti i lati).

## Modello
- Inserisci:
  - `models/model.pth`
  - `models/label_map.json` (può essere `{"漢": 123, "字": 124}` o l'inverso; il servizio si adatta)
- Se non presenti, il servizio esegue comunque ma restituisce predizioni **mock** (utile per test end-to-end).

## Sicurezza & Limiti
- Limite 20 MB per upload (configurabile).
- Solo immagini `image/jpeg` o `image/png`.
- Timeout di elaborazione consigliato a livello di reverse proxy.

## Licenza
MIT
