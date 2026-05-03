"""Scanned document -> Ollama vision (gemma4) -> plain-text PDF transcriber.

Accepts one or more scanned image files (PNG/JPG/BMP/TIFF) or a scanned PDF,
sends each page directly to a local Ollama instance running a vision-capable
model (default: gemma4), and saves the combined transcription as a plain-text PDF.
"""

import base64
import io
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests
from fpdf import FPDF
from PIL import Image

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4"

TRANSCRIBE_PROMPT = (
    "Transcribe every word of handwritten or printed text visible in this image "
    "exactly as it appears. Output only the transcribed text. "
    "Do not add any commentary, labels, headers, or explanation."
)


# -- Helpers ------------------------------------------------------------------

def _img_to_b64(img):
    """Encode a PIL image to a base64 PNG string for the Ollama API."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_pages(paths):
    """Return a list of (label, PIL.Image) for every page across all paths.

    Images are returned as-is; PDFs are rasterised at 2x scale via pypdfium2.
    """
    pages = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError:
                raise RuntimeError(
                    "pypdfium2 is required for PDF input.\n"
                    "Run:  pip install pypdfium2"
                )
            doc = pdfium.PdfDocument(path)
            for i in range(len(doc)):
                bitmap = doc[i].render(scale=2)
                pages.append((f"{name}  [page {i + 1}]", bitmap.to_pil()))
        else:
            pages.append((name, Image.open(path)))
    return pages


def get_ollama_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return [DEFAULT_MODEL]


def query_ollama_vision(b64_image, model):
    """Send a single base64-encoded image to Ollama and return the transcription."""
    payload = {
        "model": model,
        "prompt": TRANSCRIBE_PROMPT,
        "images": [b64_image],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def build_pdf(text, out_path):
    """Write text to a plain-text PDF at out_path using Helvetica."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe if safe.strip() else " ", wrapmode="CHAR")
    pdf.output(out_path)


# -- GUI ----------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Scan -> Ollama -> PDF")
        self.root.geometry("800x640")
        self.root.minsize(600, 500)

        self._pages = []
        self._result_text = ""

        self._build_ui()
        threading.Thread(target=self._populate_models, daemon=True).start()

    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        # Top bar
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Add Images...", command=self._add_images).pack(side="left")
        ttk.Button(top, text="Add PDF...",    command=self._add_pdf   ).pack(side="left", padx=4)
        ttk.Button(top, text="Clear",         command=self._clear     ).pack(side="left")
        ttk.Label(top, text="Ollama model:").pack(side="left", padx=(20, 4))
        self.model_var   = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, width=26, state="readonly"
        )
        self.model_combo.pack(side="left")

        # Queued pages list
        list_frame = ttk.LabelFrame(self.root, text="Pages queued for processing")
        list_frame.pack(fill="both", expand=False, **pad)
        sb = ttk.Scrollbar(list_frame, orient="vertical")
        self.file_list = tk.Listbox(list_frame, height=7, yscrollcommand=sb.set)
        sb.configure(command=self.file_list.yview)
        self.file_list.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4, padx=(0, 4))

        # Progress
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(prog_frame, textvariable=self.status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", length=240)
        self.progress.pack(side="right", padx=4)

        # Action buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(**pad)
        self.process_btn = ttk.Button(
            btn_frame, text="Transcribe & Save PDF...", command=self._start_processing
        )
        self.process_btn.pack(side="left")
        self.save_btn = ttk.Button(
            btn_frame, text="Save PDF Again...", command=self._save_pdf, state="disabled"
        )
        self.save_btn.pack(side="left", padx=8)

        # Text preview
        text_frame = ttk.LabelFrame(self.root, text="Transcription Preview")
        text_frame.pack(fill="both", expand=True, **pad)
        self.text_box = scrolledtext.ScrolledText(text_frame, wrap="word", state="disabled")
        self.text_box.pack(fill="both", expand=True, padx=4, pady=4)

    # -- File management ------------------------------------------------------

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select scanned images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        self._load_paths(list(paths))

    def _add_pdf(self):
        path = filedialog.askopenfilename(
            title="Select scanned PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if path:
            self._load_paths([path])

    def _load_paths(self, paths):
        if not paths:
            return
        self.status_var.set("Loading...")
        self.root.update_idletasks()
        try:
            new_pages = load_pages(paths)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self.status_var.set("Ready.")
            return
        self._pages.extend(new_pages)
        self._refresh_list()
        self.status_var.set(f"{len(self._pages)} page(s) queued.")

    def _clear(self):
        self._pages.clear()
        self.file_list.delete(0, "end")
        self._set_preview("")
        self._result_text = ""
        self.save_btn.configure(state="disabled")
        self.status_var.set("Ready.")

    def _refresh_list(self):
        self.file_list.delete(0, "end")
        for label, _ in self._pages:
            self.file_list.insert("end", label)

    # -- Model population -----------------------------------------------------

    def _populate_models(self):
        models = get_ollama_models()
        self.root.after(0, lambda: self._set_models(models))

    def _set_models(self, models):
        self.model_combo["values"] = models
        if DEFAULT_MODEL in models:
            self.model_var.set(DEFAULT_MODEL)
        elif models:
            self.model_var.set(models[0])

    # -- Processing -----------------------------------------------------------

    def _start_processing(self):
        if not self._pages:
            messagebox.showwarning("No input", "Add at least one image or PDF first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save transcription PDF as...",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not out_path:
            return

        self.process_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self._result_text = ""
        self._set_preview("")
        self.progress.configure(maximum=len(self._pages), value=0)

        threading.Thread(
            target=self._process_pages, args=(out_path,), daemon=True
        ).start()

    def _process_pages(self, out_path):
        model = self.model_var.get()
        parts = []

        for i, (label, img) in enumerate(self._pages):
            self.root.after(
                0,
                lambda lbl=label, idx=i: self.status_var.set(
                    f"Processing {lbl}  ({idx + 1}/{len(self._pages)})..."
                ),
            )
            try:
                b64  = _img_to_b64(img)
                text = query_ollama_vision(b64, model)
            except Exception as exc:
                text = f"[Error on {label}: {exc}]"
            parts.append(text)
            self.root.after(0, lambda v=i + 1: self.progress.configure(value=v))

        self._result_text = "\n\n".join(parts)
        self.root.after(0, lambda: self._on_done(out_path))

    def _on_done(self, out_path):
        self._set_preview(self._result_text)
        try:
            build_pdf(self._result_text, out_path)
            self.status_var.set(f"Saved -> {os.path.basename(out_path)}")
            messagebox.showinfo("Done", f"Transcription PDF saved to:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))
            self.status_var.set("Transcription complete -- save failed.")
        self.process_btn.configure(state="normal")
        self.save_btn.configure(state="normal")

    # -- Output ---------------------------------------------------------------

    def _set_preview(self, text):
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("1.0", text)
        self.text_box.configure(state="disabled")

    def _save_pdf(self):
        if not self._result_text:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save transcription PDF as...",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not out_path:
            return
        try:
            build_pdf(self._result_text, out_path)
            messagebox.showinfo("Saved", f"PDF saved to:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()