# -*- coding: utf-8 -*-
"""Desktop GUI for converting images to compact WEBP files."""

import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Optional

import rawpy
from PIL import Image
from pillow_heif import register_heif_opener
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# Enable HEIC/HEIF support for Pillow
register_heif_opener()

# Target size
TARGET_SIZE_KB = 300
TARGET_SIZE_BYTES = TARGET_SIZE_KB * 1024

# Quality levels to try
QUALITY_LEVELS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]

# Resize parameters
SCALE_STEP = 0.8  # each step scales the long side by 80%
MIN_LONG_SIDE = 1200  # do not scale below this long side; only adjust quality afterwards

LogFunc = Callable[[str], None]


def log_to_console(message: str) -> None:
    """Default logger that prints to stdout."""
    print(message, flush=True)


def is_image_file(path: Path) -> bool:
    """Check that the extension matches a supported image format."""
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".dng",
        ".arw",  # RAW (DNG, Sony ARW)
    }


def open_image_any_format(input_path: Path) -> Image.Image:
    """
    Open an image of various formats.
    RAW (DNG, ARW) is handled via rawpy, others via Pillow.
    """
    ext = input_path.suffix.lower()

    if ext in {".dng", ".arw"}:
        with rawpy.imread(str(input_path)) as raw:
            rgb = raw.postprocess()
        return Image.fromarray(rgb, "RGB")

    return Image.open(input_path)


def compress_image_to_webp(
    input_path: Path, output_path: Path, log_func: LogFunc = log_to_console
) -> str:
    """
    Compress a single file into WEBP with step-by-step logging.
    - Adjusts quality to reach the target size.
    - If the file is still too large, progressively scales down the resolution.
    - Writes to a temporary file before an atomic rename to the target.
    """

    def log(msg: str) -> None:
        log_func(msg)

    try:
        if output_path.exists():
            return f"[SKIP] {input_path.name} -> {output_path.name} (exists)"

        log(f"   - Загружаю изображение: {input_path.name}")
        base_img = open_image_any_format(input_path)

        if base_img.mode not in ("RGB", "RGBA"):
            log("   - Привожу изображение к RGB")
            base_img = base_img.convert("RGB")

        has_alpha = base_img.mode == "RGBA"

        base_w, base_h = base_img.size
        log(f"   - Размер исходного файла: {base_w}x{base_h}")

        scale = 1.0
        iteration = 0

        final_bytes = None
        final_quality = None
        final_size = None
        final_size_px = (base_w, base_h)

        while True:
            iteration += 1
            log(f"   - Итерация #{iteration}")

            # Resize if needed
            if scale < 1.0:
                new_w = max(1, int(base_w * scale))
                new_h = max(1, int(base_h * scale))
                log(f"     · Уменьшаю размер: {new_w}x{new_h}")
                img = base_img.resize((new_w, new_h), Image.LANCZOS)
            else:
                img = base_img
                log("     · Использую оригинальный размер")

            log("     · Подбираю качество...")

            best_bytes = None
            best_quality = None

            for q in QUALITY_LEVELS:
                buffer = BytesIO()
                save_kwargs = {
                    "format": "WEBP",
                    "quality": q,
                    "method": 6,
                }
                if has_alpha:
                    save_kwargs["lossless"] = False

                img.save(buffer, **save_kwargs)
                data = buffer.getvalue()
                size = len(data)

                size_kb = size // 1024
                log(f"       q={q}: {size_kb}KB")

                if size <= TARGET_SIZE_BYTES:
                    log(f"       ✓ Вписалось! ({size_kb}KB <= {TARGET_SIZE_KB}KB)")
                    best_bytes = data
                    best_quality = q
                    break

                if best_bytes is None or size < len(best_bytes):
                    best_bytes = data
                    best_quality = q

            final_bytes = best_bytes
            final_quality = best_quality
            final_size = len(best_bytes)
            final_size_px = img.size

            long_side = max(img.size)

            log(
                f"     · Лучший результат: Q={final_quality}, "
                f"{final_size // 1024}KB, {final_size_px[0]}x{final_size_px[1]}"
            )

            if final_size <= TARGET_SIZE_BYTES:
                log("     ✓ Файл вписался в лимит — завершаю подбор.\n")
                break

            if long_side <= MIN_LONG_SIDE:
                log("     ! Достигнут минимальный размер — дальше качество только ухудшится.\n")
                break

            scale *= SCALE_STEP
            log(f"     ↘ Размер > {TARGET_SIZE_KB}KB, уменьшаю масштаб до {round(scale, 3)}\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

        with open(tmp_output_path, "wb") as f:
            f.write(final_bytes)

        tmp_output_path.replace(output_path)

        return (
            f"[OK] {input_path.name} -> {output_path.name}  "
            f"(Q={final_quality}, {final_size // 1024}KB, "
            f"{final_size_px[0]}x{final_size_px[1]})"
        )

    except Exception as e:
        return f"[ERROR] {input_path}: {e}"


def collect_files(input_dir: Path) -> Iterable[Path]:
    """Recursively gather supported files."""
    return [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h} ч {m} мин {s} сек"
    if m > 0:
        return f"{m} мин {s} сек"
    return f"{s} сек"


def process_folder(
    input_dir: Path,
    output_dir: Path,
    log_func: LogFunc = log_to_console,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
) -> None:
    """Process all supported images in a folder tree."""
    all_files = collect_files(input_dir)
    total_found = len(all_files)

    if total_found == 0:
        log_func("Нет изображений для обработки.")
        if progress_callback:
            progress_callback(0, 0, 0.0, "")
        return

    todo = []
    already_done = 0
    for src in all_files:
        rel = src.relative_to(input_dir)
        dst = (output_dir / rel).with_suffix(".webp")
        if dst.exists():
            already_done += 1
        else:
            todo.append((src, dst))

    total_todo = len(todo)

    log_func(f"Всего найдено файлов:           {total_found}")
    log_func(f"Уже обработано ранее:           {already_done}")
    log_func(f"Осталось обработать в этот раз: {total_todo}")
    log_func(f"Входная папка:   {input_dir}")
    log_func(f"Выходная папка:  {output_dir}")
    log_func("Структура подпапок будет сохранена.\n")

    if total_todo == 0:
        log_func("Все файлы уже обработаны, делать больше нечего.")
        if progress_callback:
            progress_callback(total_found, total_found, 0.0, "")
        return

    start_time = time.time()
    done = 0

    try:
        for idx, (src, dst) in enumerate(todo, start=1):
            rel_path = src.relative_to(input_dir)
            log_func(f">> [{idx}/{total_todo}] Обработка: {rel_path}")

            msg = compress_image_to_webp(src, dst, log_func=log_func)
            done += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / max(done, 1)
            remaining = (total_todo - done) * avg_time

            log_func(msg)
            log_func("-" * 80)

            if progress_callback:
                progress_callback(done, total_todo, remaining, str(rel_path))

    except KeyboardInterrupt:
        log_func("\n\nОстановка по запросу пользователя (Ctrl+C).")
        log_func(
            "Уже готовые .webp-файлы сохранены и будут пропущены при следующем запуске."
        )

    total_time = time.time() - start_time
    log_func("\nИтог:")
    log_func(f"Обработано в этом запуске: {done} из {total_todo}")
    log_func(f"Затраченное время:         {format_time(total_time)}")

    if done < total_todo:
        log_func(
            "При следующем запуске с теми же параметрами скрипт продолжит с оставшихся файлов.\n"
        )
    else:
        log_func("Все запланированные файлы обработаны.\n")

    if progress_callback:
        progress_callback(done, total_todo, 0.0, "")


class CompressionApp:
    """Simple Tkinter GUI to manage folder selection and compression."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Сжатие изображений в WEBP")
        self.root.geometry("820x600")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Готов к работе")
        self.current_file_var = tk.StringVar(value="Файл: —")

        self._worker: Optional[threading.Thread] = None

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 12, "pady": 8}

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        accent = "#3b82f6"
        bg = "#f5f7fb"
        self.root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg)
        style.configure("Header.TLabel", font=("Inter", 18, "bold"), background=bg)
        style.configure("Subheader.TLabel", font=("Inter", 11), foreground="#4b5563", background=bg)
        style.configure("Accent.TButton", font=("Inter", 11, "bold"), padding=6)
        style.configure("TEntry", padding=6)
        style.configure(
            "Custom.Horizontal.TProgressbar",
            thickness=14,
            troughcolor="#e5e7eb",
            background=accent,
            bordercolor="#e5e7eb",
            lightcolor=accent,
            darkcolor=accent,
        )

        header = ttk.Frame(self.root)
        header.pack(fill="x", **padding)
        ttk.Label(header, text="Сжатие изображений в WEBP", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text="Быстро уменьшайте фотографии до ~300KB без лишних действий.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 4))

        input_frame = ttk.LabelFrame(self.root, text="Входная папка")
        input_frame.pack(fill="x", **padding)

        ttk.Entry(input_frame, textvariable=self.input_var).pack(
            side="left", fill="x", expand=True, padx=(10, 5), pady=10
        )
        ttk.Button(
            input_frame,
            text="Выбрать папку",
            command=self.select_input,
            style="Accent.TButton",
        ).pack(side="left", padx=10, pady=10)

        output_frame = ttk.LabelFrame(self.root, text="Папка для сохранения")
        output_frame.pack(fill="x", **padding)

        ttk.Entry(output_frame, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=(10, 5), pady=10
        )
        ttk.Button(
            output_frame,
            text="Куда складывать",
            command=self.select_output,
            style="Accent.TButton",
        ).pack(side="left", padx=10, pady=10)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", **padding)

        self.progress = ttk.Progressbar(
            control_frame,
            orient="horizontal",
            mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", expand=True, side="left")

        ttk.Button(
            control_frame,
            text="Начать сжатие",
            command=self.start_processing,
            style="Accent.TButton",
        ).pack(side="left", padx=(10, 0))

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", **padding)
        ttk.Label(status_frame, textvariable=self.status_var, font=("Inter", 11, "bold")).pack(
            anchor="w"
        )
        ttk.Label(status_frame, textvariable=self.current_file_var, style="Subheader.TLabel").pack(
            anchor="w"
        )

        log_frame = ttk.LabelFrame(self.root, text="Ход работы")
        log_frame.pack(fill="both", expand=True, **padding)

        self.log_text = ScrolledText(log_frame, height=18, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def select_input(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self.input_var.set(folder)
            if not self.output_var.get():
                default_output = Path(folder).parent / "compressed_webp"
                self.output_var.set(str(default_output))

    def select_output(self) -> None:
        folder = filedialog.askdirectory(title="Куда сохранять WEBP")
        if folder:
            self.output_var.set(folder)

    def log(self, message: str) -> None:
        def append() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, append)

    def update_progress(self, done: int, total: int, remaining: float, current: str) -> None:
        def _update() -> None:
            percent = (done / total * 100) if total else 0
            self.progress['value'] = percent
            eta = format_time(remaining) if remaining else "—"
            self.status_var.set(f"Готово: {done}/{total} | Осталось: {eta}")
            self.current_file_var.set(f"Файл: {current if current else '—'}")

        self.root.after(0, _update)

    def start_processing(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("В процессе", "Сжатие уже выполняется.")
            return

        input_dir = Path(self.input_var.get()).expanduser()
        output_dir_input = self.output_var.get().strip()
        output_dir = Path(output_dir_input).expanduser() if output_dir_input else None

        if not input_dir.is_dir():
            messagebox.showerror("Ошибка", "Укажите корректную входную папку.")
            return

        if output_dir is None:
            output_dir = input_dir.parent / "compressed_webp"

        self.status_var.set("Запускаю сжатие...")
        self.current_file_var.set("Файл: —")
        self.progress['value'] = 0
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        def worker() -> None:
            try:
                process_folder(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    log_func=self.log,
                    progress_callback=self.update_progress,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.log(f"Ошибка: {exc}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Ошибка", f"Во время сжатия произошла ошибка: {exc}"
                    ),
                )
            finally:
                self.root.after(0, lambda: self.status_var.set("Готово"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()


def run_cli(argv: list[str]) -> None:
    """Run the legacy CLI when arguments are provided."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Сжать все изображения в папке до WEBP ~300KB (однопоточно,"
            " с прогрессом и возобновлением)."
        )
    )
    parser.add_argument(
        "input_folder",
        help="Папка с исходными фотографиями (например, photos)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        help="Папка для сохранения (по умолчанию: compressed_webp рядом с входной)",
        default=None,
    )

    args = parser.parse_args(argv)

    input_dir = Path(args.input_folder).resolve()
    if not input_dir.is_dir():
        print("Указанная папка не существует или это не папка.")
        raise SystemExit(1)

    output_dir = (
        Path(args.output_folder).resolve()
        if args.output_folder
        else input_dir.parent / "compressed_webp"
    )

    process_folder(input_dir=input_dir, output_dir=output_dir)


def run_gui() -> None:
    root = tk.Tk()
    app = CompressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])
    else:
        run_gui()
