import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

from face_analyzer import detect_face, detect_eyes_mediapipe

# Cascade for live face mesh on camera feed
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
from glasses_generator import generate_glasses_image

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#0f0f0f"
PANEL   = "#1a1a1a"
ACCENT  = "#00d4aa"
ACCENT2 = "#ff6b35"
TEXT    = "#ffffff"
SUBTEXT = "#aaaaaa"
BTN_BG  = "#1e1e1e"

WINDOW_W = 1000
WINDOW_H = 680
CAM_W    = 560
CAM_H    = 420


class GlassesApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GlassMatch — AI Glasses Advisor")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.cap           = None
        self.running       = False
        self._last_frame   = None
        self._raw_frame    = None
        self._face_rect    = None
        self._eyes         = []
        self.face_shape    = None
        self.result_frame  = None
        self.try_count     = 0
        self.current_style = None
        self.current_specs = None
        self._after_id     = None

        self._build_ui()
        self._start_camera()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="👓  GlassMatch",
                 font=("Helvetica", 18, "bold"), fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr, text="AI-Powered Glasses Advisor",
                 font=("Helvetica", 10), fg=SUBTEXT, bg=BG).pack(side="left", padx=14)

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # Left: camera
        left = tk.Frame(main, bg=PANEL, width=CAM_W, height=CAM_H + 50)
        left.pack(side="left", padx=(0, 16))
        left.pack_propagate(False)
        self.cam_label = tk.Label(left, bg="#000000", width=CAM_W, height=CAM_H)
        self.cam_label.pack(padx=4, pady=4)
        self.status_var = tk.StringVar(value="📷  Position your face in the frame")
        tk.Label(left, textvariable=self.status_var,
                 font=("Helvetica", 10), fg=ACCENT, bg=PANEL).pack(pady=(0, 6))

        # Right: info + controls
        right = tk.Frame(main, bg=BG, width=380)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)

        tk.Label(right, text="Face Status", font=("Helvetica", 10), fg=SUBTEXT, bg=BG).pack(anchor="w")
        self.shape_var = tk.StringVar(value="Waiting")
        tk.Label(right, textvariable=self.shape_var,
                 font=("Helvetica", 22, "bold"), fg=ACCENT, bg=BG).pack(anchor="w")

        tk.Label(right, text="Overlay Status", font=("Helvetica", 10),
                 fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(8, 0))
        self.style_var = tk.StringVar(value="Ready")
        tk.Label(right, textvariable=self.style_var,
                 font=("Helvetica", 18, "bold"), fg=TEXT, bg=BG).pack(anchor="w")

        # AI message box
        msg_frame = tk.Frame(right, bg=PANEL)
        msg_frame.pack(fill="x", pady=10)
        self.msg_label = tk.Label(
            msg_frame,
            text="Scan your face — the app sends your photo to OpenAI and creates the glasses directly.",
            font=("Helvetica", 10), fg=SUBTEXT, bg=PANEL,
            wraplength=340, justify="left"
        )
        self.msg_label.pack(padx=20, pady=10, anchor="w")

        # Result thumbnail
        self.result_label = tk.Label(right, bg=BG)
        self.result_label.pack(pady=(0, 8))

        # Buttons
        btn = tk.Frame(right, bg=BG)
        btn.pack(fill="x")

        self.scan_btn = self._btn(btn, "📸  Scan My Face",        ACCENT,  "#000000", self._scan_face)
        self.scan_btn.pack(fill="x", pady=(0, 8))

        self.try_btn = self._btn(btn, "🔄  Try More (AI Redesign)", BTN_BG, ACCENT,   self._try_more)
        self.try_btn.pack(fill="x", pady=(0, 8))
        self.try_btn.config(state="disabled")

        self.show_btn = self._btn(btn, "🛍️  Show to Shopkeeper",   ACCENT2, "#ffffff", self._show_shopkeeper)
        self.show_btn.pack(fill="x")
        self.show_btn.config(state="disabled")

        tk.Button(right, text="↩  Reset", font=("Helvetica", 9),
                  fg=SUBTEXT, bg=BG, bd=0, cursor="hand2",
                  command=self._reset).pack(pady=(10, 0))

    def _btn(self, parent, text, bg, fg, cmd):
        return tk.Button(parent, text=text, font=("Helvetica", 11, "bold"),
                         bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                         bd=0, padx=12, pady=10, cursor="hand2", command=cmd)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            return
        self.running = True
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self._last_frame = frame.copy()
            display = cv2.resize(frame, (CAM_W, CAM_H))
            display = self._draw_face_mesh(display)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.cam_label.configure(image=img)
            self.cam_label.image = img
        self._after_id = self.after(30, self._update_frame)

    def _draw_face_mesh(self, frame):
        """Draws sci-fi face mesh dots and lines on live camera feed."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        CYAN  = (0, 255, 180)    # main mesh color
        CYAN2 = (0, 180, 120)    # secondary lines
        DOT   = (0, 255, 220)    # landmark dots

        for (fx, fy, fw, fh) in faces:
            cx = fx + fw // 2
            cy = fy + fh // 2

            # ── Landmark points (approximate face geometry) ──
            landmarks = [
                # forehead top
                (cx, fy + int(fh * 0.05)),
                # forehead left/right
                (fx + int(fw * 0.2),  fy + int(fh * 0.12)),
                (fx + int(fw * 0.8),  fy + int(fh * 0.12)),
                # temple left/right
                (fx + int(fw * 0.05), fy + int(fh * 0.35)),
                (fx + int(fw * 0.95), fy + int(fh * 0.35)),
                # eye left center
                (fx + int(fw * 0.28), fy + int(fh * 0.38)),
                # eye right center
                (fx + int(fw * 0.72), fy + int(fh * 0.38)),
                # nose top
                (cx, fy + int(fh * 0.45)),
                # nose bottom
                (cx, fy + int(fh * 0.58)),
                # nose left/right
                (fx + int(fw * 0.38), fy + int(fh * 0.58)),
                (fx + int(fw * 0.62), fy + int(fh * 0.58)),
                # cheek left/right
                (fx + int(fw * 0.08), fy + int(fh * 0.58)),
                (fx + int(fw * 0.92), fy + int(fh * 0.58)),
                # mouth left/right
                (fx + int(fw * 0.32), fy + int(fh * 0.72)),
                (fx + int(fw * 0.68), fy + int(fh * 0.72)),
                # mouth center
                (cx, fy + int(fh * 0.74)),
                # jaw left/right
                (fx + int(fw * 0.15), fy + int(fh * 0.82)),
                (fx + int(fw * 0.85), fy + int(fh * 0.82)),
                # chin
                (cx, fy + int(fh * 0.97)),
            ]

            # ── Mesh connection lines ──
            connections = [
                (0, 1), (0, 2),           # forehead top to sides
                (1, 3), (2, 4),           # forehead to temples
                (1, 5), (2, 6),           # forehead to eyes
                (3, 5), (4, 6),           # temples to eyes
                (5, 7), (6, 7),           # eyes to nose top
                (7, 8),                   # nose bridge
                (8, 9), (8, 10),          # nose bottom
                (3, 11), (4, 12),         # temples to cheeks
                (11, 16), (12, 17),       # cheeks to jaw
                (13, 15), (14, 15),       # mouth corners to center
                (16, 18), (17, 18),       # jaw to chin
                (9, 13), (10, 14),        # nose to mouth
                (5, 9), (6, 10),          # eyes to nose sides
            ]

            # Draw lines
            for (i, j) in connections:
                pt1, pt2 = landmarks[i], landmarks[j]
                cv2.line(frame, pt1, pt2, CYAN2, 1, cv2.LINE_AA)

            # Draw dots
            for pt in landmarks:
                cv2.circle(frame, pt, 3, DOT, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, CYAN, 1,  cv2.LINE_AA)

            # ── Outer face oval ──
            cv2.ellipse(frame,
                        (cx, fy + fh // 2),
                        (fw // 2, fh // 2),
                        0, 0, 360, CYAN, 1, cv2.LINE_AA)

            # ── Eye boxes ──
            eye_y  = fy + int(fh * 0.34)
            eye_h  = int(fh * 0.12)
            eye_w  = int(fw * 0.22)
            l_eye_cx = fx + int(fw * 0.28)
            r_eye_cx = fx + int(fw * 0.72)
            for ecx in [l_eye_cx, r_eye_cx]:
                cv2.rectangle(frame,
                              (ecx - eye_w//2, eye_y),
                              (ecx + eye_w//2, eye_y + eye_h),
                              CYAN, 1, cv2.LINE_AA)

            # ── Scan line animation ──
            t = time.time()
            scan_y = fy + int((fh * ((t * 0.8) % 1.0)))
            cv2.line(frame,
                     (fx, scan_y), (fx + fw, scan_y),
                     (0, 255, 200), 1, cv2.LINE_AA)

            # ── Corner brackets on face rect ──
            blen = 14
            thick = 2
            corners = [
                (fx, fy), (fx + fw, fy),
                (fx, fy + fh), (fx + fw, fy + fh)
            ]
            dirs = [(1,1), (-1,1), (1,-1), (-1,-1)]
            for (bx, by), (dx, dy) in zip(corners, dirs):
                cv2.line(frame, (bx, by), (bx + dx*blen, by), CYAN, thick)
                cv2.line(frame, (bx, by), (bx, by + dy*blen), CYAN, thick)

            # ── Label ──
            cv2.putText(frame, "SCANNING...", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

        return frame

    # ── Core Flow ─────────────────────────────────────────────────────────────

    def _scan_face(self):
        if self._last_frame is None:
            messagebox.showwarning("No Frame", "Camera not ready.")
            return
        self.status_var.set("🔍  Detecting face...")
        self.scan_btn.config(state="disabled")
        frame = self._last_frame.copy()
        threading.Thread(target=self._process_scan, args=(frame,), daemon=True).start()

    def _process_scan(self, frame):
        face_rect = detect_face(frame)
        if face_rect is None:
            self.after(0, lambda: self._error("No face detected. Try better lighting."))
            return

        eyes = detect_eyes_mediapipe(frame, face_rect)
        self._face_rect = face_rect
        self._eyes      = eyes
        self._raw_frame = frame

        self.after(0, lambda: self._run_ai(frame, face_rect, eyes))

    def _run_ai(self, frame, face_rect, eyes):
        self.status_var.set("🤖  Creating glasses directly with OpenAI...")

        def worker():
            try:
                result = generate_glasses_image(frame, style="glasses")
                self.result_frame = result
                self.current_style = "openai"
                self.current_specs = None

                self.after(0, lambda: self._show_result(result))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self._error(err_msg))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, result_bgr):
        self.shape_var.set("Detected")
        self.style_var.set("OpenAI Glasses")
        self.msg_label.config(text="Your photo now shows glasses created directly by OpenAI.", fg=TEXT)
        self.status_var.set("✅  OpenAI generated the result successfully!")

        thumb = cv2.resize(result_bgr, (340, 200))
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.result_label.configure(image=photo)
        self.result_label.image = photo

        self.scan_btn.config(state="normal")
        self.try_btn.config(state="normal")
        self.show_btn.config(state="normal")

    def _try_more(self):
        if self._raw_frame is None or self._face_rect is None:
            return
        self.try_btn.config(state="disabled")
        self.show_btn.config(state="disabled")
        self.status_var.set("🤖  OpenAI is repositioning the glasses...")
        self._run_ai(self._raw_frame, self._face_rect, self._eyes)

    def _show_shopkeeper(self):
        if self.result_frame is None:
            return
        ShopkeeperWindow(self, self.result_frame)

    def _error(self, msg):
        self.status_var.set(f"❌  {msg}")
        self.scan_btn.config(state="normal")

    def _reset(self):
        self.result_frame = self._face_rect = None
        self._eyes = []
        self.shape_var.set("Waiting")
        self.style_var.set("Ready")
        self.msg_label.config(
            text="Scan your face — the app sends your photo to OpenAI and creates the glasses directly.",
            fg=SUBTEXT)
        self.result_label.configure(image=""); self.result_label.image = None
        self.try_btn.config(state="disabled"); self.show_btn.config(state="disabled")
        self.status_var.set("📷  Position your face in the frame")

    def _on_close(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        if self.cap:
            self.cap.release()
        self.destroy()


# ── Shopkeeper Window ─────────────────────────────────────────────────────────

class ShopkeeperWindow(tk.Toplevel):
    def __init__(self, parent, result_bgr):
        super().__init__(parent)
        self.title("Show to Shopkeeper")
        self.configure(bg="#000000")
        self.geometry("700x580")
        self.resizable(False, False)

        img_h = 380
        img_w = int(result_bgr.shape[1] * img_h / result_bgr.shape[0])
        display = cv2.cvtColor(cv2.resize(result_bgr, (img_w, img_h)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(display))

        tk.Label(self, image=photo, bg="#000000").pack(pady=(20, 8))
        self.photo = photo

        tk.Label(self,
                 text="Internet glasses have been overlaid on the detected face.",
                 font=("Helvetica", 14, "bold"), fg=ACCENT, bg="#000000").pack(pady=(0, 10))

        tk.Label(self,
                 text="Show this screen to the shopkeeper or save the result.",
                 font=("Helvetica", 11), fg=SUBTEXT, bg="#000000").pack(pady=4)

        tk.Button(self, text="Close", font=("Helvetica", 11, "bold"),
                  bg=ACCENT2, fg="white", bd=0, padx=20, pady=8,
                  cursor="hand2", command=self.destroy).pack(pady=10)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = GlassesApp()
    app.mainloop()