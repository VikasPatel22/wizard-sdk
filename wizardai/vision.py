"""
WizardAI Vision Module
----------------------
Real-time camera access via OpenCV, frame capture and processing utilities,
and optional face / object detection hooks.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .exceptions import CameraNotFoundError, VisionError
from .utils import Logger


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Frame = "Any"   # cv2 frame (numpy.ndarray) – avoid hard numpy import at module level
DetectionResult = Dict[str, Union[str, float, Tuple[int, int, int, int]]]


# ---------------------------------------------------------------------------
# VisionModule
# ---------------------------------------------------------------------------

class VisionModule:
    """Real-time camera access and image processing via OpenCV.

    Provides frame capture, display, saving, and hooks for detection.
    All OpenCV imports are deferred so the rest of WizardAI works even if
    ``opencv-python`` is not installed.

    Example::

        cam = VisionModule(device_id=0)
        cam.open()

        frame = cam.capture_frame()
        cam.save_frame(frame, "snapshot.jpg")

        # Process frames in a loop
        def on_frame(frame):
            faces = cam.detect_faces(frame)
            print(f"Faces: {len(faces)}")

        cam.start_stream(on_frame)
        time.sleep(10)
        cam.stop_stream()
        cam.close()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            device_id: OpenCV camera device index (0 = default webcam).
            width:     Requested frame width in pixels.
            height:    Requested frame height in pixels.
            fps:       Requested frames per second.
            logger:    Optional Logger instance.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = logger or Logger("VisionModule")

        self._cap = None          # cv2.VideoCapture
        self._stream_thread: Optional[threading.Thread] = None
        self._streaming = threading.Event()
        self._face_cascade = None
        self._frame_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Open the camera device.

        Raises:
            CameraNotFoundError: If the device cannot be opened.
        """
        try:
            import cv2
        except ImportError:
            raise VisionError(
                "OpenCV is required for VisionModule. "
                "Install it with: pip install opencv-python"
            )

        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise CameraNotFoundError(self.device_id)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.logger.info(
            f"Camera {self.device_id} opened: {self.width}x{self.height} @ {self.fps}fps"
        )

    def close(self):
        """Release the camera device."""
        self.stop_stream()
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self.logger.info(f"Camera {self.device_id} released.")
        self._cap = None

    def is_open(self) -> bool:
        """Return True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture_frame(self) -> Frame:
        """Capture and return a single frame.

        Returns:
            A numpy ndarray (BGR) representing the captured frame.

        Raises:
            VisionError: If the camera is not open or the read fails.
        """
        if not self.is_open():
            raise VisionError("Camera is not open. Call open() first.")

        ret, frame = self._cap.read()
        if not ret:
            raise VisionError("Failed to capture frame from camera.")
        return frame

    def capture_frames(self, n: int, delay: float = 0.0) -> List[Frame]:
        """Capture *n* frames, with optional *delay* seconds between each.

        Args:
            n:     Number of frames to capture.
            delay: Seconds to wait between captures.

        Returns:
            List of captured frames.
        """
        frames = []
        for _ in range(n):
            frames.append(self.capture_frame())
            if delay > 0:
                time.sleep(delay)
        return frames

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    def save_frame(
        self,
        frame: Frame,
        path: Union[str, Path],
        quality: int = 95,
    ) -> Path:
        """Save *frame* to disk as an image file.

        The format is inferred from the file extension (jpg, png, bmp, …).

        Args:
            frame:   Frame to save.
            path:    Destination file path.
            quality: JPEG quality (1-100), ignored for other formats.

        Returns:
            The resolved Path that was saved.
        """
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        params = []
        if p.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(str(p), frame, params)
        self.logger.debug(f"Frame saved to {p}")
        return p

    def load_image(self, path: Union[str, Path]) -> Frame:
        """Load an image from disk as a BGR frame.

        Args:
            path: Path to the image file.

        Returns:
            Frame (numpy ndarray, BGR).
        """
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")

        frame = cv2.imread(str(path))
        if frame is None:
            raise VisionError(f"Could not load image from: {path}")
        return frame

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def resize_frame(self, frame: Frame, width: int, height: int) -> Frame:
        """Resize *frame* to *width* x *height* pixels."""
        import cv2
        return cv2.resize(frame, (width, height))

    def to_grayscale(self, frame: Frame) -> Frame:
        """Convert a BGR frame to grayscale."""
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def to_rgb(self, frame: Frame) -> Frame:
        """Convert a BGR frame to RGB (for display with matplotlib, PIL, etc.)."""
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def flip(self, frame: Frame, axis: int = 1) -> Frame:
        """Flip a frame.

        Args:
            frame: Input frame.
            axis:  1 = horizontal flip, 0 = vertical, -1 = both.
        """
        import cv2
        return cv2.flip(frame, axis)

    def draw_rectangle(
        self,
        frame: Frame,
        x: int,
        y: int,
        w: int,
        h: int,
        colour: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> Frame:
        """Draw a rectangle on *frame* (in-place).

        Args:
            frame:     Input frame (modified in-place).
            x, y:      Top-left corner.
            w, h:      Width and height.
            colour:    BGR colour tuple.
            thickness: Line thickness in pixels.
        """
        import cv2
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, thickness)
        return frame

    def draw_text(
        self,
        frame: Frame,
        text: str,
        x: int,
        y: int,
        font_scale: float = 0.7,
        colour: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> Frame:
        """Overlay *text* on *frame* at position (*x*, *y*)."""
        import cv2
        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness,
        )
        return frame

    def encode_to_base64(self, frame: Frame, ext: str = ".jpg") -> str:
        """Encode a frame as a base64 string (useful for API payloads).

        Args:
            frame: Input BGR frame.
            ext:   Image format extension ('.jpg', '.png', …).

        Returns:
            Base64-encoded image string.
        """
        import base64
        import cv2
        ret, buf = cv2.imencode(ext, frame)
        if not ret:
            raise VisionError("Frame encoding failed.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        frame: Frame,
        scale_factor: float = 1.1,
        min_neighbours: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[DetectionResult]:
        """Detect faces in *frame* using OpenCV Haar Cascades.

        Requires the ``opencv-python`` package (includes Haar cascade data).

        Args:
            frame:          BGR or grayscale frame to analyse.
            scale_factor:   How much the image size is reduced at each scale.
            min_neighbours: Minimum neighbours required to retain a detection.
            min_size:       Minimum face size in pixels.

        Returns:
            List of dicts, each with keys 'x', 'y', 'w', 'h'.
        """
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")

        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise VisionError("Could not load Haar cascade for face detection.")

        gray = self.to_grayscale(frame) if len(frame.shape) == 3 else frame
        detections = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbours,
            minSize=min_size,
        )

        results: List[DetectionResult] = []
        for x, y, w, h in (detections if len(detections) else []):
            results.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return results

    def annotate_faces(self, frame: Frame) -> Tuple[Frame, List[DetectionResult]]:
        """Detect faces and draw bounding boxes on *frame*.

        Returns:
            Tuple of (annotated frame, list of detection dicts).
        """
        faces = self.detect_faces(frame)
        import copy
        annotated = copy.deepcopy(frame)
        for face in faces:
            self.draw_rectangle(annotated, face["x"], face["y"], face["w"], face["h"])
            self.draw_text(annotated, "Face", face["x"], face["y"] - 10)
        return annotated, faces

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def add_frame_callback(self, callback: Callable[[Frame], None]):
        """Register a callback invoked with each streamed frame.

        Args:
            callback: ``fn(frame)`` called for every captured frame.
        """
        self._frame_callbacks.append(callback)

    def start_stream(
        self,
        callback: Optional[Callable[[Frame], None]] = None,
        show_preview: bool = False,
    ):
        """Start capturing frames in a background thread.

        Args:
            callback:     Optional additional per-frame callback.
            show_preview: If True, display a live preview window (requires a
                          display environment).
        """
        if self._streaming.is_set():
            self.logger.warning("Stream already running.")
            return
        if not self.is_open():
            self.open()
        if callback:
            self.add_frame_callback(callback)

        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(show_preview,),
            daemon=True,
            name="wizardai-vision-stream",
        )
        self._stream_thread.start()
        self.logger.info("Vision stream started.")

    def stop_stream(self):
        """Stop the background streaming thread."""
        if not self._streaming.is_set():
            return
        self._streaming.clear()
        if self._stream_thread:
            self._stream_thread.join(timeout=3.0)
        self.logger.info("Vision stream stopped.")

    def _stream_loop(self, show_preview: bool):
        """Internal streaming loop (runs in a daemon thread)."""
        try:
            import cv2
        except ImportError:
            self.logger.error("OpenCV is required for streaming.")
            return

        interval = 1.0 / self.fps if self.fps > 0 else 0
        while self._streaming.is_set():
            t_start = time.monotonic()
            try:
                frame = self.capture_frame()
                for cb in self._frame_callbacks:
                    try:
                        cb(frame)
                    except Exception as exc:
                        self.logger.error(f"Frame callback error: {exc}")

                if show_preview:
                    cv2.imshow("WizardAI Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._streaming.clear()
                        break
            except VisionError as exc:
                self.logger.error(f"Vision error in stream loop: {exc}")
                break

            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if show_preview:
            import cv2
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        status = "open" if self.is_open() else "closed"
        streaming = "streaming" if self._streaming.is_set() else "idle"
        return f"VisionModule(device={self.device_id}, status={status}, {streaming})"
