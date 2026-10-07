import ctypes
import os

class WindowsKeyboard:
    """Legge lo stato dei tasti con GetAsyncKeyState (API di Windows).

    Serve perche' con la GUI di runSofa v26.06 (GLFW/ImGui) i KeypressedEvent non arrivano ai
    controller Python. I tasti vengono considerati solo quando la finestra in primo piano
    appartiene a questo processo (cioe' a runSofa), cosi' non si reagisce a quello che si scrive
    in altre applicazioni."""

    def __init__(self):
        self.user32 = ctypes.WinDLL("user32")
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p,
                                                         ctypes.POINTER(ctypes.c_ulong)]
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.pid = os.getpid()

    def has_focus(self):
        hwnd = self.user32.GetForegroundWindow()
        if not hwnd:
            return False
        pid = ctypes.c_ulong()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value == self.pid

    def is_down(self, key):
        """key: lettera maiuscola, es. "W" (il codice virtuale di Windows coincide con l'ASCII)."""
        return bool(self.user32.GetAsyncKeyState(ord(key)) & 0x8000)
