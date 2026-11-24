import ctypes

GLFW_PRESS = 1
GLFW_FOCUSED = 0x00020001


class LinuxKeyboard:
    """Legge lo stato dei tasti dalla finestra GLFW di runSofa (stessa interfaccia di WindowsKeyboard).

    Con la GUI di runSofa v26.06 (GLFW/ImGui) i KeypressedEvent non arrivano ai controller Python.
    Su Linux non c'e' un equivalente di GetAsyncKeyState (sotto Wayland un processo non puo' leggere
    la tastiera globale), ma i controller girano nel thread della GUI, con il contesto OpenGL della
    finestra di runSofa attivo: si usa quindi la libglfw gia' caricata da runSofa per chiedere lo
    stato dei tasti a quella finestra (glfwGetKey). Funziona sia con X11 sia con Wayland, e i tasti
    contano solo quando la finestra di runSofa ha il focus.

    La libreria viene cercata alla prima richiesta, perche' quando si costruisce la scena la
    finestra puo' non esistere ancora. Se non si trova (GUI diversa da GLFW, esecuzione senza
    finestra) i tasti risultano sempre non premuti."""

    def __init__(self):
        self.glfw = None
        self.warned = False

    def _load(self):
        if self.glfw is not None:
            return True
        path = None
        with open("/proc/self/maps") as maps:   # la libglfw caricata dal plugin SofaGLFW
            for line in maps:
                if "libglfw.so" in line:
                    path = line.split()[-1]
                    break
        if path is None:
            if not self.warned:
                print("[tastiera] libglfw non caricata (GUI non GLFW?): comandi da tastiera disattivati")
                self.warned = True
            return False
        glfw = ctypes.CDLL(path)   # gia' caricata: restituisce la stessa istanza usata da runSofa
        glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
        glfw.glfwGetWindowAttrib.argtypes = [ctypes.c_void_p, ctypes.c_int]
        glfw.glfwGetWindowAttrib.restype = ctypes.c_int
        glfw.glfwGetKey.argtypes = [ctypes.c_void_p, ctypes.c_int]
        glfw.glfwGetKey.restype = ctypes.c_int
        self.glfw = glfw
        return True

    def _window(self):
        return self.glfw.glfwGetCurrentContext() if self._load() else None

    def has_focus(self):
        window = self._window()
        return bool(window) and self.glfw.glfwGetWindowAttrib(window, GLFW_FOCUSED) == 1

    def is_down(self, key):
        """key: lettera maiuscola, es. "W" (il codice GLFW delle lettere coincide con l'ASCII)."""
        window = self._window()
        return bool(window) and self.glfw.glfwGetKey(window, ord(key)) == GLFW_PRESS
