"""
Resa "neurale" dell'immagine del broncoscopio, alternativa a quella procedurale di
endoscope_camera.py (texture dei vasi + luce sulla punta).

IDEA
  La telecamera simulata calcola gia' una mappa di profondita' (distanza delle pareti per ogni
  pixel). Il modello di broncho-sim2real (https://github.com/stefanasandei/broncho-sim2real,
  codice in third_party/broncho_sim2real) e' stato addestrato su broncoscopie VERE a trasformare
  una mappa di profondita' in un'immagine RGB realistica: colori, riflessi, muco, vasi e
  sfocature li "inventa" la rete in modo coerente con la geometria.

COME FUNZIONA IL MODELLO (flow matching)
  E' un modello generativo: parte da rumore casuale x0 e lo trasforma passo dopo passo
  nell'immagine. La rete (una U-Net da ~100 milioni di parametri) non predice direttamente
  l'immagine ma una "velocita'" v(x, t, profondita'): di quanto e in che direzione spostare ogni
  pixel al tempo t (da 0 = rumore a 1 = immagine). L'immagine si ottiene integrando
  dx/dt = v da t = 0 a t = 1:
    - Euler: x <- x + dt * v(x, t)                          (1 valutazione della rete per passo)
    - Heun:  come Euler, poi si corregge con la media tra la velocita' iniziale e quella nel punto
             proposto                                         (2 valutazioni per passo, piu' preciso)
  La repo originale usa 100 passi di Heun (199 valutazioni della rete, ~6 s per immagine sulla
  RTX 3050). Con pochi passi di Euler (4-8) si scende a 0.1-0.3 s per immagine, con immagini un po'
  piu' "morbide".

OTTIMIZZAZIONI CUDA (misurate su RTX 3050 Laptop, 256x256 px, una valutazione della rete):
  fp32                                        65 ms
  bf16 (tensor core, come in addestramento)   46 ms   <- autocast bf16
  + torch.compile (fusione dei kernel)        32 ms   <- usato
  Provate e scartate perche' non aiutano su questa rete/GPU:
  - cudnn.benchmark: stesso tempo, ma triplica la memoria (610 MB invece di ~270 MB) cercando gli
    algoritmi, e la GPU da 4 GB e' condivisa con SOFA e con la telecamera;
  - CUDA graph: < 3% (la rete e' limitata dal calcolo, non dal costo di lancio dei kernel);
  - torch.compile "max-autotune": nessun guadagno e oltre 3 minuti di compilazione;
  - channels_last: piu' lento.
  Inoltre: i pesi delle convoluzioni sono convertiti in bf16 una volta sola (niente conversione a
  ogni passo), il rumore iniziale e' allocato una volta sola sulla GPU, la GPU non aspetta mai la
  CPU durante l'integrazione (una sola sincronizzazione alla fine, quando si copia l'immagine).
  La codifica della profondita' resta sulla CPU con il codice originale: costa ~6 ms (2-5% del
  totale) e cosi' l'input della rete e' identico, bit per bit, a quello dell'addestramento.

COERENZA TRA FOTOGRAMMI
  Un modello generativo con rumore diverso a ogni immagine produrrebbe un "tremolio" (dettagli che
  cambiano da un fotogramma all'altro). Qui il rumore iniziale e' FISSO (scelto da `seed`): a
  geometria simile corrispondono immagini simili. reseed() lo cambia (es. a ogni episodio).

PESI
  Il checkpoint addestrato (best.pt) non e' ancora pubblicato dall'autore. Il file va scaricato a
  parte (quando disponibile) e indicato in NEURAL_CHECKPOINT (scene_maze.py). Il caricamento
  rifiuta checkpoint addestrati con una codifica della profondita' diversa da quella attesa.

USO
    renderer = NeuralBronchoscopeRenderer("best.pt", resolution=256, steps=8)
    rgb = renderer.render(depth, valid)        # depth (H, W) in metri; rgb uint8 (256, 256, 3)

  Modo asincrono (per la finestra di runSofa: la simulazione non aspetta la rete):
    renderer = NeuralBronchoscopeRenderer(..., asynchronous=True)
    renderer.submit(depth, valid)              # non blocca
    rgb = renderer.latest()                    # ultima immagine pronta (o None)
"""

import os
import sys
import threading
import time

import numpy as np
import torch

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "broncho_sim2real")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
from depth_encoding import brea_geometry_image   # noqa: E402  (codice originale, non modificato)
from unet import FlowUNet                         # noqa: E402

# Protocollo di codifica della profondita' con cui deve essere stato addestrato il checkpoint
# (copiato da utils.conditioning_metadata() della repo originale)
CONDITIONING = {"protocol": "brea_continuous_near_bright_128_v2", "mode": "continuous",
                "native_size": 128, "polarity": "near_bright",
                "percentiles": [0.02, 0.98], "resize": "bicubic_antialias",
                "rgb_resize": "area_down_bicubic_up"}


class NeuralBronchoscopeRenderer:
    """Profondita' -> immagine realistica di broncoscopio (vedi la docstring del modulo).

    Parametri:
      checkpoint    percorso di best.pt (broncho-sim2real)
      resolution    lato dell'immagine generata [px] (addestramento: 256; multiplo di 16)
      steps         passi di integrazione (100 = qualita' piena della repo, 4-8 = veloce)
      solver        "euler" (1 valutazione per passo) o "heun" (2 per passo, piu' preciso)
      seed          seme del rumore iniziale (fisso: immagini coerenti nel tempo)
      compile       torch.compile della rete (-30% di tempo; la prima volta ~1 minuto di
                    compilazione, poi in cache su disco)
      weights       "ema" (media mobile dei pesi, consigliata) o "model"
      asynchronous  True = la rete gira in un thread separato (submit()/latest())
      device        "cuda" (consigliato) o "cpu" (molto lento)
    """

    def __init__(self, checkpoint, resolution=256, steps=8, solver="euler", seed=0, compile=True,
                 weights="ema", asynchronous=False, device=None):
        if solver not in ("euler", "heun"):
            raise ValueError(f"solver sconosciuto: {solver!r} (usa 'euler' o 'heun')")
        if steps <= 0:
            raise ValueError("steps deve essere positivo")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.resolution = resolution
        self.steps = steps
        self.solver = solver
        self.use_compile = compile and self.device.type == "cuda"

        self.model = self._load(checkpoint, weights)
        multiple = self.model.spatial_multiple
        if resolution % multiple:
            raise ValueError(f"la risoluzione deve essere un multiplo di {multiple}")

        # tempi di integrazione t_k = k / steps, preparati una volta sola sulla GPU
        self.times = [torch.full((1,), k / steps, device=self.device) for k in range(steps)]
        self.noise = None
        self.reseed(seed)

        # stato del modo asincrono
        self.asynchronous = asynchronous
        self._cond = threading.Condition()
        self._request = None           # (depth, valid) in attesa di essere elaborata
        self._result = None            # ultima immagine pronta
        self._ready = threading.Event()
        self.last_time = None          # durata dell'ultima generazione [s]
        self._forward = None

        if asynchronous:
            # tutto (compilazione compresa) avviene nel thread di lavoro: torch.compile va usato
            # sempre dallo stesso thread
            threading.Thread(target=self._worker, daemon=True).start()
        else:
            self._prepare()

    # --- Caricamento -------------------------------------------------------------
    def _load(self, path, weights):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"checkpoint di broncho-sim2real non trovato: {path!r}. I pesi (best.pt) non sono "
                "ancora pubblicati dall'autore: vanno scaricati a parte e indicati in "
                "NEURAL_CHECKPOINT (scene_maze.py), oppure usa CAMERA_RENDERER = \"procedural\".")
        # weights_only=True: carica solo tensori e tipi semplici, niente codice arbitrario;
        # mmap=True: non copia tutto il file in RAM prima di usarlo
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state.get("conditioning") != CONDITIONING:
            raise ValueError("il checkpoint e' stato addestrato con una codifica della profondita' "
                             "diversa da quella attesa (brea_continuous_near_bright_128_v2)")
        architecture = dict((state.get("config") or {}).get("model") or {})
        if not architecture:
            raise ValueError("il checkpoint non contiene la configurazione dell'architettura")
        if architecture.pop("architecture", "unet") != "unet":
            raise ValueError("architettura del checkpoint non supportata (attesa: unet)")
        architecture["gradient_checkpointing"] = False   # serve solo in addestramento
        model = FlowUNet(**architecture)
        model.load_state_dict(state[weights])
        model = model.to(self.device).eval().requires_grad_(False)

        if self.device.type == "cuda":
            # pesi di convoluzioni e strati lineari in bf16 una volta per tutte: autocast non
            # deve riconvertirli a ogni valutazione. Le GroupNorm restano in float32 (autocast le
            # esegue comunque in float32, come in addestramento)
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    module.to(torch.bfloat16)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[neurale] broncho-sim2real caricato: {n_params:.0f}M parametri, pesi '{weights}', "
              f"{self.device}")
        self.random_weights = bool(state.get("random", False))
        if self.random_weights:
            print("[neurale] ATTENZIONE: pesi CASUALI (make_random_checkpoint), non addestrati: "
                  "l'immagine sara' rumore colorato. Sostituiscili con best.pt in NEURAL_CHECKPOINT.")
        return model

    def _prepare(self):
        """Prepara la funzione che valuta la rete (compilata se richiesto) e la "scalda"."""
        def velocity(x, t, depth):
            # bf16 sui tensor core; cache_enabled=False: niente cache dei pesi convertiti (sono
            # gia' bf16) e compatibilita' con torch.compile
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda", cache_enabled=False):
                return self.model(x, t, depth)

        if self.use_compile and not os.path.basename(sys.executable).lower().startswith("python"):
            # dentro runSofa sys.executable e' runSofa stesso: torch.compile non deve lanciare i
            # suoi processi di compilazione paralleli (rilancerebbe runSofa), compila qui
            import torch._inductor.config as inductor_config
            inductor_config.compile_threads = 1
        self._forward = torch.compile(velocity) if self.use_compile else velocity
        if self.use_compile:
            print("[neurale] compilazione della rete con torch.compile (la prima volta ~1 minuto, "
                  "poi e' in cache)...")
        start = time.perf_counter()
        # "riscaldamento": una generazione completa su una profondita' di prova, cosi' compilazione e
        # inizializzazioni una tantum (~4 s alla prima immagine) avvengono adesso e non a
        # simulazione avviata
        coords = (np.arange(self.resolution) + 0.5) / self.resolution * 2.0 - 1.0
        radius = np.sqrt(coords[None, :] ** 2 + coords[:, None] ** 2)
        self.render((0.01 + 0.02 * np.clip(1.0 - radius, 0.0, 1.0)).astype(np.float32), radius <= 1.0)
        print(f"[neurale] pronto ({time.perf_counter() - start:.1f} s): {self.steps} passi "
              f"{self.solver}, {self.resolution}x{self.resolution} px")

    def reseed(self, seed):
        """Nuovo rumore iniziale (es. a ogni episodio). Lo stesso rumore = immagini coerenti."""
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.noise = torch.randn((1, 3, self.resolution, self.resolution), device=self.device,
                                 generator=generator)

    # --- Codifica della profondita' ----------------------------------------------------
    def encode(self, depth, valid=None):
        """Profondita' metrica (H, W) [m] -> condizionamento (1, 1, S, S) in [-1, 1] sulla GPU.

        Usa la funzione originale (brea_geometry_image): i pixel non validi (fuori dal cerchio
        dell'ottica o senza pareti) vengono riempiti con il pixel valido piu' vicino, poi le
        distanze vengono trasformate nella distribuzione di "disparita'" (vicino = chiaro) su cui
        la rete e' stata addestrata. Contano solo i RANGHI delle distanze, non i metri: la scala
        assoluta della scena non influisce."""
        encoded = brea_geometry_image(np.asarray(depth, dtype=np.float32), valid, size=self.resolution)
        # conversione in [-1, 1] sulla CPU come nella repo originale (sulla GPU la divisione arrotonda
        # diversamente nell'ultima cifra); poi un'unica copia di 256 kB verso la GPU
        tensor = torch.from_numpy(encoded).float().div(127.5).sub(1.0)
        return tensor[None, None].to(self.device)

    # --- Generazione --------------------------------------------------------------------
    @torch.inference_mode()
    def render(self, depth, valid=None):
        """Genera l'immagine (uint8 (S, S, 3)). Se non c'e' nessuna parete valida (lente dentro il
        materiale) restituisce un'immagine nera."""
        if self._forward is None:
            self._prepare()
        start = time.perf_counter()
        try:
            condition = self.encode(depth, valid)
        except ValueError:
            return np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        image = self.noise.clone()
        dt = 1.0 / self.steps
        for k in range(self.steps):
            # stesso schema di utils.sample_flow della repo originale
            velocity = self._forward(image, self.times[k], condition).float()
            proposal = image + dt * velocity
            if self.solver == "heun" and k < self.steps - 1:
                next_velocity = self._forward(proposal, self.times[k + 1], condition).float()
                proposal = image + 0.5 * dt * (velocity + next_velocity)
            image = proposal

        # [-1, 1] -> [0, 255]; la conversione avviene sulla GPU, alla CPU arrivano solo 196 kB
        rgb = image[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(1, 2, 0)
        result = rgb.cpu().numpy()   # unica sincronizzazione CPU-GPU
        self.last_time = time.perf_counter() - start
        return result

    # --- Modo asincrono -------------------------------------------------------------------
    def submit(self, depth, valid=None):
        """Chiede una nuova immagine senza aspettarla. Se la rete e' ancora occupata con la
        richiesta precedente, quella in attesa viene sostituita da questa (conta solo l'ultima)."""
        with self._cond:
            self._request = (np.array(depth, dtype=np.float32, copy=True),
                             None if valid is None else np.array(valid, dtype=bool, copy=True))
            self._cond.notify()

    def latest(self):
        """Ultima immagine generata (uint8 (S, S, 3)), o None se non ce n'e' ancora nessuna."""
        return self._result

    def wait(self, timeout=None):
        """Aspetta la prima immagine del modo asincrono (utile nei test)."""
        return self._ready.wait(timeout)

    def _worker(self):
        self._prepare()
        while True:
            with self._cond:
                while self._request is None:
                    self._cond.wait()
                depth, valid = self._request
                self._request = None
            self._result = self.render(depth, valid)
            self._ready.set()


def fisheye_valid_mask(depth, max_distance):
    """Pixel validi di un'immagine fisheye di endoscope_camera: dentro il cerchio dell'ottica e con
    una parete colpita (la telecamera assegna max_distance ai pixel senza parete e fuori cerchio)."""
    size = depth.shape[0]
    coords = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    inside = coords[None, :] ** 2 + coords[:, None] ** 2 <= 1.0
    return inside & (depth < 0.999 * max_distance)


def apply_circle(rgb):
    """Annerisce gli angoli fuori dal cerchio dell'ottica, come l'immagine procedurale."""
    size = rgb.shape[0]
    coords = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    outside = coords[None, :] ** 2 + coords[:, None] ** 2 > 1.0
    out = rgb.copy()
    out[outside] = 0
    return out


# =============================================================================
# Pesi casuali (segnaposto finche' non sono disponibili quelli addestrati)
# =============================================================================
# Architettura della repo originale (config/unet.yaml)
DEFAULT_ARCHITECTURE = dict(architecture="unet", in_channels=3, condition_channels=1, base_channels=160,
                            channel_mult=[1, 2, 3, 4], blocks_per_level=2, attention_head_dim=64,
                            detail_channels=32, dropout=0.0, gradient_checkpointing=False)


def make_random_checkpoint(path, seed=0):
    """Scrive un checkpoint con la stessa struttura di best.pt ma pesi CASUALI.

    Serve solo a far funzionare tutta la catena (caricamento, compilazione, finestra, training)
    finche' non si hanno i pesi veri: la rete genera rumore colorato, non immagini realistiche.
    I pesi sono salvati in bf16 (~200 MB invece di ~400 MB); al caricamento tornano float32."""
    torch.manual_seed(seed)
    architecture = dict(DEFAULT_ARCHITECTURE)
    architecture.pop("architecture")
    model = FlowUNet(**architecture)
    with torch.no_grad():
        # la rete originale inizializza a zero alcuni strati (es. l'ultimo): la velocita' sarebbe
        # sempre nulla e l'uscita sarebbe il rumore iniziale. Qui tutti i pesi sono casuali, con
        # ampiezza ~1/sqrt(ingressi) perche' i valori restino di ordine 1 attraverso la rete
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.normal_(0.0, 0.5 / parameter[0].numel() ** 0.5)
    weights = {name: value.to(torch.bfloat16) if value.is_floating_point() else value
               for name, value in model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": weights, "ema": weights, "epoch": 0, "step": 0,
                "config": {"model": DEFAULT_ARCHITECTURE}, "conditioning": CONDITIONING,
                "metrics": {}, "random": True}, path)
    return path


if __name__ == "__main__":
    # python neural_render.py [percorso]  -> crea il file di pesi casuali (default weights/random_weights.pt)
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights", "random_weights.pt")
    print(f"pesi casuali scritti in {make_random_checkpoint(target)}")
