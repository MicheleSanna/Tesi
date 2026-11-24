"""
Finestra dell'endoscopio, in un processo separato da runSofa.

Perche' un processo separato: OpenCV disegna le finestre con una propria copia di Qt, che nello
stesso processo di runSofa entra in conflitto con le librerie della GUI di SOFA (GLFW) e lo fa
andare in crash. Qui invece Qt vive da solo; runSofa (EndoscopeController in controller.py)
lancia questo script e gli manda le immagini attraverso una pipe (lo stdin di questo processo).

PROTOCOLLO (molto semplice): per ogni immagine
    8 byte  = altezza e larghezza (due interi senza segno a 32 bit, little-endian)
    H*W*3 byte = pixel BGR uint8, riga per riga

Lo script termina quando runSofa chiude la pipe (fine simulazione) o quando si chiude la finestra.
Non serve lanciarlo a mano.
"""

import struct
import sys

import cv2
import numpy as np

WINDOW = "Endoscopio"


def read_exact(stream, n):
    """Legge esattamente n byte (una pipe puo' restituirne meno per volta). None = pipe chiusa."""
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def main():
    stream = sys.stdin.buffer
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while True:
        header = read_exact(stream, 8)
        if header is None:
            break                                   # runSofa ha chiuso la pipe
        height, width = struct.unpack("<II", header)
        pixels = read_exact(stream, height * width * 3)
        if pixels is None:
            break
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        cv2.imshow(WINDOW, image)
        cv2.waitKey(1)                              # fa ridisegnare la finestra
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break                                   # finestra chiusa dall'utente
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
