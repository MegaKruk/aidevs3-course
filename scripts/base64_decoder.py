import base64, itertools

CIPH = "GhUiPj1fKTM3NCY1KSUmNxkP"
KEY  = "ANDRZEJ".encode()

if __name__=="__main__":
    data = base64.b64decode(CIPH)
    pt = bytes(c ^ k for c, k in zip(data, itertools.cycle(KEY)))
    print(pt)
