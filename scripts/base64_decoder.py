import base64, itertools, re

CIPH = "GhUiPj1fKTM3NCY1KSUmNxkP"
KEY  = "ANDRZEJ".encode()

if __name__=="__main__":
    data = base64.b64decode(CIPH)
    pt = bytes(c ^ k for c, k in zip(data, itertools.cycle(KEY)))
    print(pt)
    clean = ''.join(chr(b) for b in pt if 32 <= b < 127)  # drukowalne ASCII
    m = re.search(r'flg[:\s\[{]*(\w+)', clean, re.I)
    flag = m.group(1).upper()
    print(flag)
