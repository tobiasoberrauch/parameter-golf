#!/usr/bin/env python3
"""Compress a Python script into a self-extracting LZMA+base85 wrapper.

Usage: python compress.py train_gpt.py > train_gpt_compressed.py
"""
import base64
import lzma
import sys

def compress_script(input_path: str) -> str:
    source = open(input_path, 'r', encoding='utf-8').read()
    compressed = lzma.compress(
        source.encode('utf-8'),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
    )
    encoded = base64.b85encode(compressed).decode('ascii')
    wrapper = (
        'import lzma as L,base64 as B;'
        f'exec(L.decompress(B.b85decode("{encoded}"),'
        'format=L.FORMAT_RAW,filters=[{"id":L.FILTER_LZMA2}]))'
    )
    return wrapper

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <script.py>", file=sys.stderr)
        sys.exit(1)
    print(compress_script(sys.argv[1]))
