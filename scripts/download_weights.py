import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.request


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, size, sha256):
    actual_size = os.path.getsize(path)
    if actual_size != size:
        return f"size mismatch: got {actual_size}, expected {size}"
    digest = sha256_of(path)
    if digest != sha256:
        return f"sha256 mismatch: got {digest}, expected {sha256}"
    return None


def download(url, dst, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dst, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", required=True,
                        help="источник; можно указать несколько, пробуются по порядку")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--local", default=None, help="готовый файл, если он есть")
    parser.add_argument("--retries", type=int, default=2, help="попыток на источник")
    parser.add_argument("--timeout", type=int, default=60, help="таймаут сокета, с")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"

    if args.local and os.path.isfile(args.local):
        print(f"[weights] using local file {args.local}", flush=True)
        shutil.copyfile(args.local, tmp)
        error = verify(tmp, args.size, args.sha256)
        if error:
            sys.exit(f"[weights] local file is broken: {error}")
    else:
        error = "no sources"
        for url in args.url:
            for attempt in range(1, args.retries + 1):
                print(f"[weights] {url} (attempt {attempt})", flush=True)
                start = time.time()
                try:
                    download(url, tmp, args.timeout)
                except Exception as e:
                    error = str(e)
                    print(f"[weights]   error after {time.time() - start:.0f}s: {e}", flush=True)
                    continue
                error = verify(tmp, args.size, args.sha256)
                if error is None:
                    print(f"[weights]   downloaded in {time.time() - start:.1f}s", flush=True)
                    break
                print(f"[weights]   {error}", flush=True)
            if error is None:
                break
        if error is not None:
            sys.exit(
                f"[weights] download failed ({error}). Download scaled_offline.pth manually "
                "(see checkpoints/README.md), put it into ./checkpoints/ and rebuild."
            )

    os.replace(tmp, args.out)
    print(f"[weights] OK {args.out} size={args.size} sha256={args.sha256}", flush=True)


if __name__ == "__main__":
    main()
