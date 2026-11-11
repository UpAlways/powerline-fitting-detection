"""Verify the transferred pretrained files without loading model objects."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    folder = Path(__file__).resolve().parent
    try:
        manifest = json.loads((folder / "pretrained_manifest.json").read_text(encoding="utf-8"))
        records = manifest["weights"]
        names = [record["file"] for record in records]
        if len(names) != 5 or len(set(names)) != 5:
            raise ValueError("expected five distinct weight files")
        if any(not isinstance(name, str) or Path(name).name != name or name in (".", "..") for name in names):
            raise ValueError("invalid weight filename")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Cannot read manifest: {}".format(exc))
        return 1

    passed = 0
    for record in records:
        path = folder / record["file"]
        try:
            size = path.stat().st_size
            if size != record["bytes"]:
                print("FAIL {}: {} bytes, expected {}".format(path.name, size, record["bytes"]))
                continue
            digest = sha256(path)
            expected = record.get("sha256")
            prefix = record.get("sha256_prefix")
            if expected:
                valid = digest == expected.lower()
            elif prefix:
                valid = digest.startswith(prefix.lower())
            else:
                raise KeyError("no expected SHA-256 or prefix")
            if not valid:
                print("FAIL {}: SHA-256 mismatch".format(path.name))
                continue
        except FileNotFoundError:
            print("MISSING {}".format(path.name))
            continue
        except (OSError, KeyError, TypeError, AttributeError) as exc:
            print("FAIL {}: {}".format(path.name, exc))
            continue
        print("OK   {} ({:.1f} MiB){}".format(path.name, size / 1024**2, "" if expected else " [official hash prefix]"))
        passed += 1

    print("Verified {}/{} files.".format(passed, len(records)))
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
