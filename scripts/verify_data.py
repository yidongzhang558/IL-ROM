"""Verify the Case 1 data files against their recorded size and SHA-256."""
from pathlib import Path
import hashlib
import json
import stat

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'data'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    results = []
    for item in manifest['data_files']:
        path = base / item['path']
        errors = []
        if not path.is_file():
            errors.append('missing')
        else:
            info = path.lstat()
            if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024):
                errors.append('linked file, expected a physical copy')
            if info.st_nlink > 1:
                errors.append('hard linked file, expected an independent copy')
            if info.st_size != item['bytes']:
                errors.append('size mismatch')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                errors.append('SHA-256 mismatch')
        results.append(dict(path=item['path'], ok=not errors, errors=errors))
    ok = all(item['ok'] for item in results)
    print(json.dumps(dict(ok=ok, count=len(results), bytes=sum(i['bytes'] for i in manifest['data_files']), files=results), indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
