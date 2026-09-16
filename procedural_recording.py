"""Streaming, versioned SQLite recordings. Replay needs no controller/model."""
import json
import sqlite3
import zlib
from pathlib import Path

VERSION = 1

class RaceRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # Never overwrite an existing recording.
        with self.path.open('xb'):
            pass
        self.db = sqlite3.connect(str(self.path))
        self.db.execute('CREATE TABLE metadata (version INTEGER, dt REAL)')
        self.db.execute('INSERT INTO metadata VALUES (?,?)',(VERSION,.05))
        self.db.execute('CREATE TABLE frames (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        self.db.commit()
        self.count = 0

    def append(self, snapshot):
        payload = zlib.compress(json.dumps(snapshot,separators=(',',':'),allow_nan=False).encode(),1)
        self.db.execute('INSERT INTO frames VALUES (?,?)',(self.count,payload))
        self.count += 1
        if self.count%100 == 0:
            self.db.commit()

    def close(self):
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

class RaceReplay:
    def __init__(self,path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            version,self.dt = self.db.execute('SELECT version,dt FROM metadata').fetchone()
            if version != VERSION:
                raise ValueError('Unsupported recording version')
            self.count = self.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
            if self.count == 0:
                raise ValueError('Recording has no frames')
        except Exception:
            self.db.close()
            raise

    def frame(self,index):
        index = max(0,min(self.count-1,int(index)))
        row = self.db.execute('SELECT payload FROM frames WHERE id=?',(index,)).fetchone()
        return json.loads(zlib.decompress(row[0]))

    def close(self): self.db.close()
