from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    @property
    def tag(self): return self.raw.get('paths',{}).get('experiment_tag','default')
    @property
    def root(self): return Path(self.raw.get('paths',{}).get('repo_root',Path(__file__).resolve().parent))
    @property
    def data_root(self): return Path(self.raw.get('paths',{}).get('data_root',self.root))
    def section(self, name): return self.raw.get(name,{})
    def path(self, kind, tag=None): return self.data_root / kind / (tag or self.tag)
    def data_dir(self, tag=None): return self.data_root / 'data' / (tag or self.tag)
    def runs_dir(self, tag=None): return self.data_root / 'runs' / (tag or self.tag)
    def results_dir(self, tag=None): return self.root / 'results' / (tag or self.tag)
def load(path='config.yaml', tag=None):
    p=Path(path); p=p if p.is_absolute() else Path(__file__).resolve().parent/p
    raw=yaml.safe_load(p.read_text(encoding='utf-8')) or {}; raw.setdefault('paths',{})
    if tag: raw['paths']['experiment_tag']=tag
    return Settings(raw)
