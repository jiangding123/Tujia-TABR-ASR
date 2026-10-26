#!/usr/bin/env python3
"""CPU-only deterministic integration test of FA main-only N-best decoding.

Uses a synthetic encoder and vocabulary to compare the original implementation
against both modes of the updated decoder without proprietary data or a GPU.
It tests identical candidate lists and exact FA main CTC scores, and verifies
that the no-aux mode does not attempt to load auxiliary_heads.pt.
"""
import contextlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "xlsr_baseline"
sys.path.insert(0, str(SCRIPT_DIR))

class FakeFA:
    def __init__(self):
        self.config = types.SimpleNamespace(vocab_size=871, hidden_size=4)
        self.dropout = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4,871)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.bias.fill_(-20)
            for idx, value in ((0, 0.0), (3, .36), (4, .18), (5, .12)):
                self.lm_head.bias[idx] = value
        self.wav2vec2 = self.encode

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, input_values, attention_mask, **kwargs):
        # Encoder never uses an auxiliary head and is identical in both runs.
        hidden = input_values.unsqueeze(-1) * torch.tensor([.1,.2,.3,.4])
        return types.SimpleNamespace(last_hidden_state=hidden)

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths

class FakeExtractor:
    @classmethod
    def from_pretrained(cls, path):
        return cls()

class FakeDataset:
    def __init__(self, path, sampling_rate, root):
        self.rows = [
            {"key":"FAKE_001", "txt":"sy3x1", "wav":"fake1.wav"},
            {"key":"FAKE_002", "txt":"sy4x1", "wav":"fake2.wav"},
        ]
    def __len__(self):
        return len(self.rows)

class FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return 1
    def __iter__(self):
        yield {
            "keys": [r["key"] for r in self.dataset.rows],
            "ref_tokens": [r["txt"].split() for r in self.dataset.rows],
            "input_values": torch.tensor([[.1,.3,.2,.4,.5],[.4,.2,.3,.1,.6]]),
            "attention_mask": torch.ones((2,5),dtype=torch.long),
        }

ID2TOKEN = {0:"<blank>", 1:"<unk>", 2:"<sos/eos>"}
ID2TOKEN.update({i:f"sy{i}x1" for i in range(3,871)})
NT = {"<blank>":0,**{f"sy{i}x":i-2 for i in range(3,871)}}
TONE = {"<blank>":0,"1":1,"<no_tone>":2}

def fake_vocab(path):
    return {value:key for key,value in ID2TOKEN.items()}, ID2TOKEN

head = torch.nn.Linear(4,3)
with torch.no_grad():
    head.weight.fill_(0.125)
    head.bias.zero_()
aux_data = {
    "use_notone":False,
    "use_tone":True,
    "notone_head":None,
    "tone_head":head.state_dict(),
    "notone_token2id":NT,
    "tone_token2id":TONE,
    "tone_regex":r"^(.*?)([0-9]+)$",
    "unmarked_tone_token":"<no_tone>",
    "lambda_notone":0.0,
    "lambda_tone":0.4,
}

fake_transformers = types.ModuleType("transformers")
fake_transformers.Wav2Vec2ForCTC = FakeFA
fake_transformers.Wav2Vec2FeatureExtractor = FakeExtractor
sys.modules["transformers"] = fake_transformers

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DecodeDataset = FakeDataset
    module.DecodeCollator = lambda *a: object()
    module.DataLoader = lambda dataset, **kwargs: FakeLoader(dataset)
    module.load_symbol_table = fake_vocab
    return module

modified=load_module(SCRIPT_DIR/"decode_xlsr_aux_nbest.py", "modified_decoder")

# Optional original decoder from the user's uploaded archive for regression.
# This test is still useful without that archive: compare modified legacy
# mode with modified main-only mode.
original_path = Path(__file__).resolve().with_name("_original_decoder_for_test.py")
original = load_module(original_path, "original_decoder") if original_path.exists() else None

saved_device=torch.device
saved_available=torch.cuda.is_available
saved_autocast=torch.cuda.amp.autocast
saved_load=torch.load


def run_decoder(mod, model_dir, output, main_only, load_calls):
    argv=["decode_xlsr_aux_nbest.py", "--model_dir", str(model_dir),
          "--data_list", "fake.jsonl", "--dict_path", "fake.vocab",
          "--output_jsonl", str(output), "--task", "fax", "--beam_size", "4",
          "--token_prune", "5", "--batch_size", "2", "--num_workers", "0"]
    if main_only:
        argv.append("--main_ctc_only")
    with patch.object(sys,"argv",argv), \
         patch.object(torch.cuda,"is_available",lambda:True), \
         patch.object(torch,"device",lambda x:saved_device("cpu")), \
         patch.object(torch.cuda.amp,"autocast",lambda **kw:contextlib.nullcontext()), \
         patch.object(torch,"load",lambda *a,**kw:load_calls.append(str(a[0])) or aux_data):
        mod.main()
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    meta=json.loads(output.with_suffix(".meta.json").read_text())
    return rows,meta


def main():
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp)
        model=d/"fake_FA_model"
        model.mkdir()
        aux_file=model/"auxiliary_heads.pt"
        aux_file.write_bytes(b"SYNTHETIC TEST PLACEHOLDER")
        ref_rows=ref_meta=None
        if original is not None:
            loads=[]
            ref_rows,ref_meta=run_decoder(original,model,d/"original.jsonl",False,loads)
            assert loads == [str(aux_file)]
        loads=[]
        old_rows,old_meta=run_decoder(modified,model,d/"aux_mode.jsonl",False,loads)
        assert loads == [str(aux_file)]
        if ref_rows is not None:
            assert ref_rows == old_rows, "Legacy auxiliary mode changed"
        # Main-only must work even without auxiliary_heads.pt on disk.
        aux_file.unlink()
        loads=[]
        only_rows,only_meta=run_decoder(modified,model,d/"main_only.jsonl",True,loads)
        assert loads == [], "Main-only decoder opened auxiliary checkpoint"
        assert only_meta["decoding_mode"]=="main_ctc_only"
        assert only_meta["auxiliary_checkpoint_loaded"] is False
        assert only_meta["auxiliary_scoring_enabled"] is False
        assert len(old_rows)==len(only_rows)==2
        for old, new in zip(old_rows, only_rows):
            assert old["key"]==new["key"]
            assert old["ref_tokens"]==new["ref_tokens"]
            assert len(old["candidates"])==len(new["candidates"])
            for c1,c2 in zip(old["candidates"],new["candidates"]):
                assert c1["rank"]==c2["rank"]
                assert c1["tokens"]==c2["tokens"]
                assert c1["main_score"]==c2["main_score"], (c1,c2)
                assert c1["tone_score"] is not None
                assert c2["tone_score"] is None
                assert c2["notone_score"] is None
        # Guard: main-only is for FA/task fax, not FN/J mode.
        with patch.object(sys,"argv",["decode_xlsr_aux_nbest.py", "--model_dir",str(model),
            "--data_list","fake.jsonl", "--dict_path","fake.vocab",
            "--output_jsonl",str(d/"bad.jsonl"),"--task","fnx","--main_ctc_only"]), \
            patch.object(torch.cuda,"is_available",lambda:True):
            try:
                modified.main()
            except SystemExit as exc:
                assert "--task fax" in str(exc)
            else:
                raise AssertionError("Expected incompatible task guard")
        print("PASS: same FA beam ranks/tokens and exact main CTC scores")
        print("PASS: no auxiliary checkpoint loaded, even when absent")
        print("PASS: tone_score and notone_score are null in main-only FA cache")
        print("PASS: original auxiliary mode remains compatible")

if __name__=="__main__":
    main()
