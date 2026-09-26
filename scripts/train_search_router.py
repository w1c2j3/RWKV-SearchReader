"""State-only G1J classifier training, physically restricted to GPU 3."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

GPU = "GPU-a9570da2-547a-c2b3-0cab-7bbdc1a8a8b0"
ROOT = Path(__file__).resolve().parents[1]
PEFT = Path("/home/chase/chase/RWKV-PEFT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/search_router_v1")
    parser.add_argument("--checkpoint", default="/mnt/nas-model/g1j/rwkv7-g1j-1.5b-20260831-ctx16384.pth")
    parser.add_argument("--init-state")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serve-state")
    parser.add_argument("--port", type=int, default=18237)
    parser.add_argument("--served-model-name", default="rwkv7-g1j-1.5b-search-router-state-v1")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=bool(args.serve_state))
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, WKV="fla", RWKV_MY_TESTING="x070",
        RWKV_TRAIN_TYPE="state", FUSED_KERNEL="0", RWKV_HEAD_SIZE_A="64", RWKV_JIT_ON="0",
        PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", HF_HUB_OFFLINE="1",
        TRITON_CACHE_DIR=os.environ.get("SEARCH_READER_STATE_CACHE", str(output / "triton")), TORCH_EXTENSIONS_DIR=str(output / "extensions"))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Training time budget reached")))
    signal.alarm(1800)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20260914)
    assert torch.cuda.device_count() == 1
    sys.path.insert(0, str(PEFT))
    from rwkvt.rwkv7.model import RWKV7
    from rwkv_search_reader.routing import build_decision_prompt
    from rwkv_search_reader.model_client import build_completion_prompt
    spec = importlib.util.spec_from_file_location("router_tokenizer", PEFT / "json2binidx_tool/tools/rwkv_tokenizer.py")
    tokenizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizer_module)
    tokenizer = tokenizer_module.TRIE_TOKENIZER(str(PEFT / "json2binidx_tool/rwkv_vocab_v20230424.txt"))
    checkpoint = Path(args.checkpoint)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    layers = max(int(n.split(".")[1]) for n in weights if n.startswith("blocks.")) + 1
    vocab, hidden = weights["emb.weight"].shape
    cfg = SimpleNamespace(vocab_size=vocab, n_layer=layers, n_embd=hidden, dim_att=hidden,
        dim_ffn=weights["blocks.0.ffn.key.weight"].shape[0], head_size_a=64,
        head_size_divisor=8, ctx_len=16384 if args.serve_state else 4096, grad_cp=1, peft="state", train_type="state",
        my_testing="x070", dropout=0)
    with torch.device("meta"):
        model = RWKV7(cfg)
    names = [f"blocks.{i}.att.time_state" for i in range(layers)]
    expected = model.state_dict()
    assert set(expected)-set(weights) == set(names)
    assert not set(weights)-set(expected)
    assert all(expected[n].shape == tensor.shape for n,tensor in weights.items())
    model.load_state_dict({**weights, **{n:torch.zeros((hidden//64,64,64), dtype=torch.bfloat16) for n in names}}, strict=True, assign=True)
    model.requires_grad_(False)
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    for block in model.blocks:
        block.att.time_state = torch.nn.Parameter(block.att.time_state.detach().float())
    state = [p for n,p in model.named_parameters() if n in names]
    base = [p for n,p in model.named_parameters() if n not in names]
    versions = [p._version for p in base]
    assert all(not p.requires_grad for p in base)

    def read(split):
        rows = []
        for line in (ROOT / args.dataset / (split+".jsonl")).read_text().splitlines():
            row = json.loads(line)
            prompt = build_completion_prompt("", build_decision_prompt(row["messages"]))
            prefix = tokenizer.encode(prompt)
            full = tokenizer.encode(prompt+row["target"])
            assert full[:len(prefix)] == prefix
            assert len(full) < cfg.ctx_len
            rows.append({**row, "prefix": prefix, "tokens": full+[0]})
        return rows

    if args.serve_state:
        saved = torch.load(args.serve_state, weights_only=True, map_location="cpu")
        assert set(saved) == set(names)
        model.load_state_dict(saved, strict=False)
        model.eval()
        model.requires_grad_(False)
        signal.alarm(0)
        warm_ids = tokenizer.encode(build_completion_prompt("", build_decision_prompt([
            {"role": "user", "content": "请判断这句问候是否需要联网：你好。"}
        ])))
        with torch.no_grad():
            for _ in range(3):
                logits = model(torch.tensor([warm_ids], device="cuda:0"))
                warm_ids.append(int(logits[0,-1].argmax()))
                del logits
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class Handler(BaseHTTPRequestHandler):
            def send(self, status, body):
                encoded = json.dumps(body, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            def do_GET(self):
                if self.path == "/health":
                    return self.send(200, {"status": "ok"})
                if self.path == "/v1/models":
                    return self.send(200, {"data": [{"id": args.served_model_name}]})
                return self.send(404, {"error": "not_found"})
            def do_POST(self):
                if self.path != "/v1/completions":
                    return self.send(404, {"error": "not_found"})
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 512000:
                        return self.send(413, {"error": "body too large"})
                    body = json.loads(self.rfile.read(length))
                    if body.get("model") != args.served_model_name:
                        return self.send(400, {"error": "unknown model"})
                    if not isinstance(body.get("prompt"), str):
                        return self.send(400, {"error": "prompt must be text"})
                    ids = tokenizer.encode(body["prompt"])
                    if not ids or len(ids)+8 > cfg.ctx_len:
                        return self.send(400, {"error": "context limit exceeded"})
                    generated = []
                    with torch.no_grad():
                        for _ in range(min(8, max(1, int(body.get("max_tokens", 8))))):
                            logits = model(torch.tensor([ids], device="cuda:0"))
                            token = int(logits[0,-1].argmax()); del logits
                            if token == 0: break
                            ids.append(token); generated.append(token)
                    self.send(200, {"choices": [{"text": tokenizer.decode(generated)}]})
                except (ValueError, KeyError, TypeError):
                    self.send(400, {"error": "invalid request"})
                except Exception:
                    self.send(500, {"error": "inference failed"})
        print(json.dumps({"serving_state": args.serve_state, "port": args.port}), flush=True)
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
        return

    if args.init_state:
        initial = torch.load(args.init_state, weights_only=True, map_location="cpu")
        assert set(initial) == set(names)
        model.load_state_dict(initial, strict=False)

    train, validation = read("train"), read("validation")
    (output/"config.json").write_text(json.dumps({"model":str(checkpoint),"gpu":GPU,
        "epochs":args.epochs,"lr":args.lr,"dataset":args.dataset,"init_state":args.init_state,"accumulation":4,"train":len(train),
        "validation":len(validation),"seed":20260914,"trainable":"time_state only"},indent=2))

    @torch.no_grad()
    def evaluate(rows):
        model.eval()
        records=[]
        for row in rows:
            ids=list(row["prefix"]); generated=[]; nll=None
            for _ in range(8):
                logits=model(torch.tensor([ids],device="cuda:0"))
                if nll is None:
                    target = torch.tensor([row["tokens"][len(row["prefix"])]], device="cuda:0")
                    nll = float(torch.nn.functional.cross_entropy(logits[0,-1:].float(), target))
                token=int(logits[0,-1].argmax()); del logits
                if token==0:break
                generated.append(token);ids.append(token)
            text=tokenizer.decode(generated).strip()
            prediction={"true":True,"false":False}.get(text)
            records.append({"id":row["id"],"target":row["needs_search"],"output":text,
                "correct":prediction is row["needs_search"],"format_ok":prediction is not None,"nll":nll})
        return {"correct":sum(r["correct"] for r in records),"total":len(records),
            "format_errors":sum(not r["format_ok"] for r in records),"mean_nll":sum(r["nll"] for r in records)/len(records),"cases":records}

    baseline=evaluate(validation)
    (output/"baseline-validation.json").write_text(json.dumps(baseline,ensure_ascii=False,indent=2))
    print(json.dumps({"baseline_validation":{k:v for k,v in baseline.items() if k!='cases'}}),flush=True)
    best=(-1, float("-inf"))
    optimizer=torch.optim.AdamW(state,lr=args.lr,weight_decay=0)
    step=0
    for epoch in range(args.epochs):
        rows=list(train);random.Random(20260914+epoch).shuffle(rows)
        model.train()
        for start in range(0,len(rows),4):
            group=rows[start:start+4];optimizer.zero_grad(set_to_none=True)
            losses=[];began=time.monotonic()
            for row in group:
                tokens=row["tokens"]
                logits=model(torch.tensor([tokens[:-1]],device="cuda:0"))
                begin=len(row["prefix"])-1
                targets=torch.tensor(tokens[len(row["prefix"]):],device="cuda:0")
                loss=torch.nn.functional.cross_entropy(logits[0,begin:].float(),targets)
                if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
                (loss/len(group)).backward();losses.append(float(loss.detach()))
                del logits,loss,targets
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state)
            assert all(p.grad is None and not p.requires_grad and p._version==v for p,v in zip(base,versions))
            norm=torch.nn.utils.clip_grad_norm_(state,1.0,error_if_nonfinite=True)
            optimizer.step();step+=1
            print(json.dumps({"epoch":epoch+1,"step":step,"loss":sum(losses)/len(losses),
                "grad_norm":float(norm),"seconds":round(time.monotonic()-began,2)}),flush=True)
        metric=evaluate(validation)
        (output/f"epoch-{epoch+1}-validation.json").write_text(json.dumps(metric,ensure_ascii=False,indent=2))
        torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},output/f"state-epoch-{epoch+1}.pth")
        print(json.dumps({"epoch":epoch+1,"validation":{k:v for k,v in metric.items() if k!='cases'}}),flush=True)
        score = (metric["correct"], -metric["mean_nll"])
        if score > best:
            best=score
            torch.save({n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names},output/"best-state.pth")
    best_state=torch.load(output/"best-state.pth",weights_only=True,map_location="cpu")
    model.load_state_dict(best_state,strict=False)
    # Test is only loaded after checkpoint selection is complete.
    test=evaluate(read("test"))
    (output/"test.json").write_text(json.dumps(test,ensure_ascii=False,indent=2))
    print(json.dumps({"test":{k:v for k,v in test.items() if k!='cases'}}),flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
