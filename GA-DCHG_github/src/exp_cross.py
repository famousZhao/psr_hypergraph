"""Cross-domain experiments for GA-DCHG.

Protocols (source -> target, both recording-disjoint):
  cwru_03    : CWRU load 0 hp -> load 3 hp            (single-source, hardest)
  cwru_0123  : CWRU loads {0,1,2} -> load 3 hp        (multi-source)

Stages (argv):
  caches     build trace + static (dyn) hypergraph caches per condition
  screen     Component 1: descriptor domain-invariance -> select the anchor
  buildgad   build the anchor-guided caches (gad / gad_nt / gad_na)
  train      run all models x seeds on the cross-domain protocols
  audit      Component 4: test-time intervention on the anchor descriptor
"""
import json
import sys
import numpy as np
import torch

import config as C
import gad
from model import DynHyperNetPlus, CNN1D

PER_CLASS = 60
SEEDS = [0, 1, 2]
EPOCHS = 60
D = C.PSR_M + C.NODEF_DIM
DESC = ["kappa", "chi", "rho"]

PROTOCOLS = {
    "cwru_03":   dict(ds="cwru", src=[0],       tgt=[3],   tags="src0",  tagt="tgt3"),
    "cwru_0123": dict(ds="cwru", src=[0, 1, 2], tgt=[3],   tags="src012", tagt="tgt3"),
}


# --------------------------------------------------------------------------- caches
def stage_caches():
    for proto, p in PROTOCOLS.items():
        for tag, cond in ((p["tags"], p["src"]), (p["tagt"], p["tgt"])):
            gad.build_trace_cache_cond(p["ds"], cond, tag, per_class=PER_CLASS)
            gad.build_cond_cache(p["ds"], cond, tag, "dyn", per_class=PER_CLASS)
        print("caches", proto, "done", flush=True)


# --------------------------------------------------------------------------- screen
def stage_screen():
    out = {}
    for proto, p in PROTOCOLS.items():
        scr = gad.screen_anchors(p["ds"], p["src"], p["tgt"], p["tags"], p["tagt"],
                                 per_class=PER_CLASS)
        w1s = [scr[n]["w1"] for n in DESC]
        scr["_anchor"] = DESC[int(np.argmin(w1s))]
        scr["_nonanchor"] = DESC[int(np.argmax(w1s))]
        out[proto] = scr
        print(f"screen {proto}: {json.dumps(scr)}", flush=True)
    (C.RESULTS / "cross_screen.json").write_text(json.dumps(out, indent=1))
    print("wrote cross_screen.json", flush=True)


def _load_screen():
    return json.loads((C.RESULTS / "cross_screen.json").read_text())


def _anchor_idx(proto):
    scr = _load_screen()[proto]
    return (DESC.index(scr["_anchor"]), DESC.index(scr["_nonanchor"]))


# --------------------------------------------------------------------------- buildgad
def _gad_cache_specs(proto):
    """(ds, cond, tag, anchor_idx, temporal) triples to build for this protocol."""
    a_star, a_na = _anchor_idx(proto)
    p = PROTOCOLS[proto]
    specs = []
    for tag, cond in ((p["tags"], p["src"]), (p["tagt"], p["tgt"])):
        specs.append((p["ds"], cond, tag, a_star, True))    # gad
        specs.append((p["ds"], cond, tag, a_star, False))   # gad_nt
        specs.append((p["ds"], cond, tag, a_na, True))      # gad_na
    return specs, a_star, a_na


def stage_buildgad():
    for proto in PROTOCOLS:
        specs, _, _ = _gad_cache_specs(proto)
        for ds, cond, tag, a, temporal in specs:
            gad.build_cond_cache(ds, cond, tag, "gad", anchor_idx=a,
                                 temporal=temporal, per_class=PER_CLASS)
        print("buildgad", proto, "done", flush=True)


# --------------------------------------------------------------------------- train
def make_model(proto, p, model, a_star, a_na):
    nc = 10 if p["ds"] == "cwru" else 2
    if model == "cnn":
        return CNN1D(nc), "raw", "dyn", None, True
    if model == "dynhg":
        return (DynHyperNetPlus(nc, d_node=D, attention="phys", dual=True,
                                use_raw=False), "hyper", "dyn", None, True)
    if model == "gad":
        return gad.GADNet(nc, d_node=D, anchor_idx=a_star), "gad", "gad", a_star, True
    if model == "gad_noalign":
        return gad.GADNet(nc, d_node=D, anchor_idx=a_star), "gad", "gad", a_star, True
    if model == "gad_notemporal":
        return gad.GADNet(nc, d_node=D, anchor_idx=a_star), "gad", "gad", a_star, False
    if model == "gad_noanchor":
        return gad.GADNet(nc, d_node=D, anchor_idx=a_na), "gad", "gad", a_na, True
    raise ValueError(model)


def _run_proto(proto, p, a_star, a_na):
    ds, tags, tagt = p["ds"], p["tags"], p["tagt"]
    src_y = np.load(C.CACHE / f"trace_{ds}_{tags}.npz")["y"]
    tgt_z = np.load(C.CACHE / f"hg_{ds}_{tagt}_dyn.npz")
    tgt_fid = tgt_z["fid"]
    models = ["cnn", "dynhg", "gad", "gad_noalign", "gad_notemporal", "gad_noanchor"]
    out = {m: [] for m in models}
    for m in models:
        for sd in SEEDS:
            model, kind, mode, a, temporal = make_model(proto, p, m, a_star, a_na)
            tr_src, va_src = gad.split_source(src_y, seed=sd)
            ad_tgt, te_tgt = gad.split_target(tgt_fid, seed=sd)
            use_align = (m == "gad")
            r = gad.fit_cross(model, ds, tags, tagt, mode, kind,
                              tr_src, va_src, ad_tgt, te_tgt, seed=sd,
                              epochs=EPOCHS, use_align=use_align,
                              anchor_idx=a, temporal=temporal)
            acc = r["acc"]
            out[m].append(acc)
            print(f"  {proto}/{m}/seed{sd} acc={acc:.4f} t={r['train_time']:.0f}s",
                  flush=True)
    res = {}
    cnn = float(np.mean(out["cnn"]))
    for m, accs in out.items():
        res[m] = dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
                      gap_cnn=float(np.mean(accs) - cnn))
    res["cnn"]["gap_cnn"] = 0.0
    return res


def stage_train():
    for proto, p in PROTOCOLS.items():
        a_star, a_na = _anchor_idx(proto)
        res = _run_proto(proto, p, a_star, a_na)
        (C.RESULTS / f"cross_{proto}.json").write_text(json.dumps(res, indent=1))
        print("wrote cross_%s.json" % proto, flush=True)


# --------------------------------------------------------------------------- audit
def stage_audit():
    for proto, p in PROTOCOLS.items():
        a_star, a_na = _anchor_idx(proto)
        ds, tags, tagt = p["ds"], p["tags"], p["tagt"]
        tgt_z = np.load(C.CACHE / f"hg_{ds}_{tagt}_dyn.npz")
        ad_tgt, te_tgt = gad.split_target(tgt_z["fid"], seed=SEEDS[0])
        model = gad.GADNet(10 if ds == "cwru" else 2, d_node=D, anchor_idx=a_star)
        src_y = np.load(C.CACHE / f"trace_{ds}_{tags}.npz")["y"]
        tr_src, va_src = gad.split_source(src_y, seed=SEEDS[0])
        gad.fit_cross(model, ds, tags, tagt, "gad", "gad", tr_src, va_src,
                      ad_tgt, te_tgt, seed=SEEDS[0], epochs=EPOCHS,
                      anchor_idx=a_star, temporal=True)
        aud = gad.intervention_audit(model, ds, tagt, "gad", te_tgt,
                                     anchor_idx=a_star, temporal=True)
        aud["anchor"] = DESC[a_star]
        (C.RESULTS / f"audit_{proto}.json").write_text(json.dumps(aud, indent=1))
        print("audit", proto, json.dumps(aud), flush=True)


STAGES = {"caches": stage_caches, "screen": stage_screen,
          "buildgad": stage_buildgad, "train": stage_train, "audit": stage_audit}

if __name__ == "__main__":
    args = sys.argv[1:]
    protos = [a for a in args if a in PROTOCOLS] or list(PROTOCOLS)
    stages = [a for a in args if a in STAGES]
    if not stages:
        stages = ["caches", "screen", "buildgad", "train", "audit"]
    for proto in protos:
        if len(protos) < len(PROTOCOLS):
            print(f"=== protocol {proto} ===", flush=True)
        globals()["PROTOCOLS"] = {proto: PROTOCOLS[proto]}
        for s in stages:
            print(f"== stage {s} ==", flush=True)
            STAGES[s]()

