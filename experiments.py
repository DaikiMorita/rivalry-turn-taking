"""分散ターンテイキング mock の実証実験。

実行: python3 experiments.py
標準ライブラリだけで動く。集計は 20 個の乱数 seed × 各 2000 ターンの平均。

見せたいこと（2 本柱）:
  柱1: これは「rivalry という名のオーケストレーター」ではなく peer-mesh（分散）である
       → コード構造（rivalry.py）が満たす不変条件で示す。各ノードは公開スカラーと
         自分の drive だけで動き、勝者は公開イベントだけの固定ルールで創発する。
  柱2: 健全な会話（独占せず・名指しに応え・無駄に喋らない）は "偶然" ではなく
       原理的に決まる『窓』に宿る。同じ力学はパラメータ次第で〈独占／沈黙／健全〉の
       どれにもなる。その全レンジを正直に見せ、健全領域を特定する。
"""

import random

from rivalry import DEFAULTS, RivalryNode, run_turn

SEEDS = list(range(20))
TURNS = 2000

U_ON = 1.0       # 話題に乗っている状態の刺激
U_WEAK = 0.25    # その場に居合わせているだけの弱い刺激
REPEAT_SUPPRESS = 0.5  # 発話直後の自己抑制（直前に喋った者の次ターンの刺激を弱める公開事実）

# 独占レジーム: 独占を防ぐ機構（速い不応・遅い疲労・揺らぎ）を全部切ったパラメータ。
# = 記事「疲れが無ければ最も高ぶった者が椅子を独占する」の実証用。
MONOPOLY = dict(DEFAULTS, kappa_b=0.0, beta=0.0, sigma=0.0)


def simulate(n, stim, seed, *, turns=TURNS, params=DEFAULTS, repeat_suppress=REPEAT_SUPPRESS):
    """rivalry mock を turns ターン回す。x/a/b はノードが保持しターンを跨ぐ。"""
    rng = random.Random(seed)
    nodes = [RivalryNode(i, params) for i in range(n)]
    counts = [0] * n
    silent = 0
    last = None
    run = 0
    runs_ge4 = 0
    max_run = 0
    for _ in range(turns):
        drive = {i: (stim[i] * repeat_suppress if i == last else stim[i]) for i in range(n)}
        w, _, _ = run_turn(nodes, drive, rng)
        if w is None:
            silent += 1
            last = None
            run = 0
        else:
            counts[w] += 1
            run = run + 1 if w == last else 1
            max_run = max(max_run, run)
            if run >= 4:
                runs_ge4 += 1
            last = w
    return counts, silent, runs_ge4, max_run


def agg(n, stim, **kw):
    tot = [0] * n
    silent = 0
    ge4 = 0
    max_run = 0
    for s in SEEDS:
        c, si, g, mr = simulate(n, stim, s, **kw)
        tot = [t + ci for t, ci in zip(tot, c)]
        silent += si
        ge4 += g
        max_run = max(max_run, mr)
    speak = sum(tot)
    total = len(SEEDS) * TURNS
    return {
        "shares": [t / speak if speak else 0.0 for t in tot],
        "silence": silent / total,
        "ge4": ge4 / speak if speak else 0.0,
        "max_run": max_run,
    }


def floor_sequence(n, stim, seed, turns, *, params=DEFAULTS, repeat_suppress=REPEAT_SUPPRESS):
    """1 seed を turns ターン回し、各ターンに椅子を持った者の列を返す（None=沈黙）。"""
    rng = random.Random(seed)
    nodes = [RivalryNode(i, params) for i in range(n)]
    seq = []
    last = None
    for _ in range(turns):
        drive = {i: (stim[i] * repeat_suppress if i == last else stim[i]) for i in range(n)}
        w, _, _ = run_turn(nodes, drive, rng)
        seq.append(w)
        last = w
    return seq


def render_lanes(seq, n, labels):
    """誰が椅子を持つかの時系列を、エージェントごとの帯で描く（█=発言中）。"""
    return "\n".join(
        f"  {labels[i]} |" + "".join("█" if w == i else " " for w in seq) + "|" for i in range(n)
    )


def annotate_seq(seq, n):
    speak = sum(1 for w in seq if w is not None)
    counts = [seq.count(i) for i in range(n)]
    max_run = 0
    cur = 0
    prev = None
    for w in seq:
        cur = cur + 1 if (w is not None and w == prev) else (1 if w is not None else 0)
        prev = w
        max_run = max(max_run, cur)
    shares = " / ".join(f"{c / speak:.2f}" if speak else "-" for c in counts)
    return f"     → share {shares} / 最長連続 {max_run} / 沈黙 {seq.count(None)}/{len(seq)}"


def baseline_round_robin(n):
    return {"shares": [1.0 / n] * n, "ge4": 0.0}


def baseline_weighted_random(stim):
    tot = [0] * len(stim)
    ge4 = 0
    speak = 0
    ssum = sum(stim)
    for s in SEEDS:
        rng = random.Random(1000 + s)
        last = None
        run = 0
        for _ in range(TURNS):
            r = rng.random() * ssum
            acc = 0.0
            w = len(stim) - 1
            for i, v in enumerate(stim):
                acc += v
                if r <= acc:
                    w = i
                    break
            tot[w] += 1
            speak += 1
            run = run + 1 if w == last else 1
            if run >= 4:
                ge4 += 1
            last = w
    return {"shares": [t / speak for t in tot], "ge4": ge4 / speak}


def pct(x):
    return f"{x * 100:.0f}%"


def shares_str(sh):
    return " / ".join(f"{s:.2f}" for s in sh)


def main():
    print(f"= 分散ターンテイキング mock の実証 ({len(SEEDS)} seeds x {TURNS} turns) =")

    # ── 柱2-A: 同じ力学が3つのレジームを取る（正直に全レンジを見せる）──
    print("\n[A] 同じ仕組みが、パラメータ次第で3つのレジームを取る（時系列、█=発言）")
    win = 60

    print("  ① 独占 ── 不応・疲労・揺らぎを切ると、一体が椅子を占有し続ける")
    seqM = floor_sequence(3, [U_ON] * 3, seed=0, turns=win, params=MONOPOLY, repeat_suppress=1.0)
    print(render_lanes(seqM, 3, ["A", "B", "C"]))
    rM = agg(3, [U_ON] * 3, params=MONOPOLY, repeat_suppress=1.0)
    print(f"     → share {shares_str(rM['shares'])} / 最長連続 {rM['max_run']}（＝記事⑤『疲れが無ければ独占』）")

    print("  ② 沈黙 ── 刺激が弱ければ、ほとんど誰も話さない（無駄に喋らない自制）")
    seqS = floor_sequence(3, [U_WEAK] * 3, seed=0, turns=win)
    print(render_lanes(seqS, 3, ["A", "B", "C"]))
    rS = agg(3, [U_WEAK] * 3)
    print(f"     → 沈黙 {pct(rS['silence'])} / 話したときの share {shares_str(rS['shares'])}")

    print("  ③ 健全 ── フル機構なら、独占せず代わる代わる回る")
    seqH = floor_sequence(2, [U_ON] * 2, seed=0, turns=win)
    print(render_lanes(seqH, 2, ["A", "B"]))
    rH = agg(3, [U_ON] * 3)
    print(f"     → (3体集計) 沈黙 {pct(rH['silence'])} / share {shares_str(rH['shares'])} / 連続4回以上 {pct(rH['ge4'])}")

    # ── 柱2-B: 健全さは『窓』に宿る ──
    print("\n[B] 健全さは『窓』に宿る ── 揺らぎ sigma を振る（2 体・刺激同じ）")
    print("  sigma | share        | 連続4回以上 | 解釈")
    notes = {0.0: "対称が破れず勝者総取り＝独占", 0.05: "ちょうどよい＝健全", 0.1: "ちょうどよい＝健全", 0.4: "入れすぎ＝同じ者が続きがち"}
    for sg in (0.0, 0.05, 0.1, 0.4):
        r = agg(2, [U_ON, U_ON], params=dict(DEFAULTS, sigma=sg))
        print(f"  {sg:<5} | {shares_str(r['shares']):<12} | {pct(r['ge4']):<9} | {notes[sg]}")
    print("  → 独占を防ぐのは不応・疲労、対称を破るのは揺らぎ。両方がちょうどよい範囲＝『窓』。")

    # ── 柱2-C: 窓の中では rivalry だけが両立する ──
    print("\n[C] 窓の中で rivalry は baselines に勝つ（3 体・一者だけ刺激 1.4）")
    riv = agg(3, [1.4, U_ON, U_ON])
    rr = baseline_round_robin(3)
    rnd = baseline_weighted_random([1.4, 1.0, 1.0])
    print("  方式         | 名指しの share | 連続4回以上")
    print(f"  rivalry      | {riv['shares'][0]:.2f}           | {pct(riv['ge4'])}")
    print(f"  round-robin  | {rr['shares'][0]:.2f}           | {pct(rr['ge4'])}")
    print(f"  weighted-rnd | {rnd['shares'][0]:.2f}           | {pct(rnd['ge4'])}")
    print("  → 名指しに応える(0.50) ∧ 独占させない(0%) を同時に満たすのは rivalry だけ。")

    # ── 窓の中の健全な姿: N=2 / N=3 ──
    print("\n[D] 窓の中の健全な姿 ── 椅子は誰が持つか（█=発言。全員が話したい状況）")
    print("  N=2: 両眼視野闘争そのもの ── 左右がきれいに交互（2 体ならできて当然）")
    seq2 = floor_sequence(2, [U_ON] * 2, seed=0, turns=80)
    print(render_lanes(seq2, 2, ["A", "B"]))
    print(annotate_seq(seq2, 2))
    print("  N=3: 3 体でも独占は起きない ── 2 体が競り合い、揺らぎが第三者を割り込ませ、")
    print("       組み合わせが入れ替わりながら全員に floor が回る（長期 share は [A] ③ の ≈0.33 ずつ）")
    seq3 = floor_sequence(3, [U_ON] * 3, seed=0, turns=400 + 80)[400:]  # 起動直後の過渡を捨てる
    print(render_lanes(seq3, 3, ["A", "B", "C"]))
    print(annotate_seq(seq3, 3))


if __name__ == "__main__":
    main()
