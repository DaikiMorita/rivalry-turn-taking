"""記事 §「モデルの実証」の数値実験を再生成する。

実行: python3 experiments.py
標準ライブラリだけで動く。20 個の乱数 seed × 各 2000 ターンの平均。

検証したい主張:
  (1) 対等な弱い刺激（ambient）では、ほとんど誰も話さない（= 無駄に喋り出さない自制）。
  (2) みなが乗っていれば、均等に回り、同じ者が連続では座らない（独占しない）。
  (3) 名指し（一者の刺激だけ上げる）は滑らかに効くが、不応のせいで「一回おき」が上限。
  (4) 揺らぎ（sigma）はちょうどよい範囲が要る（0 だと対称が破れず勝者総取り）。
  (5) round-robin / 重みつき random と比べ、rivalry だけが「名指しに応える」と
      「独占させない」を同時に満たす。
"""

import random

from rivalry import PARAMS, turn

SEEDS = list(range(20))
TURNS = 2000

# 場の刺激水準（このデモ専用の設定値。原理を見せるために選んだ代表値）。
U_AMBIENT = 0.45   # その場に居合わせているだけ（名指しなし）の弱い刺激
U_ON = 1.00        # 話題に乗っている状態の刺激

# 発話直後の自己抑制: 直前に喋った者の「次のターンの刺激」を弱める係数。
REPEAT_SUPPRESS = 0.5


def simulate(n, stim, seed, *, turns=TURNS, p=PARAMS, repeat_suppress=REPEAT_SUPPRESS):
    """rivalry 力学で turns ターン回す。

    stim: 長さ n の list[float]（各エージェントの基準刺激）。
    x/a/b はターンを跨いで持ち越す。直前に喋った者だけ次ターンの刺激を弱める。
    戻り値: (counts: list[int], silent: int, runs_ge4: int)
        runs_ge4 = 「同じ者が 4 回以上連続で座っている」状態にあった発話ターン数。
    """
    rng = random.Random(seed)
    x = {i: 0.0 for i in range(n)}
    a = {i: 0.0 for i in range(n)}
    b = {i: 0.0 for i in range(n)}
    counts = [0] * n
    silent = 0
    last = None
    run = 0
    runs_ge4 = 0
    for _ in range(turns):
        u = {}
        for i in range(n):
            ui = stim[i]
            if i == last:
                ui *= repeat_suppress
            u[i] = ui
        w, x, a, b = turn(x, a, b, u, p, rng)
        if w is None:
            silent += 1
            last = None
            run = 0
        else:
            counts[w] += 1
            run = run + 1 if w == last else 1
            if run >= 4:
                runs_ge4 += 1
            last = w
    return counts, silent, runs_ge4


def agg(n, stim, **kw):
    """SEEDS 全体で集計し、share / 沈黙率 / 連続4回以上率 を返す。"""
    tot = [0] * n
    silent = 0
    ge4 = 0
    for s in SEEDS:
        c, si, g = simulate(n, stim, s, **kw)
        tot = [t + ci for t, ci in zip(tot, c)]
        silent += si
        ge4 += g
    speak = sum(tot)
    total = len(SEEDS) * TURNS
    return {
        "shares": [t / speak if speak else 0.0 for t in tot],
        "silence": silent / total,
        "ge4": ge4 / speak if speak else 0.0,
        "speak_frac": speak / total,
    }


def baseline_round_robin(n):
    """順番に回すだけ（中央の固定規則）。刺激を一切見ない・常に誰かが話す。"""
    return {"shares": [1.0 / n] * n, "ge4": 0.0, "silence": 0.0}


def baseline_weighted_random(stim):
    """刺激で重みづけしたランダム。名指しは重みに反映できるが連続を止められない。"""
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
    return {"shares": [t / speak for t in tot], "ge4": ge4 / speak, "silence": 0.0}


def pct(x):
    return f"{x * 100:.0f}%"


def shares_str(shares):
    return " / ".join(f"{s:.2f}" for s in shares)


def main():
    print(f"= rivalry turn-taking 実証 ({len(SEEDS)} seeds x {TURNS} turns) =\n")

    print("(1) 対等な ambient では黙る（沈黙率）")
    for n in (2, 3, 4):
        r = agg(n, [U_AMBIENT] * n)
        print(f"  {n} 体: 沈黙 {pct(r['silence'])} / 話したときの share {shares_str(r['shares'])}")

    print("\n(2) みなが乗っていれば均等に回り、独占しない（刺激を全員 {:.1f}）".format(U_ON))
    r = agg(3, [U_ON] * 3)
    print(f"  3 体: 沈黙 {pct(r['silence'])} / share {shares_str(r['shares'])} / 連続4回以上 {pct(r['ge4'])}")

    print("\n(3) 名指しは滑らかに効くが独占はできない（3 体・一者の刺激だけ上げる）")
    print("  刺激 | その者の share")
    for s in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        r = agg(3, [s, U_ON, U_ON])
        print(f"  {s:.1f}  | {r['shares'][0]:.2f}")

    print("\n(4) 揺らぎの役割（2 体・刺激同じ・sigma を変える）")
    for sg in (0.0, 0.05, 0.1, 0.4):
        p = dict(PARAMS, sigma=sg)
        r = agg(2, [U_ON, U_ON], p=p)
        print(f"  sigma={sg:<4}: share {shares_str(r['shares'])} / 連続4回以上 {pct(r['ge4'])}")

    print("\n(5) 3 方式の比較（3 体・一者だけ刺激 1.4）")
    riv = agg(3, [1.4, U_ON, U_ON])
    rr = baseline_round_robin(3)
    rnd = baseline_weighted_random([1.4, 1.0, 1.0])
    print("  方式         | 名指しの share | 連続4回以上")
    print(f"  rivalry      | {riv['shares'][0]:.2f}           | {pct(riv['ge4'])}")
    print(f"  round-robin  | {rr['shares'][0]:.2f}           | {pct(rr['ge4'])}")
    print(f"  weighted-rnd | {rnd['shares'][0]:.2f}           | {pct(rnd['ge4'])}")


if __name__ == "__main__":
    main()
