"""両眼視野闘争（binocular rivalry）型の神経力学で、司会者なしに「発言の椅子」を回す最小モデル。

記事「複数の AI を司会者なしで会話させる」で示した玩具シミュレータの本体。
- 標準ライブラリ（math / random）だけで動く。特定のアプリ・フレームワークには一切依存しない。
- 各エージェントの外部刺激 u_i がどこから来るか（言語モデル / ルール / 人間）は問わない。
  この力学は「活性 x_i がしきい値を最初に越えたのは誰か」だけを見て発言者を決める。

力学（各エージェント i）:
    x_i : 活性（いまどれだけ発言に乗り出しているか）
    a_i : 遅い疲労（長く立ち上がるほど溜まり、ゆっくり抜ける）
    b_i : 速い不応（発話直後に跳ね、ターンの中で素早く抜ける = 神経の不応期）

    dx_i = -alpha*x_i + w_s*phi(x_i)            # 減衰 + 自己興奮（双安定）
           - w_I*(y - phi(x_i))                 # 相互抑制（自分以外の占有度の合計）
           - beta*a_i - b_i                     # 遅い疲労 + 速い不応
           + g_u*u_i + noise                    # 外部刺激 + 揺らぎ
    da_i = -gamma*a_i + eta*x_i
    db_i = -(1/tau_b)*b_i + kappa_b*phi(x_i)
    phi  = sigmoid,  y = Σ_j phi(x_j)（場の占有度 = 共有スカラー）

注: これは原理を見せるための簡約形。実プロダクトの本番エンジンは、相互抑制に
「はっきり立ち上がった者だけを強く数える急峻な占有度関数」と「在席数によらない一定の
基底抑制」を足しているが、ここではそれを省いて骨格だけを示す（ODE 定数は本番と同じ）。
"""

import math

# 本番エンジンと同じ ODE 定数（占有度整形などの作り込みは省いた簡約形）。
PARAMS = dict(
    alpha=0.5, w_s=1.0, w_I=0.7, beta=0.7, gamma=0.05, eta=0.2,
    tau_b=1.5, kappa_b=1.0, g_u=1.0, theta=0.45, sigma=0.1, substeps=10,
)


def sigmoid(x: float) -> float:
    """数値的に安定なロジスティック・シグモイド（値域 0〜1）。"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def euler_step(x, a, b, u, p, rng):
    """ODE を 1 ステップ（前進オイラー、揺らぎ込み）進める。x/a/b は dict[id -> float]。"""
    phi = {i: sigmoid(x[i]) for i in x}
    y = sum(phi.values())  # 場の占有度（共有スカラー）
    nx, na, nb = {}, {}, {}
    for i in x:
        dx = (
            -p["alpha"] * x[i] + p["w_s"] * phi[i]   # 減衰 + 自己興奮
            - p["w_I"] * (y - phi[i])                # 相互抑制（自分以外の占有度）
            - p["beta"] * a[i] - b[i]                # 遅い疲労 + 速い不応
            + p["g_u"] * u.get(i, 0.0)               # 外部刺激
            + rng.gauss(0.0, p["sigma"])             # 揺らぎ
        )
        nx[i] = x[i] + dx
        na[i] = a[i] + (-p["gamma"] * a[i] + p["eta"] * x[i])
        nb[i] = b[i] + (-(1.0 / p["tau_b"]) * b[i] + p["kappa_b"] * phi[i])
    return nx, na, nb


def turn(x, a, b, u, p, rng):
    """ひと区切り（substeps ステップ）積分し、最初に theta を越えたエージェントを返す。

    勝者は「振幅」ではなく「位相」で決める = first-passage（最初にしきい値を越えた者）。
    同時に越えたら、そのあいだの活性の積み上がり（Σphi）が大きいほうを採る。
    誰も越えなければ None（＝沈黙。今は誰も話さなくてよい、を表現できる）。
    """
    crossing, integ = {}, {i: 0.0 for i in x}
    for s in range(1, p["substeps"] + 1):
        x, a, b = euler_step(x, a, b, u, p, rng)
        for i in x:
            integ[i] += sigmoid(x[i])
            if i not in crossing and x[i] > p["theta"]:
                crossing[i] = s
    if not crossing:
        return None, x, a, b
    winner = min(crossing, key=lambda i: (crossing[i], -integ[i]))
    return winner, x, a, b
