"""分散ターンテイキング・エンジンの中核（教育用 mock）。

本番プロダクトの peer-mesh エンジンから「発言の椅子を司会者なしで回す」中核だけを
取り出した mock です。本番の drive 計算（公開事実→刺激 u の生成）・空間(zone)配線・
調整済みの本番パラメータ・各種ガードは含みません。目的は原理（分散ターンテイキング）
を再現可能な最小形で示すことです。

────────────────────────────────────────────────────────────
これは「rivalry という名前のオーケストレーター（中央の司会者）」ではありません。
peer-mesh（分散）であることを、コードの構造そのものが満たす2つの不変条件で示します:

  (1) 共有バスに乗るのは公開情報だけ。
      ・各ノードが場に出す「公開出力 psi（占有度への寄与）」の合計 = 場の占有度 y。
      ・「誰がいつ閾値を越えたか」という公開イベント。
      他ノードの私的状態(x,a,b)も発話内容も、誰も読みません。実際 RivalryNode.step() の
      引数は「自分の状態 + 公開スカラー field_y + 自分の drive」だけで、他ノードの
      オブジェクトを受け取らない = 構造的に他者の内部を読めません。
      （場の占有度 y は「中央が私的状態を集めて回る」のではなく、各自が公開出力を場に
        出し各自が合計を感じる平均場結合。蛍の同期・クオラムセンシングと同型の、
        指揮者なしの集団力学です。）

  (2) 勝者は「審議する権威」ではなく「全員共有の固定ルール」で決まる。
      decide_winner() は公開イベント（誰がいつ越えたか）だけの純関数で、文脈も中身も
      読みません。各ノードがこの関数をローカルに実行すれば同じ勝者を得ます = 中央の
      権威は不要。1か所で min() を呼ぶのは分散合意ルールの中央「実装」にすぎず、原理上は
      photo-finish のカメラのように「勝者を決める」のでなく「創発した結果を記録する」だけ。

つまり who-talks は《決定》ではなく《創発（first-passage）》です。脳の両眼視野闘争に
「どちらの像を見せるか決める小人」がいないのと同じ構図を、コードで保っています。
────────────────────────────────────────────────────────────
"""

import math

# 本番エンジンと同じ ODE 定数（占有度整形などの作り込みを省いた簡約形）。
DEFAULTS = dict(
    alpha=0.5, w_s=1.0, w_I=0.7, beta=0.7, gamma=0.05, eta=0.2,
    tau_b=1.5, kappa_b=1.0, g_u=1.0, theta=0.45, sigma=0.1, substeps=10,
)


def sigmoid(x: float) -> float:
    """数値的に安定なロジスティック・シグモイド（値域 0〜1）。"""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class RivalryNode:
    """1個の自律ノード。

    更新に使えるのは「自分の私的状態 (x,a,b) + 場の公開スカラー field_y + 自分の drive」
    だけ。他ノードのオブジェクトを受け取らない = 他者の内部を構造的に読めない。

    状態:
        x : 活性（いまどれだけ発言に乗り出しているか）
        a : 遅い疲労（長く立ち上がるほど溜まり、ゆっくり抜ける）
        b : 速い不応（発話直後に跳ね、ターン内で素早く抜ける = 不応期）
    """

    def __init__(self, node_id, params=DEFAULTS):
        self.id = node_id
        self.p = params
        self.x = 0.0
        self.a = 0.0
        self.b = 0.0

    def public_output(self) -> float:
        """場に出す公開信号（占有度への寄与）。他者はこれを「合計の形でだけ」受け取る。"""
        return sigmoid(self.x)

    def step(self, field_y: float, drive_u: float, rng) -> None:
        """自分の状態を 1 ステップ進める。引数は公開スカラーと自分の drive だけ。

        相互抑制は「自分以外の占有 = 公開合計 field_y − 自分の公開出力」で計算する。
        どちらも自分に見える量（公開スカラーと自分の出力）だけで、他者の私的状態は不要。
        """
        p = self.p
        phi = sigmoid(self.x)
        others_occupancy = field_y - phi  # 公開合計 − 自分の寄与 = 他者の占有度
        dx = (
            -p["alpha"] * self.x + p["w_s"] * phi   # 減衰 + 自己興奮
            - p["w_I"] * others_occupancy           # 相互抑制（公開スカラー経由）
            - p["beta"] * self.a - self.b           # 遅い疲労 + 速い不応
            + p["g_u"] * drive_u                    # 自分への外部刺激
            + rng.gauss(0.0, p["sigma"])            # 揺らぎ
        )
        da = -p["gamma"] * self.a + p["eta"] * self.x
        db = -(1.0 / p["tau_b"]) * self.b + p["kappa_b"] * phi
        self.x += dx
        self.a += da
        self.b += db


def field_occupancy(nodes) -> float:
    """場の公開スカラー = 各ノードの公開出力の合計（平均場 / 占有度）。

    これだけが共有バスに乗る。中央が私的状態を集めて回るのではなく、各自が公開出力を
    場に出し、各自がその合計を感じる（leaderless な集団力学の王道）。
    """
    return sum(node.public_output() for node in nodes)


def run_turn(nodes, drive, rng):
    """ひと区切り（substeps ステップ）積分し、first-passage で勝者を返す。

    各 substep では、まず場の公開スカラー y を読み、各ノードが y と自分の drive だけで
    独立に自分を更新する（固定順は乱数列の再現性のため。情報的には並列）。そのあと
    各ノードが「自分の活性が閾値を越えたか」を自分で検知して公開イベントにする。

    返り値: (winner_id or None, crossings: dict[id->step], integ: dict[id->Σpublic_output])
    """
    p = nodes[0].p
    crossings = {}
    integ = {node.id: 0.0 for node in nodes}
    for s in range(1, p["substeps"] + 1):
        y = field_occupancy(nodes)                 # 公開スカラーを場から読む
        for node in nodes:                          # 各ノードが独立に自分を更新
            node.step(y, drive[node.id], rng)
        for node in nodes:                          # 各ノードが自分の閾値超えを自分で検知
            integ[node.id] += sigmoid(node.x)
            if node.id not in crossings and node.x > p["theta"]:
                crossings[node.id] = s
    return decide_winner(crossings, integ), crossings, integ


def decide_winner(crossings, integ):
    """全員共有の固定ルール: 最初に閾値を越えた者。同時なら公開出力の積み上がりが大きい方。

    引数は公開イベント（誰がいつ越えたか / 公開出力の積分）だけ。各ノードがこの関数を
    ローカルに実行しても同じ勝者を得る = 中央の権威は不要（分散合意ルールの中央実装に
    すぎない）。誰も越えなければ None（＝沈黙。今は誰も話さなくてよい、を表現できる）。

    越えた時刻も公開出力の積分も完全に同点のとき（= 揺らぎ 0 の対称な決定論ケースだけ
    で起きる）は id が最小のノードが勝つ（min がタプル同点時に先頭キー = 挿入順 = id 昇順
    を返すため）。実運用では揺らぎが対称性を破るのでこの第 3 段は効かない。
    """
    if not crossings:
        return None
    return min(crossings, key=lambda i: (crossings[i], -integ[i]))
