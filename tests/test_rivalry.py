"""rivalry.py の最小テスト（標準ライブラリ unittest のみ・依存なし）。

ここで固定するのは、README / 記事が主張する2つのこと:
  柱1: これは「rivalry という名のオーケストレーター」ではなく peer-mesh（分散）である
       → docstring や名前ではなく、実行できる検査で不変条件を縛る:
         (1) step() は他ノードを引数に取らない（公開スカラー＋自分の drive だけ）
         (2) 場の公開スカラーに乗るのは公開出力のみ（私的状態 a,b は混ざらない）
         (3) 勝者は公開イベントだけの固定ルール（first-passage）で、評価地点に依らない
  柱2: 同じ力学がパラメータ次第で〈独占／健全〉のレジームを取る

実行:
    python3 -m unittest discover -s tests -v
    （または python3 tests/test_rivalry.py 単体でも動く）
"""

import inspect
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rivalry import (  # noqa: E402
    DEFAULTS,
    RivalryNode,
    decide_winner,
    field_occupancy,
    run_turn,
    sigmoid,
)

# 独占レジーム: 独占を防ぐ機構（速い不応・遅い疲労・揺らぎ）を全部切ったパラメータ。
MONOPOLY = dict(DEFAULTS, kappa_b=0.0, beta=0.0, sigma=0.0)


def _run(n, stim, *, params=DEFAULTS, seed=0, turns=200, repeat_suppress=0.5):
    """rivalry を turns ターン回し、各ノードの発言回数と沈黙回数を返す。"""
    rng = random.Random(seed)
    nodes = [RivalryNode(i, params) for i in range(n)]
    counts = [0] * n
    silent = 0
    last = None
    for _ in range(turns):
        drive = {i: (stim[i] * repeat_suppress if i == last else stim[i]) for i in range(n)}
        winner, _, _ = run_turn(nodes, drive, rng)
        if winner is None:
            silent += 1
            last = None
        else:
            counts[winner] += 1
            last = winner
    return counts, silent


class TestDecideWinner(unittest.TestCase):
    """柱1-(3): 勝者は『公開イベントだけの固定ルール』で決まる（first-passage）。"""

    def test_earliest_crosser_wins(self):
        # 最も早く閾値を越えた者（step 2 の 1）が勝つ
        self.assertEqual(decide_winner({0: 5, 1: 2, 2: 8}, {0: 1.0, 1: 1.0, 2: 1.0}), 1)

    def test_same_step_tiebreak_by_public_output(self):
        # 同着なら、公開出力の積み上がり（integ）が大きい方
        self.assertEqual(decide_winner({0: 3, 1: 3}, {0: 0.4, 1: 0.9}), 1)

    def test_no_crossing_is_silence(self):
        # 誰も越えなければ None（＝今は誰も話さなくてよい、を表現できる）
        self.assertIsNone(decide_winner({}, {0: 0.0, 1: 0.0}))

    def test_rule_is_location_independent(self):
        # 同じ公開イベントなら、評価する場所も dict の並び順も問わず同じ勝者になる
        # = 中央の権威に依らない（各ノードがローカルに同じ関数を実行しても結論が一致する）。
        # 勝者が一意に決まる入力（最早 crosser = 1）で、挿入順を変えても勝者が動かないことを縛る。
        ascending = decide_winner({0: 4, 1: 2, 2: 3}, {0: 0.5, 1: 0.5, 2: 0.5})
        descending = decide_winner({2: 3, 1: 2, 0: 4}, {2: 0.5, 1: 0.5, 0: 0.5})
        self.assertEqual(ascending, 1)  # 最早 crosser (step 2) は順序に依らず 1
        self.assertEqual(ascending, descending)


class TestPeerMeshInvariants(unittest.TestCase):
    """柱1-(1)(2): 中央の進行役はいない、を構造の不変条件で縛る。"""

    def test_step_takes_only_public_scalar_and_own_drive(self):
        # (1) step() の引数は (self, field_y, drive_u, rng) だけ。
        # 他ノードのコレクションを受け取らない = 構造的に他者の内部を読めない。
        params = list(inspect.signature(RivalryNode.step).parameters)
        self.assertEqual(params, ["self", "field_y", "drive_u", "rng"])

    def test_node_feels_only_the_public_scalar(self):
        # (1) 同じ field_y・drive_u・乱数なら同じ更新 = 各ノードは他者の私的状態ではなく
        #     公開合計スカラーだけを感じている（平均場結合）。
        n1, n2 = RivalryNode(0), RivalryNode(0)
        n1.step(1.3, 0.5, random.Random(7))
        n2.step(1.3, 0.5, random.Random(7))
        self.assertEqual((n1.x, n1.a, n1.b), (n2.x, n2.a, n2.b))
        # 公開スカラー field_y を変えれば更新も変わる（無視しておらず、確かに「感じて」いる）。
        n3 = RivalryNode(0)
        n3.step(2.6, 0.5, random.Random(7))
        self.assertNotEqual(n3.x, n1.x)

    def test_field_occupancy_is_sum_of_public_outputs_only(self):
        # (2) 共有バスに乗るのは公開出力 sigmoid(x) の合計だけ。私的状態 a,b は混ざらない。
        nodes = [RivalryNode(i) for i in range(3)]
        nodes[0].x, nodes[1].x, nodes[2].x = -0.4, 0.2, 1.1
        expected = sum(sigmoid(node.x) for node in nodes)
        self.assertAlmostEqual(field_occupancy(nodes), expected)
        nodes[0].a, nodes[0].b = 99.0, 99.0  # 私的状態をいじっても…
        self.assertAlmostEqual(field_occupancy(nodes), expected)  # …公開スカラーは不変


class TestRegimes(unittest.TestCase):
    """柱2: 同じ力学がパラメータ次第で〈独占／健全〉のレジームを取る。"""

    def test_monopoly_when_guards_removed(self):
        # 不応・疲労・揺らぎを切ると、対称な 3 体は完全に同期し一体が椅子を独占する
        # （sigma=0 で完全決定論。最初に並んだ者＝id 最小が勝ち続ける）。
        counts, silent = _run(3, [1.0, 1.0, 1.0], params=MONOPOLY, turns=50, repeat_suppress=1.0)
        self.assertEqual(counts, [50, 0, 0])
        self.assertEqual(silent, 0)

    def test_healthy_alternation_no_monopoly(self):
        # フル機構なら、2 体は独占せず代わる代わる回る（どちらの share も 0.5 周辺）。
        tot = [0, 0]
        for seed in range(5):
            counts, _ = _run(2, [1.0, 1.0], seed=seed, turns=200)
            tot = [t + c for t, c in zip(tot, counts)]
        speak = sum(tot)
        self.assertGreater(speak, 0)
        share0 = tot[0] / speak
        self.assertTrue(0.30 < share0 < 0.70, f"share0={share0:.2f} が 0.5 周辺でない")


if __name__ == "__main__":
    unittest.main(verbosity=2)
