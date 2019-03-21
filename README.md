# MtG風Text Generation @ LSTM
LSTMを用いた文章自動生成<br>
学習にMtGのフレーバーテキストを使用

[参考 : keras/example/lstm_text_generation.py](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)

## 環境
python 3.6.2<br>
tensorflow 1.12.0<br>
keras 2.2.4

## データセット
<dl>
<dt>data/mtg_flavor.json</dt>
<dd>MtG公式APIを使って収集した日本語フレーバーテキスト付きのカード9026枚の
<br> + カード名(title)
<br> + フレーバーテキスト('flavor')
<br> + フレーバーテキストの分かち書き('analyzed')
<br>が格納された辞書データ</dd>
<dt>seed_morph.json</dt>
<dd>テキスト生成時にseedとして用いる各フレーバーの冒頭の形態素</dd>
<dt>vocab.json</dt>
<dd>インデックス⇔形態素の変換を行うための形態素の集合</dd>
<dl>

[MtG公式API](https://docs.magicthegathering.io/)

## sample
10文を生成

 + 若いベイと動くことかぎり、もうなく餌のためになり。――ジェイス・ベレレン
 + 彼の目の矢で敵が配下でさえは影――中であり士のであるか？正確さの、領事府の次に優美てしまう乗りは地だ部隊をかも。――ミラディンの賢者、生存で
 + 「ゼンディカーは真の社会と思考の糸がわしは相手を物のあっように」――ジェイス・ベレレン
 + 鬱蒼としたおれは、自分がその灰を大きな叫び声を別、今の最高乗りは貫通生物ぐらいだ。」
 + 象の声は光だ。為すぬ石が剣だ生真面目な破滅な――となっている。
 + 「あいつらみたい姿と今というわけで、はないわ。私をそれをにするように自分である。特にも十分ている。
 + ここでは罪あるとは二つ生き物れが内と光だ。
 + 「私の物質回も団でが、災害は襲う通りが教えを焼いたんだへだ。」――実験式の？心、忠誠を得るように、根は体をだ。――的な栄光でわけでいった。
 + 「我々は己の精神を破壊に見たとき、怒りに何した苦しみ侵入人は発想てください。――相隊長大――隊長、だろう――ボロスの軍団をところをもある。あいつらはまるで翼を回る。
 + 「私が死ぬかもならばが息だ。」――試練を抜けることでしようによる火をかぎり、お前の中で仲間にたものが取れ――ぐらいだ。――ヴィダルケンの奉仕し基体の中ではラヴニカ通りで道を時間大きな獲物を知っている。」――ドロモカの戦士、ウルドナンな時代の者が祝福と見捨てられては何もいく。

精度はそんなに良くないです<br>

## 反省

+ 固有名詞の生成とテキストの生成は分離させた方が良さそう<br>
+ 形態素同士のつながりが不自然になる<br>
    → 品詞や活用形、格フレームのラベルを付与？<br>
    → 学習に使用する各シーケンスの長さを伸ばす？<br>
+「」の扱い<br>
    →左右で揃えるには長いシーケンスの学習をする必要がある？
    →いっそストップワードとして除外する？
