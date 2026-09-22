# Paper I・IIの論文対応コード: version 0.1.0

著者は Tsutomu T. Takeuchi, Gayathri Asok, Hai-Xia Ma, Yu Ogane の4名, この順で確定しています. コード・文書はBSD-3-Clause, 保存数値結果とその出典情報はCC BY 4.0です. 論文本文・数値結果・科学計算の内容は変更していません.

## 今回の登録手順

**今回はZenodoでソフトウェア用のDOIを先に予約し, そのDOIを埋め込んだ同じversion 0.1.0をGitHubとZenodoへ登録します.** 2本の論文は同じソフトウェアのversion DOIを引用します. GitHub自動保存と手動Zenodo登録を併用して, 同じ版に2つの記録を作らないでください.

まず `docs/zenodo_registration_ja.md` を開いてください. 現在のファイルに未割当のDOIやGitHub URLを捏造して入れてはいません. Zenodoの予約DOIと実際のGitHubリポジトリURLが分かれば, 最終ZIPと引用情報へ一括反映できます. 予約段階ではまだ公開済みではありません.

## 動作確認を行う場合

元の `mock_data` へファイルを戻す必要はありません. このZIPを展開した `diffusion-geometry-bulk-flow` フォルダを作業場所にします. Jupyterでその場所に移動して, 新しいCodeセルから実行します. 必要な依存packageは `requirements.txt` にあります. 現在の研究用環境を上書きする必要はありません.

```python
%run tools/verify_release.py
%run tools/finalize_deposition.py --check
%run tools/demo.py --output outputs/demo
%run tools/recompute_capacity.py --output outputs/capacity
%run tools/export_table_review.py --output outputs/paper_table_review.csv
```

各セルはShift+Enterです. 同じ出力名が既にあると上書きせず停止します. 再実行時は `outputs/demo_02` など新しい名前を指定します. 登録情報の確認だけなら最初の2行で足ります.

最小demoは元カタログを必要としません. 独立な人工点配置を生成し, 元の関数で90項目の小規模チェックを行います. 論文の物理的な再構成精度を再実証する計算ではありません. capacity処理は保存済み曲線から27組を再計算します. table-review処理は既存の373セルの照合記録を出力するもので, 新しい回帰計算ではありません.

保存CSVからの描画例:

```python
%run tools/redraw_saved_mock1.py --output outputs/paperI_figure06
```

Paper IのFig. 6に対応する1図です. 全31図を1コマンドで生成するものではありません.

## 登録対象

登録するのはこのフォルダのソース・文書・保存結果です. 元カタログNPZ, snapshot, 大容量cache, `.venv`, `outputs/`, 旧収集ZIPは含めません. 著者4名のソフトウェア公開と, 元simulationデータの再配布許諾は区別しています.

22本の科学ソースと174保存ファイルは0.1.0rc1と同じです. 新環境への依存packageのインストールとGitHub Actionsの実行は, 実際に成功するまで確認済みと扱いません. 保存した過去の環境検証と今回の登録用metadata検証は別に記録しています.
