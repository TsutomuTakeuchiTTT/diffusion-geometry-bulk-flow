# 投稿前のGitHub・Zenodo登録: version 0.1.0

## 現在の状態

配布するコード・保存結果, 4名の著者と順序, 版番号0.1.0, ライセンスの範囲を固定しました. DOI・実際のGitHub URL・公開日だけは実アカウントの情報を入れます. この資料の作成でZenodoへのアップロードやPublishは行っていません.

**今回の経路は「ZenodoでDOI予約, 同じDOIを配布物へ記入, GitHub releaseと手動Zenodo登録」です.** 既存のソフトウェア記録を既に作成している場合は新規作成せずその記録を使います. 同じ版をGitHub自動連携でも別途保存しないでください.

## まず行う操作: DOIの予約

Zenodo本番サイトにログインし, `+` → `New upload` を開きます. Resource typeを **Software** とし, 下記タイトルと著者を入力します. Digital Object Identifier欄の `Do you already have a DOI for this upload?` は **No** を選び, **Get a DOI now!** を押します. 表示された予約DOIをコピーして `Save draft` で保存してください. まだPublishは押しません.

論文自身のDOI, 他の研究のDOI, 既存の別ソフトウェアのDOIは使用しません. このdraftを削除すると予約DOIを失うので, 削除しないでください. 予約DOIはPublishまで公開登録されたDOIにはなりません. 原稿で「公開済み」と書くのはPublish後です.

予約DOIをアシスタントへ共有すると,配布物・BibTeX・Availability文へ整合的に反映できます. パスワードやアクセストークンは送らないでください. 実際のGitHubリポジトリURLも判明した時点で使います.

## Zenodoに入力する内容

| 項目 | 値 |
|---|---|
| Resource type | Software |
| Title | Diffusion geometry for galaxy bulk-flow reconstruction |
| Version | 0.1.0 |
| Publisher | Zenodo |
| Language | English |
| Files visibility | Public |
| Publication date | 実際の公開日. この資料の作成日で自動的に固定しない |
| License 1 | BSD 3-Clause, コード・文書 |
| License 2 | Creative Commons Attribution 4.0 International, 保存数値結果とその出典情報 |

4名は単なるContributorsではなく **Creators** に入れます. Name typeはPersonで, Family nameとGiven namesを分け, 次の順を維持します. ORCID自動補完で別人を選ばず, 最後に名前とIDを照合してください. 所属は可能なら機関の候補を選び, 詳細が候補にないときは下記の文字列を使えます.

### 1. Takeuchi, Tsutomu T.

ORCID: `0000-0001-8416-7673`

所属: Division of Particle and Astrophysical Science, Nagoya University, Furo-cho, Chikusa-ku, Nagoya 464-8602, Japan / The Research Center for Statistical Machine Learning, The Institute of Statistical Mathematics, 10-3 Midori-cho, Tachikawa, Tokyo 190-8562, Japan

### 2. Asok, Gayathri

ORCID: `0009-0001-5284-6759`

所属: Division of Particle and Astrophysical Science, Nagoya University, Furo-cho, Chikusa-ku, Nagoya 464-8602, Japan

### 3. Ma, Hai-Xia

ORCID: `0000-0002-5237-9433`

所属: Division of Particle and Astrophysical Science, Nagoya University, Furo-cho, Chikusa-ku, Nagoya 464-8602, Japan

### 4. Ogane, Yu

ORCID: `0009-0001-9463-0673`

所属: Division of Particle and Astrophysical Science, Nagoya University, Furo-cho, Chikusa-ku, Nagoya 464-8602, Japan

## Description: 貼り付け用英文

This software release accompanies Diffusion Geometry for Galaxy Bulk-Flow Reconstruction I: Foundations and Full-Vector Validation, and Diffusion Geometry for Galaxy Bulk-Flow Reconstruction II: Three-Dimensional Potential-Flow Reconstruction from Radial Velocities.

The archive contains 22 selected research scripts, 174 saved numerical-output and configuration records, paper-to-code mappings, explicit numerical conventions, and documented component and result-level checks. It covers diffusion-geometric full-vector reconstruction and radial potential-flow reconstruction on irregular tracer point clouds, including spectral regularization, metric-calibrated gradients, Nyström extension, selection weighting, capacity selection, and controlled measurement-noise experiments.

A self-contained synthetic demonstration does not require the cosmological catalogs or historical caches. Documented executable checks recompute 27 capacity choices from saved curves, export a provenance-linked 373-cell comparison record, and redraw the Paper I Mock-1 scale-comparison figure from saved CSVs. The synthetic component checks are algebraic implementation tests, not a rerun of the papers’ cosmological performance experiments.

The original N-body-derived catalogs, simulation snapshots, large geometry/gradient caches, and manuscript PDFs are not included. Some archived production scripts require external authorized inputs and preceding outputs. This is not a one-command reproduction of all high-mode experiments or all 31 final figures, and is not a validated real-survey analysis pipeline.

Code and documentation are licensed under BSD-3-Clause. Saved numerical records in results/saved/ and their source-provenance metadata are licensed under CC BY 4.0, as specified in results/LICENSE. These are component-specific licenses, not alternative licenses for the same files; omitted catalogs and third-party dependencies are not relicensed.

Private path text has been removed from the distributed copies without changing numerical values. Source/result fingerprints and numerical notes preserve provenance, including two documented last-digit manuscript rounding differences. The record describes a research-software archive and selected saved results, not the peer-review or publication status of the accompanying manuscripts.

Keywords: cosmology; large-scale structure; galaxy peculiar velocities; diffusion geometry; diffusion maps; potential-flow reconstruction; research software; reproducibility

License欄はコード向けBSD-3-Clauseへ変更し, 保存結果向けCC BY 4.0も追加します. Descriptionの適用範囲も残してください. 2ライセンスをどちらか自由に選べるdual licenseとして表示するものではありません. 資金番号・community・論文DOIは未提供なので推測で入力しません.

## DOIが分かった後の手順

DOI, 実際のGitHubリポジトリURL, 公開日を反映した最終フォルダを作り, 内容をGitHubへ登録します. ブラウザは1回100ファイルまでなので, 本パッケージを一度に全選択しないでください. GitHub Desktop/通常のGitを使うか, 元のディレクトリ構成を保って複数回に分けます. README, CITATION.cff, VERSION, .github, .gitignore, .gitattributesをリポジトリ直下に置きます. ZIPを単にCode画面に置くだけではソースの登録になりません.

GitHub ActionsのVerifyを実行して成功を確認し, 確認済みcommitに `v0.1.0` のreleaseを作ります. Release titleは `v0.1.0: Paper-associated code and numerical results` とできます. 説明には `RELEASE_NOTES.md` を使用し, 最終版の `diffusion-geometry-bulk-flow-v0.1.0.zip` と `.sha256` をRelease assetsへ添付します. ZIPの中身は同commitのソース一式と一致させます.

Zenodoでは先ほどの同じdraftを再び開き, **その同一ZIP** をUploadします. GitHub自動生成のSource code(zip)ではなく,明示的に添付した最終ZIPを使えばarchive全体のchecksumも一致します. Related identifierはその実際のGitHub release URLを `is identical to` として追加できます. 新たなDOIを取らず,このdraftが予約したDOIをそのまま使います.

Save draft, Previewの順に進み, 著者4名・順序・ORCID・version・License適用範囲・添付ZIPを確認後, Publishします. 公開後はDOIのページとZIPの取得を確認します. 両論文では同じ **version DOI** を引用します. 両論文のためにソフトウェア記録を2件作る必要はありません.

## 投稿直前の完了条件

GitHubのv0.1.0が公開されていること, Zenodoの同一版がPublishedかつPublicであること, DOIで到達できること, 双方のZIPのSHA-256が一致すること, Creatorsが同じ4名であること, 両論文が同じソフトウェアversion DOIを引用していることを確認します. 予約だけ, draft保存だけではここまで完了したとは扱いません.

**今回の登録準備のために新たなmock_data収集や大規模計算を行う必要はありません.** 追加の数値検証が済んだと主張することもありません.

## 参照した公式案内

- DOI予約: https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- 新規登録と公開: https://help.zenodo.org/docs/deposit/create-new-upload/
- 著者: https://help.zenodo.org/docs/deposit/describe-records/creators/
- 複数ライセンス: https://help.zenodo.org/docs/deposit/describe-records/licenses/
- GitHub release: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- GitHub upload: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

登録手順の確認日: 2026-09-22. 画面の表現が変わった場合は実際の画面を確認してください.
