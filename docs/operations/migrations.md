# DBアップグレード用マイグレーション

`uv run fravenir migrate ...` 系コマンドは、古い fravenir DB を現在のスキーマへ追いつかせるための保守用コマンドです。

## 新規導入では基本不要

新しく `fravenir create-character <id>` で作成したキャラクターDBには、現行スキーマが最初から入ります。そのため、通常の新規導入や普段の運用では、このドキュメントの手順は基本的に不要です。

このドキュメントが必要になるのは、主に次のような場合です。

- 以前の開発版で作成した `data/<character_id>/kv.sqlite` を使い続けたい
- 古いリリースから新しいリリースへアップグレードした
- Admin UI や `compact --use-llm` で、DBにカラムが無いことによるエラーが出た

## 基本方針

- マイグレーションはキャラ単位で実行します。
- 既に適用済みの場合は no-op になります。
- 既存データの削除ではなく、主にカラムやテーブルの追加を行います。
- 実行前に `--dry-run` で確認できます。

## 標準フロー

```bash
uv run fravenir migrate <subcommand> <character_id> --dry-run
uv run fravenir migrate <subcommand> <character_id> --yes
```

`nothing to do` や `already migrated` が表示されれば、適用済みです。

## 利用可能なサブコマンド

| サブコマンド | 追加内容 | 主な用途 |
|---|---|---|
| `session-id` | `episodes.session_id` と関連インデックス | 古いDBの session 情報を専用カラムへ移す |
| `judge-columns` | `merge_candidates.judge_*` | `compact --use-llm` の意味判定情報 |
| `resolved-at` | `merge_candidates.resolved_at` | resolve済み候補の解決時刻 |
| `curated-and-audit` | `entities.curated_at`、`admin_audit_log` | Admin UIでの手動編集と監査ログ |

## 実行例

```bash
uv run fravenir migrate curated-and-audit mychar --dry-run
uv run fravenir migrate curated-and-audit mychar --yes
```

サービス運用中にDBを書き換える場合は、必要に応じてMCPサーバーやAdmin UIを一度止めてから実行してください。

```bash
sudo systemctl stop fravenir@mychar.service
sudo systemctl stop fravenir-admin@mychar.service

uv run fravenir migrate curated-and-audit mychar --yes

sudo systemctl start fravenir@mychar.service
sudo systemctl start fravenir-admin@mychar.service
```

## トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `database is locked` | MCPサーバーやAdmin UIがDBを使用中 | 該当serviceを止めてから再実行 |
| `kv.sqlite` が無い | キャラクターDBが未作成 | `fravenir create-character <id>` を先に実行 |
| `nothing to do` と表示される | 適用済み | 追加対応は不要 |
| Admin UIで500が出る | 古いDBに必要なテーブル/カラムが無い | `curated-and-audit` を確認 |
| `compact --use-llm` でDBエラー | 古いDBに判定用カラムが無い | `judge-columns` と `resolved-at` を確認 |

## 関連

- `src/fravenir/migrations/`: マイグレーション実装
- `src/fravenir/cli.py`: `fravenir migrate` コマンド
- [systemd_timer.md](systemd_timer.md): service停止/再起動の運用
