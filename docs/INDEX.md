# fravenir ドキュメント

[日本語](INDEX.md) | [English](INDEX.en.md)

このディレクトリには、fravenir の公開向けドキュメントを置いています。

## 1. 概要

まず fravenir が何をするものか知りたい場合は、ここから読んでください。

- [README](../README.md): プロジェクト概要、基本概念、主なコマンド。

## 2. 導入と運用

実際にインストールしたり、サーバーとして動かしたりするときに読む文書です。

- [クイックスタート](setup/quickstart.md): 最初のキャラクター作成とMCPサーバー起動。
- [サンプル設定](../examples/README.md): `config.yaml` と `seed.yaml` の説明。
- [systemd運用](operations/systemd_timer.md): MCP本体の常駐と `fravenir compact` の定期実行。
- [admin server](operations/admin_server.md): 任意の管理UIをローカル/閉域で使うための運用メモ。
- [migrations](operations/migrations.md): 古いDBを現在のスキーマへ追いつかせる保守用リファレンス。
- [prompt injection notes](operations/prompt_injection.md): 記憶本文をLLMに渡す前の安全な囲い方。

## 3. 設計と技術仕様

内部を改造したり、フォークしたり、仕様を詳しく確認したい人向けの文書です。

- [技術設計](design/technical-design.md): アーキテクチャ、ストレージ、ACT-R活性化、検索/書き込みフロー、グラフ探索、MCPインタフェース、設計上の判断。

## 読む順番の目安

- 初めて使う人: README -> クイックスタート -> サンプル設定。
- サーバー運用する人: クイックスタート -> systemd運用 -> 必要ならadmin server。
- 古いDBを引き継ぐ人: クイックスタート -> migrations -> systemd運用。
- コントリビューター/フォークしたい人: README -> 技術設計。

