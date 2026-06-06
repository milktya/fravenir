# サンプル設定

[日本語](README.md) | [English](README.en.md)

このディレクトリには、キャラクター作成用のサンプルファイルを置いています。

- `config.yaml`: `characters/<id>/config.yaml` の雛形。
- `seed.yaml`: `characters/<id>/seed.yaml` の雛形。

初回手順を通して確認したい場合は [クイックスタート](../docs/setup/quickstart.md) を参照してください。

## 推奨する使い方

まずファイルをコピーしてから編集します。

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml

uv run fravenir create-character mychar
```

## スモークテスト用の直接指定

サンプルファイルをそのまま渡すこともできます。

```bash
uv run fravenir create-character mychar \
  --config examples/config.yaml \
  --seed examples/seed.yaml
```

これは簡単な動作確認向けです。サンプルの `seed.yaml` は identity に `example` を使っているため、実際のキャラクターではコピーして編集してください。

## 編集する主な項目

`config.yaml` では、主に以下を編集します。

- `character.id`
- OpenAI互換のローカルLLMを使う場合は `extraction.enabled` と `extraction.base_url`
- HTTP運用する場合は `server.transport`、`server.host`、`server.port`

`seed.yaml` では、主に以下を編集します。

- `identity.canonical_name`
- `identity.aliases`
- `identity.description`
- `personality`
- `initial_episodes`
