# クイックスタート

[日本語](quickstart.md) | [English](quickstart.en.md)

このガイドでは、ローカルでキャラクターを1人作成し、fravenir をMCPサーバーとして起動します。

## 前提

- Python 3.12
- uv
- 初回起動時に埋め込みモデルをダウンロードできる環境

fravenir は、生成されたDBを `data/` に、キャラクター設定を `characters/` に保存します。

## 1. 依存関係をインストールする

```bash
uv sync
```

最初のキャラクター作成や検索時に、埋め込みモデル `cl-nagoya/ruri-v3-310m` がダウンロードされることがあります。

## 2. キャラクター設定ファイルを作る

サンプルファイルをコピーします。

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml
```

両方のファイルを編集して、サンプルのIDを自分のキャラクターIDに置き換えます。

```yaml
character:
  id: mychar
```

```yaml
identity:
  canonical_name: mychar
```

`seed.yaml` には、キャラクターの初期 identity、personality、最初の記憶を書きます。`config.yaml` では、保存先、埋め込み、活性化パラメータ、LLM抽出、ログ、サーバー設定を調整します。

## 3. キャラクターを作成する

```bash
uv run fravenir create-character mychar
```

このコマンドで `data/mychar/` にSQLite DBなどの実行時データが作成されます。ステップ2で配置した `characters/mychar/` の設定を読み込んで初期化します。

作成結果を確認します。

```bash
uv run fravenir show-character mychar
```

## 4. MCPサーバーを起動する

ローカルのMCPクライアントから使う場合は、デフォルトの stdio transport を使います。

```bash
uv run fravenir serve --character mychar
```

MCPクライアント設定例:

```json
{
  "mcpServers": {
    "fravenir_mychar": {
      "command": "uv",
      "args": ["run", "fravenir", "serve", "--character", "mychar"]
    }
  }
}
```

サーバーは `memory_write`、`memory_search`、`memory_get`、`memory_explore`、`memory_trace`、`memory_delete`、`memory_compact` などのツールを公開します。

## 5. seedの変更をあとから反映する

`characters/mychar/seed.yaml` を編集したあと、次のコマンドで新しい seed 内容を反映できます。

```bash
uv run fravenir init-character mychar --force
```

既存キャラクターに personality や initial episodes を追加したいときに使います。

## 6. メンテナンスを実行する

compact を手動で実行します。

```bash
uv run fravenir compact mychar --dry-run
uv run fravenir compact mychar
```

サーバー上で定期実行したい場合は [systemd timer](../operations/systemd_timer.md) を参照してください。

## 任意: LLM抽出

fravenir は、OpenAI互換エンドポイントを使って記憶を整理できます。

- `memory_write` 時: エピソード本文からエンティティと関係を抽出し、グラフに追加します。
- `compact --use-llm` 時: 重複候補、関係の向き、矛盾候補を意味的に判定します。

`examples/config.yaml` には、ローカルエンドポイント例が入っています。

```yaml
extraction:
  enabled: true
  base_url: http://127.0.0.1:8080/v1
  api_key: dummy
```

まだLLM抽出用のエンドポイントを用意していない場合、本番運用前に `extraction.enabled` を `false` にしてください。その状態でも、キャラクター作成や基本的な記憶DBの利用はできます。

`compact --use-llm` を使う場合は、あわせて `semantic_judge.enabled` を `true` にし、`semantic_judge.base_url` と `semantic_judge.model` を環境に合わせて設定します。

## 次に読むもの

- [サンプル設定](../../examples/README.md): `config.yaml` と `seed.yaml` の内容。
- [運用ドキュメント](../operations/systemd_timer.md): 常駐サーバーや定期実行の設定。
- [技術設計](../design/technical-design.md): 内部を改造したいときの設計資料。
