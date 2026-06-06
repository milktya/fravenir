# fravenir

[日本語](README.md) | [English](README.en.md)

> [!NOTE]
> 個人の趣味プロジェクトです。issue や PR への対応は保証できません。

fravenir は、AIキャラクターのための記憶MCPサーバーです。
キャラクターが会話の中で得た出来事を保存し、あとから検索し、自分自身に関する記憶を見つけやすくし、古くなった事実や矛盾した事実を履歴つきで整理できるようにします。

## 何ができるか

fravenir は主に4つの仕組みを組み合わせています。

- **エピソード記憶**: 短い記憶を時刻つきのエピソードとして保存します。
- **自己ハブ**: identity や personality をグラフ上のエンティティとして持ち、キャラクター自身に関係する記憶を探しやすくします。
- **ACT-R風の活性化スコア**: 新しさ、参照履歴、重要度、意味的な近さ、グラフ上の関連を組み合わせて、思い出しやすい記憶を順位づけします。
- **LLMによる情報整理**: 任意で、書き込まれた記憶からエンティティ/関係を抽出したり、夜間整理で重複候補や矛盾候補を意味判定したりできます。

MCPクライアントから使うことを前提にしていて、キャラクターごとに別々のMCPサーバーとして起動します。ツール名は `memory_write`、`memory_search`、`memory_get`、`memory_explore` などのまま使います。

## 安全上の注意

Admin UI は任意機能です。ローカル環境またはVPNなどの閉じたネットワーク内で使うことを想定しており、インターネットへ直接公開する前提ではありません。
Admin UIを使う場合は、HTTP Basic auth を有効化し、`127.0.0.1` やVPN内IPへのbindを基本にしてください。外部公開が必要な場合は、必ず別途リバースプロキシ、TLS、認証、アクセス制御を用意してください。

## クイックスタート

依存関係をインストールします。

```bash
uv sync
```

サンプルファイルをコピーして編集します。

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml
```

`character.id` と `identity.canonical_name` を `mychar` に揃えてから、キャラクターを作成します。

```bash
uv run fravenir create-character mychar
```

MCPサーバーを起動します。

```bash
uv run fravenir serve --character mychar
```

詳しい初回手順は [docs/setup/quickstart.md](docs/setup/quickstart.md) を参照してください。

## MCPクライアント設定例

MCPクライアント側では、キャラクターごとにサーバー名を分けます。サーバー内のツール名は `memory_*` のままです。

```json
{
  "mcpServers": {
    "fravenir_mychar": {
      "command": "fravenir",
      "args": ["serve", "--character", "mychar"]
    }
  }
}
```

## 主なコマンド

```bash
uv run fravenir list-characters
uv run fravenir show-character <id>
uv run fravenir init-character <id> --force
uv run fravenir compact <id> [--dry-run] [--use-llm]
uv run fravenir resolve list <id>
uv run fravenir export <id> --out file.json
uv run fravenir import file.json <id>
```

## ドキュメント

- [クイックスタート](docs/setup/quickstart.md): 最初のキャラクター作成とMCP起動手順。
- [サンプル設定](examples/README.md): `config.yaml` と `seed.yaml` の説明。
- [運用ドキュメント](docs/operations/): MCP常駐、compact定期実行、admin UI、prompt injection対策。
- [DBアップグレード](docs/operations/migrations.md): 古いDBを現在のスキーマへ追いつかせる保守用リファレンス。
- [技術設計](docs/design/technical-design.md): 記憶モデル、ストレージ、スコアリング、MCPインタフェースの設計。

公開ドキュメントの入口は [docs/INDEX.md](docs/INDEX.md) です。

## 開発

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

Python 3.12、Pydantic v2、SQLite、sqlite-vec、sentence-transformers、structlog、FastMCP を使っています。

## ランタイムデータ

キャラクター設定は `characters/`、生成されたDBは `data/` に置かれます。

## ライセンス

[MIT](LICENSE)
