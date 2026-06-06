# 技術設計

[日本語](technical-design.md) | [English](technical-design.en.md)

この文書は、fravenir の公開向け技術仕様です。実装を理解したい人、フォークして拡張したい人、MCPサーバーとしての設計を確認したい人向けに、記憶モデルと主要フローを整理します。

## 目的

fravenir は、AIキャラクターの長期記憶を扱うMCPサーバーです。

単に過去ログを保存するのではなく、次のような「思い出し方」を実現することを目指しています。

- 会話や出来事をエピソードとして保存する
- キャラクター自身に関係する記憶を探しやすくする
- 古い事実を削除せず、履歴つきで無効化・抑制する
- 意味的に近い記憶やグラフ上で近い記憶を連想できるようにする
- MCPクライアントから安全に呼び出せるよう、キャラクターごとにサーバーを分離する

## アーキテクチャ

```text
MCP client
  |
  v
fravenir MCP server
  |-- memory_write
  |-- memory_search
  |-- memory_get
  |-- memory_explore
  |-- memory_compact
  |
  v
per-character storage
  |-- kv.sqlite
  |-- vdb_memories.db
  |-- vdb_entities.db
  |-- vdb_relations.db
  |-- cache/
```

1つのMCPサーバープロセスは、1つのキャラクターだけを扱います。これにより、別キャラクターの記憶が同じMCPツール空間に混ざることを避けます。

## データモデル

fravenir の中心は、エピソード、エンティティ、リレーションの3種類です。

### episodes

`episodes` は、1回の `memory_write` で保存される記憶の単位です。

主なカラム:

- `content`: 記憶本文
- `kind`: `facts` / `state` / `emo`
- `importance`: 1〜3の重要度
- `valid_from`: 有効になった時刻
- `valid_to`: 無効化された時刻。現在有効なら `NULL`
- `supersedes`: 置き換え元 episode
- `session_id`: 任意のセッション識別子
- `is_suppressed`: compact により抑制されたか

### entities

`entities` は、エピソードから抽出される人物、概念、場所、作品、感情などです。

主なカラム:

- `canonical_name`: 正規名
- `entity_type`: 種別
- `description`: 説明文
- `is_self`: キャラクター自身を表すエンティティか
- `self_weight`: 自己ハブとしての重み
- `decay_rate`: ACT-R活性化で使う減衰率
- `curated_at`: 人手で編集済みであることを示す時刻

表記揺れは `entity_aliases` で管理します。

### relations

`relations` は、episode/entity 間、または entity/entity 間の関係です。

主なカラム:

- `src_type`, `src_id`: 関係元
- `dst_type`, `dst_id`: 関係先
- `predicate`: `mentions`、`part_of`、`likes` などの関係名
- `strength`: 関係の強さ
- `fan_out`: 関係元から伸びる関連数
- `valid_from`, `valid_to`, `supersedes`: 履歴管理

### merge_candidates

`merge_candidates` は、夜間整理で見つかった「同じものかもしれないエンティティ」の候補を保存します。

LLM意味判定を使う場合は、判定ラベル、信頼度、理由、試行回数、解決時刻も記録します。

## 記憶スコアリング

検索時の順位づけは、ACT-R風の活性化とベクトル類似度を組み合わせます。

```text
score = activation + alpha_similarity * vector_similarity + alpha_importance * importance
```

activation は主に次の要素から計算されます。

- 過去に参照された回数
- 最後に参照されてからの時間
- エンティティやリレーションを通じた連想強度
- 自己ハブに関係する記憶への補正

これにより、単純な全文検索ではなく、「最近よく使われた」「重要」「意味的に近い」「キャラクター自身に関係がある」記憶が上がりやすくなります。

## 主要フロー

### memory_write

`memory_write` は記憶を1件保存します。

主な流れ:

1. `episodes` に本文を保存する
2. 本文を埋め込み、`vdb_memories.db` に保存する
3. LLM抽出が有効なら、本文から entities / relations を抽出する
4. 新しい entities / relations をDBへ保存する
5. 矛盾しやすい関係があれば、古い事実に `valid_to` を立てる

LLM抽出が失敗しても、エピソード本文と埋め込みは残ります。

### memory_search

`memory_search` はクエリに関連するエピソードを返します。

主な流れ:

1. クエリを埋め込む
2. エピソードベクトルを検索する
3. 関連エンティティやリレーションを使って候補を広げる
4. ACT-R風スコアで再ランキングする
5. アクセス履歴を更新する

### memory_get

`memory_get` は、キャラクターの自己紹介や最近の状態をコンパクトに返すための互換APIです。

会話モデルへ渡す場合は、返却された記憶本文を命令として扱わないよう、prompt injection対策を行うことを推奨します。

### memory_explore

`memory_explore` は、episode または entity を起点に、グラフを1ホップ深掘りするツールです。

`memory_search` が「関連しそうな記憶を探す」入口だとすると、`memory_explore` は「見つかった記憶の周辺をさらに覗く」ためのツールです。

主な特徴:

- 起点は `node_type` と `node_id` で指定する
- 現在は `depth=1` を基本とする
- `exclude_episode_ids` / `exclude_entity_ids` で訪問済みノードを避けられる
- `include_archived` / `include_suppressed` で無効化・抑制済みノードも含められる

### memory_compact

`memory_compact` は、記憶グラフを整理するメンテナンス処理です。

主な処理:

- relation の `fan_out` を再計算する
- relation の `strength` を更新する
- 活性化が低いエピソードを抑制する
- 類似エンティティを `merge_candidates` に追加する

`--use-llm` を付け、かつ `semantic_judge.enabled` が有効な場合は、LLMによる意味判定も行います。

## MCPインタフェース

代表的なツール:

| ツール | 役割 |
|---|---|
| `memory_write` | 記憶を1件保存する |
| `memory_search` | 関連記憶を検索する |
| `memory_get` | 自己紹介・最近の状態を返す |
| `memory_explore` | グラフを1ホップ深掘りする |
| `memory_delete` | エピソードを論理削除する |
| `memory_trace` | supersedes の履歴を辿る |
| `memory_compact` | 記憶グラフを整理する |

キャラクターIDはツール名に含めません。キャラクターごとにMCPサーバーを分け、サーバー名で識別します。

## ストレージ

キャラクターごとに、以下のようなディレクトリを使います。

```text
data/<character_id>/
  kv.sqlite
  vdb_memories.db
  vdb_entities.db
  vdb_relations.db
  cache/
```

`characters/<character_id>/` には、編集可能な `config.yaml` と `seed.yaml` を置きます。

`data/` と `characters/` はGitの追跡対象外にする想定です。

## LLM利用

fravenir のLLM利用は任意です。

主な用途:

- `memory_write` 時のエンティティ/リレーション抽出
- `memory_compact --use-llm` 時の重複候補・矛盾候補の意味判定

LLMエンドポイントは OpenAI互換APIを想定しています。ローカルLLMを使う場合は `config.yaml` の `extraction` と `semantic_judge` を環境に合わせて設定します。

## セキュリティと運用上の前提

- キャラクターごとにプロセスとデータディレクトリを分けます。
- `data/` と `characters/` は公開リポジトリへ含めないでください。
- HTTP transport や Admin UI を使う場合は、ローカルまたは閉じたネットワーク内での運用を基本にしてください。
- Admin UI はインターネットへ直接公開する前提ではありません。
- 記憶本文はユーザー由来データとして扱い、LLMに渡す際は prompt injection 対策を行ってください。

## 拡張ポイント

- ベクトルDBを sqlite-vec 以外へ差し替える
- グラフDBを外部DBへ切り出す
- compact の整理ルールを追加する
- LLM意味判定のモデルやプロンプトを差し替える
- MCPクライアントごとに表示・注入方針を調整する
