# Admin UI 運用

`fravenir admin-serve` は、ブラウザからキャラクターの記憶DBを確認し、エンティティの description や aliases を手で整えるための任意機能です。

## 安全上の注意

Admin UI はインターネットへ直接公開する前提ではありません。

基本的には、以下のどれかで使ってください。

- ローカルPC上で `127.0.0.1` にbindして使う
- VPNなどの閉じたネットワーク内だけで使う
- SSH tunnel やリバースプロキシの内側に置き、別途認証とアクセス制御を用意する

Admin UIには書き込みAPIがあります。外部公開する場合は、fravenir単体の設定だけに頼らず、TLS、認証、アクセス制御、CSRF対策を含む追加の保護を用意してください。

## 起動

ローカルで起動する場合:

```bash
uv run fravenir admin-serve mychar --host 127.0.0.1 --port 8281
```

ブラウザで開きます。

```text
http://127.0.0.1:8281/
```

## HTTP Basic auth

環境変数 `FRAVENIR_ADMIN_USER` と `FRAVENIR_ADMIN_PASSWORD` を両方セットすると、HTTP Basic auth が有効になります。

```bash
FRAVENIR_ADMIN_USER=admin \
FRAVENIR_ADMIN_PASSWORD='<random-long-password>' \
uv run fravenir admin-serve mychar --host 127.0.0.1 --port 8281
```

片方だけでは有効になりません。パスワードは十分長いランダム文字列を使ってください。

## systemdで常駐させる

サンプル: [fravenir-admin@.service](fravenir-admin@.service)

```ini
[Unit]
Description=fravenir admin UI for character %i
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<USER>
WorkingDirectory=<REPO_PATH>
Environment=PATH=/home/<USER>/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/<USER>/.local/bin/uv run fravenir admin-serve %i --host 127.0.0.1 --port 8281
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

サンプルファイル内の `<USER>` と `<REPO_PATH>` を編集してから配置します。

```bash
sudo cp docs/operations/fravenir-admin@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fravenir-admin@mychar.service
```

## systemdで認証情報を渡す

unitファイルへパスワードを直接書くと `systemctl cat` などで見えやすくなります。`EnvironmentFile=` で別ファイルから読む方が扱いやすいです。

例: `/etc/fravenir/admin-mychar.env`

```text
FRAVENIR_ADMIN_USER=admin
FRAVENIR_ADMIN_PASSWORD=<random-long-password>
```

パーミッション例:

```bash
sudo install -d -m 0750 -o root -g <USER> /etc/fravenir
sudo install -m 0640 -o root -g <USER> admin-mychar.env /etc/fravenir/admin-mychar.env
```

unit側:

```ini
[Service]
EnvironmentFile=-/etc/fravenir/admin-%i.env
ExecStart=/home/<USER>/.local/bin/uv run fravenir admin-serve %i --host 127.0.0.1 --port 8281
```

反映:

```bash
sudo systemctl daemon-reload
sudo systemctl restart fravenir-admin@mychar.service
```

## 動作確認

```bash
systemctl status fravenir-admin@mychar.service
sudo journalctl -u fravenir-admin@mychar.service -n 50 --no-pager
```

認証を有効化している場合:

```bash
curl -i http://127.0.0.1:8281/api/stats
curl -s -u admin:<password> http://127.0.0.1:8281/api/stats
```

## 複数キャラ運用

複数キャラのAdmin UIを同時に動かす場合、ポートが衝突します。キャラごとにポートを分けるか、必要なときだけ対象キャラのAdmin UIを起動してください。

## トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `Address already in use` | ポート衝突 | 別ポートにするか、既存serviceを停止 |
| `Permission denied` | `User=` と `data/<character_id>/` の所有権不一致 | `chown -R <USER>:<USER> data/<character_id>/` |
| 401が返る | Basic authが有効 | `-u user:password` を付ける |
| 500が返る | 古いDBスキーマの可能性 | [migrations.md](migrations.md) を確認 |

## 関連

- [systemd_timer.md](systemd_timer.md): MCP本体とcompact定期実行
- [migrations.md](migrations.md): 古いDBを現在のスキーマへ追いつかせる保守用リファレンス
- `src/fravenir/admin/`: 実装本体
