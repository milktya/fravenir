# systemd 運用

fravenir を Linux サーバーで常駐運用するための systemd サンプルです。

このドキュメントでは、次の3つを扱います。

- MCPサーバー本体を常駐させる `fravenir@.service`
- compactを1回実行する `fravenir-compact@.service`
- compactを毎日実行する `fravenir-compact@.timer`

Admin UI の常駐運用は [admin_server.md](admin_server.md) を参照してください。

## 前提

- systemd を持つ Linux サーバーを想定しています。
- `<USER>`、`<REPO_PATH>`、`<CHARACTER_ID>` は自分の環境に合わせて置き換えてください。
- `User=` は必ず指定してください。rootで動かすと `data/<character_id>/` の所有権が崩れやすくなります。
- MCPサーバーのサンプルは `127.0.0.1:8280` にbindします。外部から使う場合は、VPN、SSH tunnel、リバースプロキシなど、別途閉じた経路を用意してください。

## MCPサーバー本体

サンプル: [fravenir@.service](fravenir@.service)

```ini
[Unit]
Description=fravenir MCP server for character %i
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<USER>
WorkingDirectory=<REPO_PATH>
Environment=PATH=/home/<USER>/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/<USER>/.local/bin/uv run fravenir serve --character %i --transport streamable-http --host 127.0.0.1 --port 8280
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

`%i` は systemd template unit のインスタンス名です。たとえば `fravenir@mychar.service` として起動すると、`%i` は `mychar` に展開されます。

## compact定期実行

compactは、1回だけ実行する service と、それを定期発火する timer の組み合わせで運用します。

サンプル:

- [fravenir-compact@.service](fravenir-compact@.service)
- [fravenir-compact@.timer](fravenir-compact@.timer)

### service

```ini
[Unit]
Description=fravenir compact for character %i
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=<USER>
WorkingDirectory=<REPO_PATH>
Environment=PATH=/home/<USER>/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/<USER>/.local/bin/uv run fravenir compact %i
StandardOutput=journal
StandardError=journal
```

### timer

```ini
[Unit]
Description=Run fravenir compact daily for character %i

[Timer]
OnCalendar=*-*-* 05:00:00
Persistent=true
Unit=fravenir-compact@%i.service

[Install]
WantedBy=timers.target
```

`Persistent=true` により、サーバー停止中に予定時刻を過ぎた場合でも、次回起動後に1回実行されます。

## インストール手順

サンプルファイル内の `<USER>` と `<REPO_PATH>` を編集してから配置します。

```bash
sudo cp docs/operations/fravenir@.service /etc/systemd/system/
sudo cp docs/operations/fravenir-compact@.service /etc/systemd/system/
sudo cp docs/operations/fravenir-compact@.timer /etc/systemd/system/

sudo systemctl daemon-reload
```

MCPサーバーを起動します。

```bash
sudo systemctl enable --now fravenir@<CHARACTER_ID>.service
```

compact timerを有効化します。

```bash
sudo systemctl enable --now fravenir-compact@<CHARACTER_ID>.timer
```

## 動作確認

MCPサーバー:

```bash
systemctl status fravenir@<CHARACTER_ID>.service
sudo journalctl -u fravenir@<CHARACTER_ID>.service -n 50 --no-pager
```

compact timer:

```bash
systemctl status fravenir-compact@<CHARACTER_ID>.timer
systemctl list-timers 'fravenir-compact@*.timer'
```

compactを手動で1回実行:

```bash
sudo systemctl start fravenir-compact@<CHARACTER_ID>.service
sudo journalctl -u fravenir-compact@<CHARACTER_ID>.service -n 50 --no-pager
```

## 複数キャラ運用

複数キャラを同時にHTTP transportで起動する場合、ポートが衝突します。キャラごとにポートを分けたい場合は、unitをコピーして個別に編集するか、`EnvironmentFile=` でポートを外出ししてください。

単一キャラ運用なら、まずはサンプルの `127.0.0.1:8280` のままで十分です。

## トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `status=203/EXEC` | `uv` のパスが違う | `which uv` で実際のパスを確認して unit に反映 |
| `status=200/CHDIR` | `WorkingDirectory` が存在しない | `pyproject.toml` があるリポジトリルートを指定 |
| SQLite の `Permission denied` | `User=` と `data/<character_id>/` の所有権不一致 | `chown -R <USER>:<USER> data/<character_id>/` |
| `Address already in use` | ポート衝突 | portを変更するか、不要なserviceを停止 |
| timerが発火しない | timerがstartされていない | `sudo systemctl enable --now fravenir-compact@<CHARACTER_ID>.timer` を実行 |

## 関連

- [admin_server.md](admin_server.md): 任意のAdmin UI運用
- [migrations.md](migrations.md): 古いDBを現在のスキーマへ追いつかせる保守用リファレンス
