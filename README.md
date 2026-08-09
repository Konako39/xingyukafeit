# 星语茶话屋 · Xingyu Tea House · 星語茶話屋

[中文](#中文) | [English](#english) | [日本語](#日本語)

## 中文

双击 [星语茶话屋.app](<./星语茶话屋.app>) 即可使用。

- 艾莉（低审查）与沙雅（官方）双人格，各自限定三个模型档位
- 整张模型档位卡均可点击；4B / 9B / 27B 在当前会话原地切换，不会新建对话
- 两套严格隔离、跨模型和跨会话的人格记忆池：用户对话、两人茶话、文件观察、屏幕观察都带时间进入各自池中
- 人格成长系统：每个人格拥有对自己、对主人、对对方、对世界四类可塑信念（带置信度与证据计数，永久保存并全部向量索引），内置学习循环在后台从新经历中反思提炼，动态生成可自我塑造的身份提示并保留版本历史；核心人格与质量底线始终受代码保护
- 类人学习机制：情绪带惯性演化并随时间向人格基线回落，且明显情绪会轻微影响情景回忆方向（事实检索不受影响）；反思会产生真实的好奇心问题并在合适时机自然问主人；从被认可的做事方式中沉淀可复用的“技能记忆”；每 12 次反思做一轮“睡眠式”深度巩固——把成簇的细碎认知抽象成更高层认知，并更新第一人称的自传体成长小传（原始信念归档保留，永不删除）
- 人格 Agent：直接用自然语言让艾莉或沙雅操作 Mac——日历日程增删查、提醒事项/待办增删查与完成、聚焦搜索找文件、打开应用/网址/文件、记系统备忘录、控制音乐；删除只在唯一匹配时执行，多条匹配会先让主人挑选，失败会如实说明（EventKit 助手，首次使用需允许日历/提醒事项访问）
- 三级零浪费触发：/命令与快捷芯片直接结构化执行（零模型成本）；“明天下午 3 点提醒我交作业”这类高置信句型由内置中文时间解析器确定性处理（也零成本）；只有真正有歧义的自然语言才花一次小模型调用，疑问语气一律交模型判断保证质量
- 无限记忆的规模化底座：向量 FP16 落盘（体积减半）+ 进程内矩阵缓存 + NumPy 批量打分（10 万条记忆检索约 12 ms，无 NumPy 自动回退标准库 C 级点积），质量零损失
- 任务栏快捷面板即主力入口：人格卡片实时显示成长心境，一键芯片查今日/本周日程与待办，流式输出不打断回看与选中，Esc 关闭，工具执行结果带状态徽标
- 同人格 14B 单条质量审计 + 同人格 27B 结构化记忆写作 + Qwen3 0.6B 向量检索；助手坏回复保留原文但不进入 RAG
- 艾莉 × 沙雅后台茶话室：空闲时在随性闲聊、回忆旧话题、只读观察文件之间切换，并把完整经过分别写回两人的记忆池
- 低频屏幕观察默认随机间隔 1–3 小时、每天最多 6 次；不切换全屏/Space，一次读取所有显示器，截图只在内存中传给模型并立即释放
- 日常短聊采用微信 / QQ 式自然短句，抑制强塞口癖、反复叫名字和客服式小作文
- 支持选择、拖入和粘贴图片
- 快速 / 均衡 / 深度三档质量模式，深度思考耗尽时自动恢复正文
- 回复复制、重新回答、继续生成、记忆命中和上下文占用显示
- 可调温度、Top P、重复惩罚、随机种子、常驻时间和输出上限
- 内置会话、模型、API、运行性能和存储管理
- 茶话室默认空闲 15 分钟后才可运行，每轮随机冷却 3–6 小时；开始后由两个人格自然决定何时结束，主人仅回来不会打断，真正发送消息或资源紧张时才让出
- 静默启动 Ollama，不弹终端或 Ollama 图形客户端
- 六模型自动压力测试覆盖指令遵循、多轮回忆、长上下文和深度推理

完整教程见 [本地大模型使用说明](<./文档/本地大模型使用说明.md>)。

## English

Double-click [星语茶话屋.app](<./星语茶话屋.app>) to get started.

- Two distinct personas—Aili (less restricted) and Shaya (official)—each with three dedicated model tiers
- Fully clickable model-tier cards; switch between 4B, 9B, and 27B models in the current session without starting a new conversation
- Two strictly isolated persona memory pools that persist across models and sessions; user conversations, persona-to-persona chats, file observations, and screen observations are timestamped and stored in the appropriate pool
- Persona growth system: each persona maintains adaptable beliefs about herself, the user, the other persona, and the world. Beliefs include confidence scores and evidence counts, persist permanently, and are fully vector-indexed. A background learning loop reflects on new experiences, extracts insights, generates an evolving identity prompt, and preserves its version history, while core personality traits and quality safeguards remain protected by code
- Human-like learning: emotions evolve with inertia and gradually return to each persona's baseline; strong emotions can subtly influence episodic recall without affecting factual retrieval. Reflection produces genuine curiosity questions that are asked naturally at suitable moments. Reusable “skill memories” are distilled from approaches the user approves. Every 12 reflections trigger sleep-like deep consolidation, which abstracts clusters of small insights into higher-level understanding and updates a first-person autobiographical growth narrative. Original beliefs are archived and never deleted
- Persona agents: ask Aili or Shaya in natural language to operate your Mac—create, delete, and search calendar events; create, complete, delete, and search reminders and to-dos; find files with Spotlight; open apps, URLs, and files; write system notes; and control music. Deletion occurs only for a unique match; if several items match, the persona asks you to choose. Failures are reported honestly. Calendar and Reminders access must be granted to the EventKit helper on first use
- Three-tier, zero-waste command routing: slash commands and shortcut chips execute directly as structured actions with no model cost; high-confidence phrases such as “Remind me to submit my homework at 3 PM tomorrow” are handled deterministically by the built-in Chinese time parser, also at no model cost; only genuinely ambiguous natural-language requests use a single small-model call, while questions are always delegated to the model to preserve response quality
- Scalable foundation for effectively unlimited memory: vectors are stored in FP16 to halve disk usage, cached as in-process matrices, and scored in batches with NumPy. Searching 100,000 memories takes about 12 ms; when NumPy is unavailable, the system automatically falls back to standard-library C-level dot products without sacrificing quality
- The taskbar quick panel is the primary entry point: persona cards show the current growth mood in real time; one-tap chips retrieve today's or this week's events and tasks; streaming responses do not interrupt scrolling or text selection; Esc closes the panel; and tool results include status badges
- Per-response quality auditing with the persona's 14B model, structured memory writing with the persona's 27B model, and vector retrieval with Qwen3 0.6B; poor assistant responses remain visible in the conversation but are excluded from RAG
- Aili × Shaya background tea room: while idle, the personas alternate among casual conversation, revisiting past topics, and read-only file observation, with the complete interaction written back to both memory pools separately
- Low-frequency screen observation uses a random interval of 1–3 hours by default, with a maximum of six observations per day. It never switches full-screen apps or Spaces, reads all displays in one pass, and sends screenshots to the model only in memory before releasing them immediately
- Everyday conversation uses natural, concise messaging inspired by WeChat and QQ, avoiding forced catchphrases, repetitive name usage, and customer-service-style essays
- Select, drag, or paste images into conversations
- Fast, Balanced, and Deep quality modes; when the reasoning budget is exhausted, the system automatically resumes the main response
- Copy responses, regenerate answers, continue generation, and view memory matches and context usage
- Adjustable temperature, Top P, repetition penalty, random seed, keep-alive duration, and output limit
- Built-in management for sessions, models, APIs, runtime performance, and storage
- The tea room becomes eligible to run after 15 minutes of inactivity and applies a random cooldown of 3–6 hours between rounds. Once started, the two personas decide naturally when to stop. The user's return alone does not interrupt them; they yield only when the user sends a message or system resources become constrained
- Ollama starts silently without opening a terminal window or the Ollama desktop client
- Automated stress testing across six models covers instruction following, multi-turn recall, long-context handling, and deep reasoning

For the complete guide, see the [Local Model Guide](<./文档/本地大模型使用说明.md>) (Chinese).

## 日本語

[星语茶话屋.app](<./星语茶话屋.app>) をダブルクリックすると起動できます。

- アイリ（制限が少ないモデル）とシャヤ（公式モデル）の二つの独立した人格。それぞれに専用の三つのモデル階層を用意
- モデル階層カード全体をクリック可能。新しい会話を作成せず、現在のセッション内で 4B / 9B / 27B を切り替え可能
- モデルやセッションをまたいで保持される、完全に分離された二つの人格メモリープール。ユーザーとの会話、二人の茶話、ファイル観察、画面観察がタイムスタンプ付きで各プールに保存される
- 人格成長システム：各人格は、自分自身、ユーザー、もう一人の人格、世界に対する可変の信念を持つ。信念には確信度と証拠数が付与され、永続保存とベクトル索引が行われる。バックグラウンドの学習ループが新しい経験を内省して知見を抽出し、成長に応じたアイデンティティプロンプトを生成して履歴を保存する一方、中核人格と品質基準はコードで常に保護される
- 人間らしい学習機構：感情は慣性を持って変化し、時間とともに人格ごとの基準値へ戻る。強い感情はエピソード記憶の想起方向にわずかに影響するが、事実検索には影響しない。内省から自然な好奇心の質問が生まれ、適切なタイミングでユーザーに尋ねる。評価された進め方から再利用可能な「スキル記憶」を形成する。内省 12 回ごとに睡眠のような深い統合を行い、細かな認識のまとまりを高次の理解へ抽象化し、一人称の自伝的な成長記録を更新する。元の信念はアーカイブされ、削除されない
- 人格 Agent：自然言語でアイリまたはシャヤに Mac の操作を依頼可能。カレンダー予定の作成・削除・検索、リマインダーや ToDo の作成・完了・削除・検索、Spotlight によるファイル検索、アプリ・URL・ファイルの起動、システムメモへの記録、音楽操作に対応する。削除は一致項目が一つの場合のみ実行し、複数ある場合はユーザーに選択を求める。失敗時には事実をそのまま伝える。初回利用時は EventKit ヘルパーへのカレンダーおよびリマインダーのアクセス許可が必要
- 無駄を省く三段階ルーティング：スラッシュコマンドとショートカットチップは構造化アクションとして直接実行され、モデルコストはゼロ。確信度の高い中国語の日時表現は内蔵パーサーが決定的に処理し、これもモデルコストはゼロ。真に曖昧な自然言語だけが小規模モデルを一度呼び出し、疑問文は品質維持のため常にモデルが判断する
- 実質無制限のメモリーを支える拡張可能な基盤：ベクトルを FP16 で保存して容量を半減し、プロセス内の行列キャッシュと NumPy の一括スコアリングを利用。10 万件のメモリー検索は約 12 ms。NumPy がない場合は標準ライブラリの C レベル内積へ自動的にフォールバックし、品質を維持する
- タスクバーのクイックパネルが主要な入口。人格カードに成長中の気分をリアルタイム表示し、ワンタップのチップで今日または今週の予定と ToDo を確認できる。ストリーミング出力はスクロールやテキスト選択を妨げず、Esc で閉じられ、ツールの実行結果にはステータスバッジが付く
- 同一人格の 14B モデルによる回答ごとの品質監査、27B モデルによる構造化メモリー記述、Qwen3 0.6B によるベクトル検索。品質の低いアシスタント回答は会話には残るが RAG には登録されない
- アイリ × シャヤのバックグラウンド茶話室：アイドル時に気軽な雑談、過去の話題の回想、読み取り専用のファイル観察を切り替え、やり取りの全内容を二人それぞれのメモリープールへ書き戻す
- 低頻度の画面観察は既定で 1〜3 時間のランダム間隔、1 日最大 6 回。フルスクリーンや Space を切り替えず、すべてのディスプレイを一度に読み取る。スクリーンショットはメモリー上でのみモデルへ渡し、直ちに解放する
- 日常会話は WeChat / QQ のような自然で短い文体を採用し、無理な口癖、名前の繰り返し、カスタマーサービス風の長文を抑制
- 画像の選択、ドラッグ＆ドロップ、貼り付けに対応
- Fast / Balanced / Deep の三つの品質モード。深い思考の予算を使い切ると本文生成を自動的に再開
- 回答のコピー、再生成、生成の続行、メモリー一致、コンテキスト使用量の表示に対応
- Temperature、Top P、繰り返しペナルティ、乱数シード、常駐時間、出力上限を調整可能
- セッション、モデル、API、実行性能、ストレージの管理機能を内蔵
- 茶話室は既定で 15 分間のアイドル後に実行可能となり、各ラウンド間には 3〜6 時間のランダムなクールダウンがある。開始後は二つの人格が自然に終了時点を決める。ユーザーが戻っただけでは中断せず、実際にメッセージが送信された場合、またはリソースが逼迫した場合にのみ処理を譲る
- Ollama を静かに起動し、ターミナルや Ollama のデスクトップクライアントを表示しない
- 六つのモデルを対象とした自動ストレステストで、指示追従、複数ターンの記憶、長文コンテキスト、深い推論を検証

完全なガイドは [ローカル大規模言語モデル利用ガイド](<./文档/本地大模型使用说明.md>) を参照してください（中国語）。
